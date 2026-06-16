"""
implied_edge_analysis.py  --  Post-hoc "implied edge" analysis (Stage 2b).

Runs ENTIRELY on saved benchmark outputs in ./results/  -- it does NOT re-run any
method or touch the LLM server.

Motivation
----------
Our method's advantage shows up under TRANSITIVE CLOSURE evaluation: recall jumps,
precision dips slightly. The closure edge-set splits cleanly into two disjoint buckets:

    reduction edges  = transitive_reduction(GT)         -> DIRECT  parent->child links
    implied  edges   = closure(GT) \\ reduction(GT)      -> INDIRECT ancestor->descendant links

Because a deep taxonomy has |implied| >> |reduction|, the headline closure recall is
dominated by the implied bucket. This script decomposes closure recall into those two
buckets PER METHOD (the metric your advisor suggested) and additionally stratifies
implied-edge recall by GT hop-distance -- which is what actually explains *why* our
method wins: single-parent methods recover an indirect edge (a..c) only if every link
on the a->...->c chain is correct, so their recall decays with hop-distance, while a
method that states ancestry directly stays flat.

What it reads (all already on disk after a normal `python main.py` run):
    ./results/GT_<dataset>_eval.graphml                      (GT graph)
    ./results/<dataset>_<method>_condensed_closure.txt       (recovered GT edges + FPs)
    ./results/<dataset>_<method>_condensed_reduction.txt     (predicted direct edges; for --leverage)

What it writes (to --out_dir, default ./implied_analysis):
    implied_edge_summary.csv         one row per (dataset, method): bucketed recall + counts
    implied_recall_by_hop.csv        recall stratified by GT hop-distance
    precision_clawback_candidates.csv (only with --leverage) high-impact FP edges to prune
    *.png                            (only with --plot)

Usage:
    python implied_edge_analysis.py                 # summary + per-hop CSVs
    python implied_edge_analysis.py --plot          # + figures
    python implied_edge_analysis.py --leverage      # + precision-clawback candidate edges
    python implied_edge_analysis.py --selftest      # validate logic on a synthetic example
"""

import os
import re
import glob
import argparse
import networkx as nx
import pandas as pd

# ----------------------------------------------------------------------------
# Parsing: mirror the report format written by evaluator.compute_and_save_metrics
# (set_overlap match_type, used for the condensed graphs).
# ----------------------------------------------------------------------------
RE_MATCHES_GT = re.compile(r"matches GT:\s*\((.*?)\s*->\s*(.*?)\)")          # recovered GT edge
RE_TP_PRED    = re.compile(r"\[TP\]\s*PRED:\s*\((.*?)\s*->\s*(.*?)\)")        # predicted edge (TP)
RE_FP         = re.compile(r"\[FP\]\s+(.*?)\s*->\s*(.*)$")                    # predicted edge (FP)
RE_HEADER     = re.compile(r"Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)")


def parse_closure_report(path):
    """Return (recovered_gt_edges, predicted_fp_edges, header_metrics) from a *_condensed_closure.txt."""
    recovered_gt = set()
    pred_fp = set()
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if header is None:
                hm = RE_HEADER.search(line)
                if hm:
                    header = {"Precision": float(hm.group(1)),
                              "Recall": float(hm.group(2)),
                              "F1": float(hm.group(3))}
            m = RE_MATCHES_GT.search(line)
            if m:
                recovered_gt.add((m.group(1).strip(), m.group(2).strip()))
                continue
            m = RE_FP.search(line)
            if m:
                pred_fp.add((m.group(1).strip(), m.group(2).strip()))
    return recovered_gt, pred_fp, header


def parse_reduction_report(path):
    """Return the predicted DIRECT edge set (TP-pred + FP) from a *_condensed_reduction.txt.
    Returns None if the file records a non-DAG failure (no reduction available)."""
    if not os.path.exists(path):
        return None
    pred_edges = set()
    saw_failed = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "FAILED" in line:
                saw_failed = True
            m = RE_TP_PRED.search(line)
            if m:
                pred_edges.add((m.group(1).strip(), m.group(2).strip()))
                continue
            m = RE_FP.search(line.strip())
            if m:
                pred_edges.add((m.group(1).strip(), m.group(2).strip()))
    if saw_failed and not pred_edges:
        return None
    return pred_edges


# ----------------------------------------------------------------------------
# GT graph views: reduction / closure / implied buckets + hop distances.
# ----------------------------------------------------------------------------
def build_gt_views(G_gt):
    G = nx.DiGraph(G_gt)
    G.remove_edges_from(nx.selfloop_edges(G))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    # GT should already be a DAG; guard defensively so analysis never crashes.
    while not nx.is_directed_acyclic_graph(G):
        cyc = next(nx.simple_cycles(G))
        G.remove_edge(cyc[-1], cyc[0])

    red = nx.transitive_reduction(G)
    clos = nx.transitive_closure(G)
    red_edges = set(red.edges())
    clos_edges = set(clos.edges())
    implied_edges = clos_edges - red_edges          # disjoint from red_edges by construction

    # Hop distance measured on the REDUCTION (true hierarchical depth between a and c).
    dist = dict(nx.all_pairs_shortest_path_length(red))
    hop = {e: dist.get(e[0], {}).get(e[1]) for e in implied_edges}
    return red_edges, clos_edges, implied_edges, hop


def _safe_recall(target, recovered):
    return (len(target & recovered) / len(target)) if target else float("nan")


def analyze_dataset_method(G_gt_views, recovered):
    red_edges, clos_edges, implied_edges, hop = G_gt_views
    rec_implied = implied_edges & recovered
    row = {
        "n_reduction": len(red_edges),
        "n_implied": len(implied_edges),
        "n_closure": len(clos_edges),
        "recall_reduction": _safe_recall(red_edges, recovered),
        "recall_implied": _safe_recall(implied_edges, recovered),
        "recall_closure": _safe_recall(clos_edges, recovered),
    }
    # Per-hop implied recall.
    by_hop = {}
    buckets = {}
    for e in implied_edges:
        d = hop.get(e)
        if d is None:
            continue
        buckets.setdefault(d, set()).add(e)
    for d, edges in sorted(buckets.items()):
        by_hop[d] = {
            "support": len(edges),
            "recovered": len(edges & recovered),
            "recall": _safe_recall(edges, recovered),
        }
    return row, by_hop


# ----------------------------------------------------------------------------
# Precision clawback: rank predicted DIRECT edges by closure "leverage" --
# how many ancestor/descendant pairs each edge is responsible for -- and flag
# the high-leverage ones that are NOT supported by the GT closure. Removing a
# single high-leverage wrong edge eliminates many false-positive closure pairs
# at minimal recall cost.
# ----------------------------------------------------------------------------
def leverage_candidates(pred_direct_edges, gt_closure_edges, top_n=15):
    if not pred_direct_edges:
        return []
    Gp = nx.DiGraph()
    Gp.add_edges_from(pred_direct_edges)
    if not nx.is_directed_acyclic_graph(Gp):
        while not nx.is_directed_acyclic_graph(Gp):
            cyc = next(nx.simple_cycles(Gp))
            Gp.remove_edge(cyc[-1], cyc[0])
    rows = []
    for (a, c) in Gp.edges():
        n_anc = len(nx.ancestors(Gp, a)) + 1       # a + everything above a
        n_desc = len(nx.descendants(Gp, c)) + 1    # c + everything below c
        leverage = n_anc * n_desc                  # closure pairs forced through this edge
        supported = (a, c) in gt_closure_edges     # is this even a real ancestor relation?
        rows.append({
            "pred_edge": f"{a} -> {c}",
            "leverage": leverage,
            "gt_supported": supported,
            "n_ancestors": n_anc,
            "n_descendants": n_desc,
        })
    # Suspect = high leverage AND not supported by GT closure -> prime pruning target.
    rows.sort(key=lambda r: (not r["gt_supported"], r["leverage"]), reverse=True)
    suspect = [r for r in rows if not r["gt_supported"]]
    suspect.sort(key=lambda r: r["leverage"], reverse=True)
    return suspect[:top_n]


# ----------------------------------------------------------------------------
# Discovery + driver
# ----------------------------------------------------------------------------
def discover(results_dir):
    """Yield (dataset, method_suffix, closure_txt, reduction_txt, gt_graphml)."""
    for gt_path in sorted(glob.glob(os.path.join(results_dir, "GT_*_eval.graphml"))):
        base = os.path.basename(gt_path)
        dataset = base[len("GT_"):-len("_eval.graphml")]
        for clos_txt in sorted(glob.glob(os.path.join(results_dir, f"{dataset}_*_condensed_closure.txt"))):
            cbase = os.path.basename(clos_txt)
            method = cbase[len(dataset) + 1:-len("_condensed_closure.txt")]
            red_txt = os.path.join(results_dir, f"{dataset}_{method}_condensed_reduction.txt")
            yield dataset, method, clos_txt, red_txt, gt_path


def run(results_dir, out_dir, do_leverage=False):
    os.makedirs(out_dir, exist_ok=True)
    gt_cache = {}
    summary_rows = []
    hop_rows = []
    clawback_rows = []

    found = False
    for dataset, method, clos_txt, red_txt, gt_path in discover(results_dir):
        found = True
        if gt_path not in gt_cache:
            gt_cache[gt_path] = build_gt_views(nx.read_graphml(gt_path))
        views = gt_cache[gt_path]

        recovered_gt, pred_fp, header = parse_closure_report(clos_txt)
        row, by_hop = analyze_dataset_method(views, recovered_gt)
        row = {"Dataset": dataset, "Method": method, **row}
        if header:
            row["closure_F1_reported"] = header["F1"]
        summary_rows.append(row)

        for d, info in by_hop.items():
            hop_rows.append({"Dataset": dataset, "Method": method, "hop_distance": d, **info})

        if do_leverage:
            pred_direct = parse_reduction_report(red_txt)
            _, gt_closure, _, _ = views
            for c in leverage_candidates(pred_direct or set(), gt_closure):
                clawback_rows.append({"Dataset": dataset, "Method": method, **c})

    if not found:
        print(f"[!] No GT_*_eval.graphml + *_condensed_closure.txt pairs found in '{results_dir}'.")
        print("    Point --results_dir at the folder you pulled over (where main.py wrote outputs).")
        return None

    summary = pd.DataFrame(summary_rows).sort_values(["Dataset", "recall_implied"],
                                                     ascending=[True, False])
    hop_df = pd.DataFrame(hop_rows).sort_values(["Dataset", "Method", "hop_distance"])
    summary_path = os.path.join(out_dir, "implied_edge_summary.csv")
    hop_path = os.path.join(out_dir, "implied_recall_by_hop.csv")
    summary.to_csv(summary_path, index=False)
    hop_df.to_csv(hop_path, index=False)
    print(f" -> wrote {summary_path}  ({len(summary)} rows)")
    print(f" -> wrote {hop_path}  ({len(hop_df)} rows)")

    if do_leverage and clawback_rows:
        claw_path = os.path.join(out_dir, "precision_clawback_candidates.csv")
        pd.DataFrame(clawback_rows).to_csv(claw_path, index=False)
        print(f" -> wrote {claw_path}  ({len(clawback_rows)} candidate edges)")

    # Aggregated view: mean bucketed recall per method across datasets.
    print("\n" + "=" * 78)
    print("  MEAN RECALL BY BUCKET, PER METHOD (averaged across datasets)")
    print("  reduction = direct links | implied = closure-minus-reduction (indirect)")
    print("=" * 78)
    agg = (summary.groupby("Method")[["recall_reduction", "recall_implied", "recall_closure"]]
                  .mean().sort_values("recall_implied", ascending=False))
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)
    return summary, hop_df


# ----------------------------------------------------------------------------
# Optional plots
# ----------------------------------------------------------------------------
def make_plots(summary, hop_df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Mean reduction vs implied recall per method (grouped bars).
    agg = summary.groupby("Method")[["recall_reduction", "recall_implied"]].mean()
    agg = agg.sort_values("recall_implied", ascending=False)
    ax = agg.plot(kind="bar", figsize=(max(8, len(agg) * 1.2), 6))
    ax.set_ylabel("Mean recall")
    ax.set_title("Direct (reduction) vs Indirect (implied) edge recall, by method")
    ax.legend(["Reduction (direct)", "Implied (indirect)"])
    plt.tight_layout()
    p1 = os.path.join(out_dir, "implied_vs_reduction_recall.png")
    plt.savefig(p1, dpi=150); plt.close()

    # 2) Implied recall vs hop-distance, per method (the "why" figure).
    plt.figure(figsize=(9, 6))
    g = hop_df.groupby(["Method", "hop_distance"]).apply(
        lambda d: d["recovered"].sum() / d["support"].sum() if d["support"].sum() else float("nan")
    ).reset_index(name="recall")
    for method, sub in g.groupby("Method"):
        sub = sub.sort_values("hop_distance")
        plt.plot(sub["hop_distance"], sub["recall"], marker="o", label=method)
    plt.xlabel("GT hop-distance (ancestor depth between a and c)")
    plt.ylabel("Implied-edge recall (pooled across datasets)")
    plt.title("Indirect-edge recall vs hop-distance\nsingle-parent methods decay with depth; direct-ancestry methods stay flat")
    plt.ylim(0, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "implied_recall_by_hop.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f" -> wrote {p1}\n -> wrote {p2}")


# ----------------------------------------------------------------------------
# Self-test (no real data needed): synthetic GT chain + fabricated report.
# ----------------------------------------------------------------------------
def selftest():
    import tempfile
    print("Running self-test on a synthetic example...")
    # GT: a -> b -> c -> d  (chain) and a -> e (branch)
    G = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "e")])
    red, clos, implied, hop = build_gt_views(G)
    # reduction = {a-b, b-c, c-d, a-e};  implied = {a-c, a-d, b-d, (a-e none)}
    assert red == {("a", "b"), ("b", "c"), ("c", "d"), ("a", "e")}, red
    assert implied == {("a", "c"), ("a", "d"), ("b", "d")}, implied
    assert hop[("a", "c")] == 2 and hop[("a", "d")] == 3 and hop[("b", "d")] == 2, hop

    # A "single-parent-ish" method that recovered direct links + the 2-hop ones
    # but MISSED the 3-hop a->d (chain broken downstream).
    recovered = {("a", "b"), ("b", "c"), ("c", "d"), ("a", "e"), ("a", "c"), ("b", "d")}
    row, by_hop = analyze_dataset_method((red, clos, implied, hop), recovered)
    assert abs(row["recall_reduction"] - 1.0) < 1e-9, row
    assert abs(row["recall_implied"] - 2/3) < 1e-9, row     # got a-c, b-d; missed a-d
    assert by_hop[2]["recall"] == 1.0 and by_hop[3]["recall"] == 0.0, by_hop
    print("  build_gt_views / analyze_dataset_method  OK")

    # Report parsing round-trip.
    with tempfile.TemporaryDirectory() as td:
        clos_txt = os.path.join(td, "DS_M_condensed_closure.txt")
        with open(clos_txt, "w", encoding="utf-8") as f:
            f.write("Precision: 0.8000 | Recall: 0.5000 | F1: 0.6154\n")
            f.write("  [TP] PRED: (a -> c)\n        matches GT: (a -> c)\n")
            f.write("  [TP] PRED: (a -> b)\n        matches GT: (a -> b)\n")
            f.write("  [FP] x -> y\n")
        rec, fp, hdr = parse_closure_report(clos_txt)
        assert rec == {("a", "c"), ("a", "b")}, rec
        assert fp == {("x", "y")}, fp
        assert hdr["F1"] == 0.6154, hdr
    print("  parse_closure_report  OK")

    # Leverage: a wrong high edge a->c with a deep subtree should rank top.
    pred = {("a", "b"), ("b", "c"), ("c", "d"), ("root", "a")}
    cand = leverage_candidates(pred, gt_closure_edges=set(), top_n=5)
    assert cand and cand[0]["leverage"] >= cand[-1]["leverage"]
    print("  leverage_candidates  OK")
    print("ALL SELF-TESTS PASSED.")


def main():
    ap = argparse.ArgumentParser(description="Implied-edge (closure-minus-reduction) analysis from saved results.")
    ap.add_argument("--results_dir", default="./results", help="Folder with GT_*.graphml and *_condensed_*.txt")
    ap.add_argument("--out_dir", default="./implied_analysis", help="Where to write CSVs/PNGs")
    ap.add_argument("--plot", action="store_true", help="Also render figures")
    ap.add_argument("--leverage", action="store_true", help="Also emit precision-clawback candidate edges")
    ap.add_argument("--selftest", action="store_true", help="Validate logic on a synthetic example and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    out = run(args.results_dir, args.out_dir, do_leverage=args.leverage)
    if out and args.plot:
        make_plots(out[0], out[1], args.out_dir)


if __name__ == "__main__":
    main()
