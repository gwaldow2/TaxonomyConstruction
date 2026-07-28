"""Locate WHERE two prompt variants' predictions differ, structurally, instead of guessing.

Built for the open legacy_json question: full_json scores like full and shares legacy_json's
parser and output format, so parsing and format are excluded and the loss is in the prompt --
but not WHERE. This measures that directly from the two runs' recorded outputs.

It answers, in order:
  1. Is the gap RECALL or PRECISION? A variant that emits fewer, mostly-correct edges is
     abstaining; one that emits as many but wronger edges is mis-classifying. The fix is
     completely different, so this is the first split.
  2. Is it COVERAGE? Nodes and targets that produced no edge at all. If a variant simply
     answers "[]" for most targets, no amount of edge-quality analysis is relevant.
  3. WHERE in the hierarchy? Missed GT edges bucketed by depth, parent out-degree, and
     whether the endpoints are leaves -- so "it misses deep/rare links" or "it misses hub
     links" becomes a measurement.
  4. WHY are the divergent edges wrong? Re-uses fp_reason_analysis's LLM reasoner on the
     false positives each variant makes ALONE (--reason).

Inputs are what a run already writes: results/<ds>_<label>_edge_diagnostics.csv (every
predicted edge + its is_fp label) and results/GT_<ds>_eval.graphml.

    # record both runs, including the raw responses
    python main.py --method our_method --datasets WordNetFood --scale sub \
        --prompt_variant legacy_json full_json --debug_parse --results_file variants.json

    python variant_comparison.py --a legacy_json --b full_json
    python variant_comparison.py --a legacy_json --b full_json --reason      # + LLM on the FPs
    python variant_comparison.py --a legacy_json --b full_json --graph WordNetFood_SUB

Writes results/variant_diff_<a>_vs_<b>.csv and, with --graph, vis/variant_graph_<ds>.png.
"""

import os
import csv
import glob
import json
import argparse
from collections import defaultdict, Counter

import networkx as nx

RESULTS_DIR = "results"
VIS_DIR = "vis"


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def find_diag(results_dir, variant):
    """-> {dataset: path} for edge-diagnostics CSVs belonging to one --prompt_variant."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*_edge_diagnostics.csv"))):
        base = os.path.basename(p)
        if f"[{variant}]" not in base and f"_{variant}_" not in base:
            continue
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            out[rows[0]["dataset"]] = p
    return out


def load_edges(path):
    """-> {(parent, child): is_fp} exactly as the run recorded them."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["parent"], r["child"]): int(r["is_fp"]) for r in csv.DictReader(f)}


def load_gt(results_dir, dataset):
    p = os.path.join(results_dir, f"GT_{dataset}_eval.graphml")
    if not os.path.exists(p):
        return None
    G = nx.DiGraph(nx.read_graphml(p))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


# --------------------------------------------------------------------------------------
# structural analysis
# --------------------------------------------------------------------------------------

def pr_f1(edges):
    """Precision from the run's own is_fp labels (no re-derivation of correctness here)."""
    n = len(edges)
    tp = sum(1 for v in edges.values() if v == 0)
    return tp, n, (tp / n if n else 0.0)


def recall_vs_gt(edges, G_gt):
    """-> (recovered_gt_edges, total_gt_edges, missed). Recall measured on GT DIRECT edges,
    which is what makes 'which links did it miss' interpretable; the headline Cond_Clos
    metric uses closures on both sides and is reported by the run itself."""
    from evaluator import gt_closure_term_pairs, edge_is_correct
    pred = nx.DiGraph()
    pred.add_edges_from(edges)
    pred_closed = nx.transitive_closure(pred) if pred.number_of_nodes() else nx.DiGraph()
    pred_pairs = gt_closure_term_pairs(pred_closed) if pred_closed.number_of_nodes() else set()
    recovered, missed = [], []
    for (p, c) in G_gt.edges():
        (recovered if edge_is_correct(p, c, pred_pairs) else missed).append((p, c))
    return recovered, list(G_gt.edges()), missed


def hierarchy_stats(G_gt):
    """Per-node depth from the nearest root and out-degree, for locating misses."""
    roots = [n for n, d in G_gt.in_degree() if d == 0] or list(G_gt.nodes())[:1]
    depth = {}
    for r in roots:
        for n, d in nx.single_source_shortest_path_length(G_gt, r).items():
            depth[n] = min(depth.get(n, 10 ** 6), d)
    return depth, dict(G_gt.out_degree())


def describe_misses(missed, G_gt, label):
    """Where in the structure the missed GT edges sit."""
    if not missed:
        print(f"    {label}: no missed GT edges")
        return
    depth, outdeg = hierarchy_stats(G_gt)
    leaves = {n for n, d in G_gt.out_degree() if d == 0}
    dbuck = Counter(min(depth.get(p, -1), 6) for p, _ in missed)
    n_leafchild = sum(1 for _, c in missed if c in leaves)
    hub = sum(1 for p, _ in missed if outdeg.get(p, 0) >= 5)
    print(f"    {label}: {len(missed)} missed GT edges")
    print(f"      by parent depth: " + " ".join(f"d{d}={n}" for d, n in sorted(dbuck.items())))
    print(f"      child is a leaf:  {n_leafchild}/{len(missed)} ({100*n_leafchild/len(missed):.0f}%)")
    print(f"      parent is a hub (out-deg>=5): {hub}/{len(missed)} ({100*hub/len(missed):.0f}%)")


def coverage(edges, G_gt):
    """How much of the vocabulary the variant connected at all -- the abstention check."""
    touched = {n for e in edges for n in e}
    return len(touched), G_gt.number_of_nodes()


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def compare_dataset(ds, ea, eb, G_gt, na, nb):
    print(f"\n{'='*78}\n### {ds}\n{'='*78}")
    rows = []

    for name, e in ((na, ea), (nb, eb)):
        tp, n, prec = pr_f1(e)
        rec_e, gt_e, missed = recall_vs_gt(e, G_gt)
        cov, tot = coverage(e, G_gt)
        print(f"\n  [{name}]")
        print(f"    edges predicted : {n}")
        print(f"    correct (TP)    : {tp}   precision={prec:.3f}")
        print(f"    GT edges covered: {len(rec_e)}/{len(gt_e)}   recall={len(rec_e)/max(1,len(gt_e)):.3f}")
        print(f"    vocabulary touched: {cov}/{tot} nodes ({100*cov/max(1,tot):.0f}%)")
        rows.append((name, n, tp, prec, len(rec_e), len(gt_e), cov, tot, missed))

    (_, na_n, na_tp, na_p, na_r, gt_n, na_cov, tot, na_missed) = rows[0]
    (_, nb_n, nb_tp, nb_p, nb_r, _, nb_cov, _, nb_missed) = rows[1]

    print(f"\n  --- where the gap comes from ---")
    print(f"    edge count      {na}={na_n}  vs  {nb}={nb_n}   ({na_n-nb_n:+d})")
    print(f"    precision       {na}={na_p:.3f} vs {nb}={nb_p:.3f}  ({na_p-nb_p:+.3f})")
    print(f"    GT recovered    {na}={na_r}    vs {nb}={nb_r}      ({na_r-nb_r:+d} of {gt_n})")
    print(f"    vocab touched   {na}={na_cov}  vs {nb}={nb_cov}    ({na_cov-nb_cov:+d} of {tot})")

    # The diagnosis: which axis actually moved.
    if nb_n and na_n / max(1, nb_n) < 0.6:
        print(f"\n    >> {na} emits {100*(1-na_n/nb_n):.0f}% FEWER edges. This is ABSTENTION/COVERAGE,")
        print(f"       not edge quality -- it is declining to answer, not answering wrongly.")
    elif na_p < nb_p - 0.05:
        print(f"\n    >> {na} emits a similar number of edges but is {nb_p-na_p:.3f} LESS PRECISE.")
        print(f"       It is asserting relations at the same rate and getting them wrong.")
    else:
        print(f"\n    >> neither edge count nor precision differs much; the gap is elsewhere "
              f"(check the per-dataset numbers above).")

    print(f"\n  --- structural location of missed GT edges ---")
    describe_misses(na_missed, G_gt, na)
    describe_misses(nb_missed, G_gt, nb)

    sa, sb = set(ea), set(eb)
    only_a, only_b, both = sa - sb, sb - sa, sa & sb
    fp = lambda s, e: sum(1 for x in s if e[x] == 1)
    print(f"\n  --- edge set overlap ---")
    print(f"    only {na}: {len(only_a):4d}  ({fp(only_a, ea)} FP)")
    print(f"    only {nb}: {len(only_b):4d}  ({fp(only_b, eb)} FP)")
    print(f"    in both  : {len(both):4d}  ({fp(both, ea)} FP)")

    diff = []
    for e in sorted(only_a):
        diff.append({"dataset": ds, "edge_parent": e[0], "edge_child": e[1],
                     "present_in": na, "absent_from": nb, "is_fp": ea[e]})
    for e in sorted(only_b):
        diff.append({"dataset": ds, "edge_parent": e[0], "edge_child": e[1],
                     "present_in": nb, "absent_from": na, "is_fp": eb[e]})
    for (p, c) in na_missed:
        if (p, c) not in sb:
            continue
        diff.append({"dataset": ds, "edge_parent": p, "edge_child": c,
                     "present_in": f"GT+{nb}", "absent_from": na, "is_fp": 0})
    return diff


# --------------------------------------------------------------------------------------
# LLM reasoning over the divergent false positives
# --------------------------------------------------------------------------------------

def reason_over(diff, G_gt_by_ds, args, variant):
    """Classify the FPs a variant makes ALONE, with the reasoner already built for this."""
    from fp_reason_analysis import (neighbor_maps, gt_context, classify_fp,
                                    openai_responder, BUCKETS)
    from evaluator import gt_closure_term_pairs
    respond = openai_responder(args.base_url, args.api_key, args.model)
    counts, out = Counter(), []
    todo = [d for d in diff if d["present_in"] == variant and d["is_fp"] == 1][:args.max_reason]
    if not todo:
        print(f"    ({variant} makes no unique false positives to explain)")
        return counts, out
    print(f"    classifying {len(todo)} {variant}-only false positive(s)...")
    ctx_cache = {}
    for d in todo:
        ds = d["dataset"]
        if ds not in ctx_cache:
            ctx_cache[ds] = neighbor_maps(gt_closure_term_pairs(G_gt_by_ds[ds]))
        anc, desc = ctx_cache[ds]
        bucket, reason = classify_fp(respond, d["edge_parent"], d["edge_child"],
                                     gt_context(d["edge_parent"], d["edge_child"], anc, desc))
        counts[bucket] += 1
        out.append({**d, "bucket": bucket, "reason": reason})
    total = sum(counts.values()) or 1
    for b, n in counts.most_common():
        print(f"      {b:22s} {n:4d}  ({100*n/total:.0f}%)")
    return counts, out


# --------------------------------------------------------------------------------------
# structural graph picture
# --------------------------------------------------------------------------------------

def draw_graph(ds, ea, eb, G_gt, na, nb, out_path):
    """GT skeleton with each GT edge coloured by which variant recovered it, so the
    structural pattern of the loss is visible rather than inferred from aggregates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sa, sb = set(ea), set(eb)
    status, colors = {}, {"both": "#2a9d3f", nb: "#4C72B0", na: "#d1a000", "neither": "#cccccc"}
    for e in G_gt.edges():
        in_a, in_b = e in sa, e in sb
        status[e] = "both" if (in_a and in_b) else (nb if in_b else (na if in_a else "neither"))

    pos = nx.nx_agraph.graphviz_layout(G_gt, prog="dot") if _has_graphviz() else \
        nx.spring_layout(G_gt, seed=42, k=0.9)
    fig, ax = plt.subplots(figsize=(20, 14))
    for key, col in colors.items():
        es = [e for e in G_gt.edges() if status[e] == key]
        if es:
            nx.draw_networkx_edges(G_gt, pos, edgelist=es, edge_color=col, ax=ax,
                                   width=2.0 if key != "neither" else 0.8,
                                   alpha=0.9 if key != "neither" else 0.45, arrows=False)
    nx.draw_networkx_nodes(G_gt, pos, node_size=90, node_color="white",
                           edgecolors="#444444", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G_gt, pos, font_size=5,
                            labels={n: str(n).split("|")[0][:18] for n in G_gt.nodes()}, ax=ax)
    counts = Counter(status.values())
    ax.legend(handles=[Line2D([0], [0], color=c, lw=3,
                              label=f"{k}  ({counts.get(k,0)} GT edges)") for k, c in colors.items()],
              loc="upper left", fontsize=11)
    ax.set_title(f"{ds}: which GT edges each variant recovered\n"
                 f"grey = missed by both, gold = {na} only, blue = {nb} only, green = both",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _has_graphviz():
    try:
        import pygraphviz  # noqa: F401
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Compare two prompt variants' recorded outputs.")
    ap.add_argument("--a", default="legacy_json", help="Variant under investigation.")
    ap.add_argument("--b", default="full_json", help="Reference variant.")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--graph", default=None, metavar="DATASET",
                    help="Draw the GT graph for this dataset (e.g. WordNetFood_SUB).")
    ap.add_argument("--reason", action="store_true",
                    help="Run the fp_reason LLM classifier over each variant's UNIQUE false positives.")
    ap.add_argument("--max_reason", type=int, default=40)
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="woohoo")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    da, db = find_diag(args.results_dir, args.a), find_diag(args.results_dir, args.b)
    shared = sorted(set(da) & set(db))
    if not shared:
        raise SystemExit(
            f"[!] no dataset has diagnostics for BOTH '{args.a}' and '{args.b}' in {args.results_dir}/.\n"
            f"    found for {args.a}: {sorted(da) or 'none'}\n"
            f"    found for {args.b}: {sorted(db) or 'none'}\n"
            f"    Produce them with: main.py --method our_method --prompt_variant {args.a} {args.b}")
    print(f"[*] comparing '{args.a}' vs '{args.b}' over {len(shared)} dataset(s): {', '.join(shared)}")

    all_diff, gts = [], {}
    for ds in shared:
        G_gt = load_gt(args.results_dir, ds)
        if G_gt is None:
            print(f"    [!] no GT graph for {ds} -- skipping"); continue
        gts[ds] = G_gt
        all_diff += compare_dataset(ds, load_edges(da[ds]), load_edges(db[ds]), G_gt, args.a, args.b)

    out_csv = args.out_csv or os.path.join(args.results_dir, f"variant_diff_{args.a}_vs_{args.b}.csv")
    if all_diff:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_diff[0]))
            w.writeheader(); w.writerows(all_diff)
        print(f"\n[*] wrote {len(all_diff)} divergent edges -> {out_csv}")

    if args.reason and all_diff:
        print(f"\n{'='*78}\n### why are the divergent predictions wrong? (LLM reasoner)\n{'='*78}")
        rows = []
        for v in (args.a, args.b):
            print(f"\n  [{v}-only false positives]")
            _, r = reason_over(all_diff, gts, args, v)
            rows += r
        if rows:
            rp = out_csv.replace(".csv", "_reasons.csv")
            with open(rp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            print(f"\n[*] wrote {len(rows)} classified edges -> {rp}")

    if args.graph:
        if args.graph not in gts:
            raise SystemExit(f"[!] {args.graph} not among compared datasets: {', '.join(gts)}")
        out = os.path.join(VIS_DIR, f"variant_graph_{args.graph}.png")
        draw_graph(args.graph, load_edges(da[args.graph]), load_edges(db[args.graph]),
                   gts[args.graph], args.a, args.b, out)
        print(f"[*] wrote {out}")


if __name__ == "__main__":
    main()
