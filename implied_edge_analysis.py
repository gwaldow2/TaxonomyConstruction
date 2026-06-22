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
    implied_recall_by_hop.csv        recall by GT hop-distance (hop 1 = direct, >=2 = indirect) + Scale (SUB/FULL)
    synonym_recall_by_count.csv      precision/recall/F1 vs #synonyms merged into a node (per dataset/method)
    synonym_pr_pooled.csv            precision/recall/F1 vs #synonyms, pooled over taxonomies WITH synsets
    synonym_pr_optimum.csv           per-method F1-optimal synonym count (balances precision & recall)
    synonym_recall_summary.csv       per-method: recall & precision of 1-surface vs multi-synonym nodes + lift
    precision_clawback_candidates.csv (only with --leverage) high-impact FP edges to prune
    recall_by_hop_<SUB|FULL>.png     (only with --plot) hop-recall split by dataset size
    *.png                            (only with --plot; incl. synonym_pr_curves.png)

Usage:
    python implied_edge_analysis.py                 # summary + per-hop + synonym CSVs
    python implied_edge_analysis.py --plot          # + figures (support-weighted hop curve, synonym P/R/F1 panels)
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
    """Parse a *_condensed_closure.txt report.

    Returns (recovered_gt_edges, pred_tp_edges, pred_fp_edges, header_metrics):
      - recovered_gt_edges : GT closure edges that were matched   (the "matches GT:" side of a TP)
      - pred_tp_edges      : PREDICTED edges credited as correct  (the "[TP] PRED:" side of a TP)
      - pred_fp_edges      : PREDICTED edges that matched nothing  ([FP])
    pred_tp ∪ pred_fp = all predicted closure edges (the precision denominator)."""
    recovered_gt = set()
    pred_tp = set()
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
            m = RE_TP_PRED.search(line)
            if m:
                pred_tp.add((m.group(1).strip(), m.group(2).strip()))
                continue
            m = RE_FP.search(line)
            if m:
                pred_fp.add((m.group(1).strip(), m.group(2).strip()))
    return recovered_gt, pred_tp, pred_fp, header


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

    # Hop distance measured on the REDUCTION graph (true hierarchical depth).
    # Computed for ALL closure edges: a direct/reduction edge has hop == 1, an
    # implied/indirect edge has hop >= 2. Including hop 1 lets the per-hop curve
    # compare direct-edge recall against the indirect (hop>=2) recall.
    dist = dict(nx.all_pairs_shortest_path_length(red))
    hop = {e: dist.get(e[0], {}).get(e[1]) for e in clos_edges}
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
    # Per-hop recall over ALL closure edges: hop 1 = direct (reduction) edges,
    # hop >= 2 = indirect (implied) edges.
    by_hop = {}
    buckets = {}
    for e in clos_edges:
        d = hop.get(e)
        if d is None:
            continue
        buckets.setdefault(d, set()).add(e)
    for d, edges in sorted(buckets.items()):
        by_hop[d] = {
            "support": len(edges),
            "recovered": len(edges & recovered),
            "recall": _safe_recall(edges, recovered),
            "kind": "direct" if d == 1 else "indirect",
        }
    return row, by_hop


# ----------------------------------------------------------------------------
# Synonym-richness analysis: does a node having MORE merged synonyms raise the
# recall of edges incident to it? When several synonymous terms are condensed
# into one node, a method has more surface forms to latch onto when asserting an
# ancestor/descendant relationship -- and (under set_overlap matching) more ways
# for a predicted edge to be credited. This isolates that effect.
# ----------------------------------------------------------------------------
def parse_synonyms(node_str):
    """Return the surface forms merged into one cluster node.

    Mirrors data_manager.parse_lemma_format so the script stays standalone
    (runnable on just the saved outputs, with no project imports)."""
    s = str(node_str).strip().lower()
    m = re.match(r'^([^(]+)\s*\((.*)\)$', s)
    if m:
        primary = m.group(1).strip()
        syns = [t.strip() for t in m.group(2).split(',') if t.strip()]
        if syns and syns[0] == primary:
            return syns
    if '|' in s:
        return [t.strip() for t in s.split('|') if t.strip()]
    return [s]


def synonym_node_rows(clos_edges, recovered, pred_tp, pred_fp):
    """One row per node: synonym count vs the RECALL and PRECISION of edges touching it.

    Every edge is attributed to BOTH endpoints, so for a node n:
      recall    = recovered GT closure edges touching n / GT closure edges touching n
      precision = correct predicted edges touching n  / predicted edges touching n
    where predicted edges = pred_tp (credited) + pred_fp (wrong). node_depth (number
    of GT ancestors) is reported so a later pass can check whether any synonym effect
    is just a depth/generality confound."""
    from collections import defaultdict
    gt_total = defaultdict(int)        # GT closure edges incident to node  (recall denom)
    gt_recovered = defaultdict(int)
    pred_total = defaultdict(int)      # predicted closure edges incident    (precision denom)
    pred_correct = defaultdict(int)
    anc_count = defaultdict(int)
    nodes = set()

    for e in clos_edges:
        a, c = e
        nodes.add(a); nodes.add(c)
        gt_total[a] += 1; gt_total[c] += 1
        anc_count[c] += 1                       # a is an ancestor of c
        if e in recovered:
            gt_recovered[a] += 1; gt_recovered[c] += 1
    for e in pred_tp:
        a, c = e
        nodes.add(a); nodes.add(c)
        pred_total[a] += 1; pred_total[c] += 1
        pred_correct[a] += 1; pred_correct[c] += 1
    for e in pred_fp:
        a, c = e
        nodes.add(a); nodes.add(c)
        pred_total[a] += 1; pred_total[c] += 1

    rows = []
    for n in nodes:
        gt_t, pred_t = gt_total[n], pred_total[n]
        rows.append({
            "node": n,
            "syn_count": len(parse_synonyms(n)),
            "node_depth": anc_count[n],
            "gt_incident": gt_t,
            "gt_recovered": gt_recovered[n],
            "node_recall": (gt_recovered[n] / gt_t) if gt_t else float("nan"),
            "pred_incident": pred_t,
            "pred_correct": pred_correct[n],
            "node_precision": (pred_correct[n] / pred_t) if pred_t else float("nan"),
        })
    return rows


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
def dataset_scale(dataset):
    """Group a dataset by the size of the graph it was run on.

    Runs are usually on a ~100-node closed SUBset of the full ontology, so
    pooling SUB and FULL together mixes very different graph sizes. The trailing
    _SUB / _FULL tag in the dataset name is the size bucket; anything else -> NA.
    """
    tag = dataset.rsplit("_", 1)[-1].upper() if "_" in dataset else ""
    return tag if tag in ("SUB", "FULL") else "NA"


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


def run(results_dir, out_dir, do_leverage=False, min_support=20):
    os.makedirs(out_dir, exist_ok=True)
    gt_cache = {}
    summary_rows = []
    hop_rows = []
    syn_rows = []
    clawback_rows = []

    found = False
    for dataset, method, clos_txt, red_txt, gt_path in discover(results_dir):
        found = True
        if gt_path not in gt_cache:
            gt_cache[gt_path] = build_gt_views(nx.read_graphml(gt_path))
        views = gt_cache[gt_path]
        clos_edges_gt = views[1]

        recovered_gt, pred_tp, pred_fp, header = parse_closure_report(clos_txt)
        row, by_hop = analyze_dataset_method(views, recovered_gt)
        row = {"Dataset": dataset, "Method": method, **row}
        if header:
            row["closure_F1_reported"] = header["F1"]
        summary_rows.append(row)

        scale = dataset_scale(dataset)
        for d, info in by_hop.items():
            hop_rows.append({"Dataset": dataset, "Scale": scale, "Method": method, "hop_distance": d, **info})

        for r in synonym_node_rows(clos_edges_gt, recovered_gt, pred_tp, pred_fp):
            syn_rows.append({"Dataset": dataset, "Method": method, **r})

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

    # Synonym richness: recall AND precision vs number of merged surface forms.
    syn_df = pd.DataFrame(syn_rows)
    syn_by_count = None
    syn_summary = None
    syn_optimum = None
    if not syn_df.empty:
        def _pr_f1(g):
            g = g.copy()
            g["recall"] = (g["gt_recovered"] / g["gt_incident"]).where(g["gt_incident"] > 0)
            g["precision"] = (g["pred_correct"] / g["pred_incident"]).where(g["pred_incident"] > 0)
            p, r = g["precision"], g["recall"]
            g["f1"] = (2 * p * r / (p + r)).where((p + r) > 0)
            return g

        # Per (dataset, method, syn_count): edge-weighted precision/recall/F1.
        syn_by_count = _pr_f1(
            syn_df.groupby(["Dataset", "Method", "syn_count"])
                  .agg(n_nodes=("node", "size"),
                       gt_incident=("gt_incident", "sum"),
                       gt_recovered=("gt_recovered", "sum"),
                       pred_incident=("pred_incident", "sum"),
                       pred_correct=("pred_correct", "sum"))
                  .reset_index())
        syn_count_path = os.path.join(out_dir, "synonym_recall_by_count.csv")
        syn_by_count.to_csv(syn_count_path, index=False)
        print(f" -> wrote {syn_count_path}  ({len(syn_by_count)} rows)")

        # The hypothesis is only testable on taxonomies WITH synsets -> restrict.
        synset_ds = sorted(syn_df.loc[syn_df["syn_count"] >= 2, "Dataset"].unique())
        sub = syn_df[syn_df["Dataset"].isin(synset_ds)]

        if synset_ds:
            # Pooled over synset datasets, per (method, syn_count): the P/R/F1 curve.
            pooled = _pr_f1(
                sub.groupby(["Method", "syn_count"])
                   .agg(n_nodes=("node", "size"),
                        gt_incident=("gt_incident", "sum"),
                        gt_recovered=("gt_recovered", "sum"),
                        pred_incident=("pred_incident", "sum"),
                        pred_correct=("pred_correct", "sum"))
                   .reset_index())
            pooled_path = os.path.join(out_dir, "synonym_pr_pooled.csv")
            pooled.to_csv(pooled_path, index=False)
            print(f" -> wrote {pooled_path}  (synset datasets: {', '.join(synset_ds)})")

            # Optimal synonym count per method = argmax F1 over well-supported buckets.
            opt_rows = []
            for method, d in pooled.groupby("Method"):
                ok = d[(d["gt_incident"] >= min_support) & (d["pred_incident"] >= min_support)].dropna(subset=["f1"])
                if ok.empty:
                    continue
                best = ok.loc[ok["f1"].idxmax()]
                opt_rows.append({
                    "Method": method,
                    "best_syn_count": int(best["syn_count"]),
                    "precision": best["precision"], "recall": best["recall"], "f1": best["f1"],
                    "max_syn_count_seen": int(d["syn_count"].max()),
                })
            if opt_rows:
                syn_optimum = pd.DataFrame(opt_rows).sort_values("f1", ascending=False)
                opt_path = os.path.join(out_dir, "synonym_pr_optimum.csv")
                syn_optimum.to_csv(opt_path, index=False)
                print(f" -> wrote {opt_path}")

        # Headline: single-surface vs multi-synonym nodes (over synset datasets).
        sum_rows = []
        for method, d in sub.groupby("Method"):
            single, multi = d[d["syn_count"] == 1], d[d["syn_count"] >= 2]
            def _ew(x, num, den):
                return x[num].sum() / x[den].sum() if x[den].sum() else float("nan")
            sum_rows.append({
                "Method": method,
                "n_nodes": len(d),
                "n_multi_syn_nodes": int((d["syn_count"] >= 2).sum()),
                "recall_syn1": _ew(single, "gt_recovered", "gt_incident"),
                "recall_syn2plus": _ew(multi, "gt_recovered", "gt_incident"),
                "recall_lift": _ew(multi, "gt_recovered", "gt_incident") - _ew(single, "gt_recovered", "gt_incident"),
                "precision_syn1": _ew(single, "pred_correct", "pred_incident"),
                "precision_syn2plus": _ew(multi, "pred_correct", "pred_incident"),
                "precision_lift": _ew(multi, "pred_correct", "pred_incident") - _ew(single, "pred_correct", "pred_incident"),
                "spearman_recall": (d["syn_count"].rank().corr(d["node_recall"].rank())
                                    if d["syn_count"].nunique() > 1 and d["node_recall"].nunique() > 1
                                    else float("nan")),
            })
        if sum_rows:
            syn_summary = pd.DataFrame(sum_rows).sort_values("recall_lift", ascending=False, na_position="last")
            syn_sum_path = os.path.join(out_dir, "synonym_recall_summary.csv")
            syn_summary.to_csv(syn_sum_path, index=False)
            print(f" -> wrote {syn_sum_path}")
        if not synset_ds:
            print(" [i] No datasets with synonyms (synsets) found; synonym P/R analysis is N/A.")
    else:
        print(" [i] No nodes available for synonym analysis.")

    # Aggregated view: mean bucketed recall per method across datasets.
    print("\n" + "=" * 78)
    print("  MEAN RECALL BY BUCKET, PER METHOD (averaged across datasets)")
    print("  reduction = direct links | implied = closure-minus-reduction (indirect)")
    print("=" * 78)
    agg = (summary.groupby("Method")[["recall_reduction", "recall_implied", "recall_closure"]]
                  .mean().sort_values("recall_implied", ascending=False))
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)

    if syn_summary is not None:
        print("\n" + "=" * 90)
        print("  SYNONYM RICHNESS vs RECALL & PRECISION (pooled over taxonomies WITH synsets)")
        print("  *_syn1 = nodes w/ 1 surface form | *_syn2plus = >=2 merged synonyms")
        print("  (low n_multi_syn_nodes => weak evidence; effect may be a depth confound)")
        print("=" * 90)
        print(syn_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 90)

    if syn_optimum is not None:
        print("\n" + "=" * 90)
        print(f"  OPTIMAL #SYNONYMS PER METHOD (argmax F1 over buckets with >= {min_support} edges)")
        print("=" * 90)
        print(syn_optimum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 90)

    return summary, hop_df, syn_by_count


# ----------------------------------------------------------------------------
# Optional plots
# ----------------------------------------------------------------------------
def select_methods(methods, pattern):
    """Pick which methods to show in the synonym plot.

    pattern None / "all" / "*" -> keep everything. Otherwise keep methods whose
    name contains the (case-insensitive) substring; if none match, fall back to
    all so the panel is never empty. Default "our" selects the Our Method family
    (OurMethod, Our_Method_k=10, ...)."""
    methods = list(methods)
    if pattern is None or str(pattern).strip().lower() in ("all", "*", ""):
        return methods
    pl = str(pattern).strip().lower()
    hits = [m for m in methods if pl in m.lower()]
    return hits or methods


def make_plots(summary, hop_df, syn_by_count, out_dir, min_support=20, synonym_method="our"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

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
    written.append(p1)

    # 2) SUPPORT-WEIGHTED edge recall vs hop-distance, BROKEN OUT BY DATASET SIZE.
    #    hop 1 = direct (reduction) edges, hop >= 2 = indirect (implied) edges, so
    #    direct vs indirect recall sit on one axis. Runs are usually on ~100-node
    #    SUBsets, so SUB and FULL are plotted separately to keep pooling consistent
    #    (otherwise a few large FULL graphs dominate the edge-weighted curve).
    #    - line value = edge-weighted pooled recall (sum recovered / sum support);
    #    - marker AREA proportional to that point's pooled support;
    #    - points below --min_support drawn faint+dotted (the unreliable thin tail);
    #    - grey bars (right axis) show the GT support distribution per hop;
    #    - shaded band marks hop 1 (direct edges).
    def _draw_hop(ax1, sub_hop, title):
        pooled = (sub_hop.groupby(["Method", "hop_distance"])
                  .agg(recovered=("recovered", "sum"), support=("support", "sum"))
                  .reset_index())
        pooled["recall"] = pooled["recovered"] / pooled["support"]
        per_ds = sub_hop.drop_duplicates(["Dataset", "hop_distance"])
        support_by_hop = per_ds.groupby("hop_distance")["support"].sum()

        ax2 = ax1.twinx()
        ax2.bar(support_by_hop.index, support_by_hop.values, color="gray", alpha=0.12,
                width=0.85, zorder=0)
        ax2.set_ylabel("GT edges at this hop (support)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        ax1.axvspan(0.5, 1.5, color="black", alpha=0.05, zorder=0)  # hop-1 (direct) band
        for method, s in pooled.groupby("Method"):
            s = s.sort_values("hop_distance")
            solid = s[s["support"] >= min_support]
            faint = s[s["support"] < min_support]
            color = None
            if not solid.empty:
                line, = ax1.plot(solid["hop_distance"], solid["recall"], "-", label=method, zorder=3)
                color = line.get_color()
            ax1.scatter(s["hop_distance"], s["recall"],
                        s=12 + 6 * np.sqrt(s["support"].to_numpy()),
                        color=color, alpha=0.85, zorder=4, label=(None if color else method))
            if not faint.empty:
                ax1.plot(faint["hop_distance"], faint["recall"], ":", color=color, alpha=0.35, zorder=2)

        ax1.set_xlabel("GT hop-distance  (1 = direct parent->child;  >=2 = indirect ancestor)")
        ax1.set_ylabel("Edge recall (edge-weighted, pooled)")
        ax1.set_ylim(0, 1.02)
        ax1.set_zorder(ax2.get_zorder() + 1)   # draw recall lines above the support bars
        ax1.patch.set_visible(False)
        ax1.set_title(title)
        ax1.legend(fontsize=7, ncol=2, loc="lower right")

    hop_scaled = hop_df if "Scale" in hop_df.columns else hop_df.assign(Scale="ALL")
    scales = [s for s in ["SUB", "FULL", "NA", "ALL"] if s in set(hop_scaled["Scale"])]
    for scale in scales:
        sub_hop = hop_scaled[hop_scaled["Scale"] == scale]
        fig, ax1 = plt.subplots(figsize=(10, 6.5))
        _draw_hop(ax1, sub_hop, f"Recall vs hop-distance — {scale} datasets\n"
                                f"hop 1 = direct, ≥ 2 = indirect; marker area ∝ #edges; "
                                f"dotted = below {min_support}-edge support")
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"recall_by_hop_{scale}.png")
        plt.savefig(p2, dpi=150); plt.close()
        written.append(p2)

    # 3) Synonym richness vs PRECISION / RECALL / F1, one panel per method.
    #    Restricted to taxonomies that actually have synsets (>=2 surface forms),
    #    so the question "do synonyms help/hurt, and what's the optimum" is read
    #    straight off the GT. Dashed line marks the F1-optimal synonym count.
    if syn_by_count is not None and not syn_by_count.empty:
        synset_ds = syn_by_count.loc[syn_by_count["syn_count"] >= 2, "Dataset"].unique()
        sub_sy = syn_by_count[syn_by_count["Dataset"].isin(synset_ds)]
        if len(synset_ds) and sub_sy["syn_count"].nunique() > 1:
            ps = (sub_sy.groupby(["Method", "syn_count"])
                  .agg(gt_incident=("gt_incident", "sum"), gt_recovered=("gt_recovered", "sum"),
                       pred_incident=("pred_incident", "sum"), pred_correct=("pred_correct", "sum"))
                  .reset_index())
            ps["recall"] = (ps["gt_recovered"] / ps["gt_incident"]).where(ps["gt_incident"] > 0)
            ps["precision"] = (ps["pred_correct"] / ps["pred_incident"]).where(ps["pred_incident"] > 0)
            P, R = ps["precision"], ps["recall"]
            ps["f1"] = (2 * P * R / (P + R)).where((P + R) > 0)

            methods = select_methods(sorted(ps["Method"].unique()), synonym_method)
            ncols = min(3, len(methods))
            nrows = (len(methods) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows),
                                     squeeze=False, sharex=True, sharey=True)
            series = [("precision", "tab:red", "pred_incident", "Precision"),
                      ("recall", "tab:blue", "gt_incident", "Recall"),
                      ("f1", "tab:green", "pred_incident", "F1")]
            for i, method in enumerate(methods):
                ax = axes[i // ncols][i % ncols]
                d = ps[ps["Method"] == method].sort_values("syn_count")
                for col, color, supcol, lbl in series:
                    ax.plot(d["syn_count"], d[col], "-", color=color, alpha=0.45)
                    ax.scatter(d["syn_count"], d[col],
                               s=10 + 5 * np.sqrt(d[supcol].clip(lower=1).to_numpy()),
                               color=color, alpha=0.85, label=lbl)
                ok = d[(d["gt_incident"] >= min_support) & (d["pred_incident"] >= min_support)].dropna(subset=["f1"])
                if not ok.empty:
                    best = ok.loc[ok["f1"].idxmax()]
                    ax.axvline(best["syn_count"], color="tab:green", ls="--", alpha=0.6)
                    ax.annotate(f"F1* @ {int(best['syn_count'])}", (best["syn_count"], 0.04),
                                fontsize=7, color="tab:green", ha="center")
                ax.set_title(method, fontsize=9)
                ax.set_ylim(0, 1.02)
                ax.grid(alpha=0.2)
            for j in range(len(methods), nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            axes[0][0].legend(fontsize=7, loc="lower right")
            _all_sel = synonym_method is None or str(synonym_method).strip().lower() in ("all", "*", "")
            _sel = "all methods" if _all_sel else f"method: '{synonym_method}'"
            fig.suptitle(f"Synonym richness vs Precision / Recall / F1 — {_sel} (taxonomies with synsets)\n"
                         f"marker area ∝ #edges; dashed = F1-optimal synonym count over >= {min_support}-edge buckets",
                         fontsize=11)
            fig.text(0.5, 0.01, "Synonyms merged into the node (surface forms)", ha="center")
            fig.text(0.005, 0.5, "Score (edge-weighted, pooled over synset datasets)",
                     va="center", rotation="vertical")
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
            p3 = os.path.join(out_dir, "synonym_pr_curves.png")
            plt.savefig(p3, dpi=150); plt.close()
            written.append(p3)

    print(" -> wrote " + "\n -> wrote ".join(written))


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
    # hop is now over ALL closure edges: direct edges == 1, implied >= 2.
    assert hop[("a", "b")] == 1 and hop[("a", "e")] == 1, hop
    assert hop[("a", "c")] == 2 and hop[("a", "d")] == 3 and hop[("b", "d")] == 2, hop

    # A "single-parent-ish" method that recovered direct links + the 2-hop ones
    # but MISSED the 3-hop a->d (chain broken downstream).
    recovered = {("a", "b"), ("b", "c"), ("c", "d"), ("a", "e"), ("a", "c"), ("b", "d")}
    row, by_hop = analyze_dataset_method((red, clos, implied, hop), recovered)
    assert abs(row["recall_reduction"] - 1.0) < 1e-9, row
    assert abs(row["recall_implied"] - 2/3) < 1e-9, row     # got a-c, b-d; missed a-d
    # by_hop now includes hop 1 (direct edges): all 4 recovered -> recall 1.0.
    assert by_hop[1]["support"] == 4 and by_hop[1]["recall"] == 1.0, by_hop
    assert by_hop[1]["kind"] == "direct" and by_hop[2]["kind"] == "indirect", by_hop
    assert by_hop[2]["recall"] == 1.0 and by_hop[3]["recall"] == 0.0, by_hop
    assert dataset_scale("CellOntology_SUB") == "SUB" and dataset_scale("LLMs4OL_PO_FULL") == "FULL"
    assert dataset_scale("medium_components") == "NA"
    print("  build_gt_views / analyze_dataset_method / dataset_scale  OK")

    # Report parsing round-trip.
    with tempfile.TemporaryDirectory() as td:
        clos_txt = os.path.join(td, "DS_M_condensed_closure.txt")
        with open(clos_txt, "w", encoding="utf-8") as f:
            f.write("Precision: 0.8000 | Recall: 0.5000 | F1: 0.6154\n")
            f.write("  [TP] PRED: (a -> c)\n        matches GT: (a -> c)\n")
            f.write("  [TP] PRED: (a -> b)\n        matches GT: (a -> b)\n")
            f.write("  [FP] x -> y\n")
        rec, ptp, fp, hdr = parse_closure_report(clos_txt)
        assert rec == {("a", "c"), ("a", "b")}, rec
        assert ptp == {("a", "c"), ("a", "b")}, ptp     # predicted side of the TPs
        assert fp == {("x", "y")}, fp
        assert hdr["F1"] == 0.6154, hdr
    print("  parse_closure_report  OK")

    # Leverage: a wrong high edge a->c with a deep subtree should rank top.
    pred = {("a", "b"), ("b", "c"), ("c", "d"), ("root", "a")}
    cand = leverage_candidates(pred, gt_closure_edges=set(), top_n=5)
    assert cand and cand[0]["leverage"] >= cand[-1]["leverage"]
    print("  leverage_candidates  OK")

    # Synonym richness: parsing + incident-edge attribution (recall AND precision).
    assert parse_synonyms("b (b, b2)") == ["b", "b2"]
    assert parse_synonyms("x|y|z") == ["x", "y", "z"]
    assert parse_synonyms("plain") == ["plain"]
    clos2 = {("a", "b|b2"), ("b|b2", "c"), ("a", "c")}     # GT closure of a->b|b2->c
    recovered2 = {("a", "b|b2"), ("b|b2", "c")}            # missed the implied a->c
    pred_tp2 = {("a", "b|b2"), ("b|b2", "c")}              # those two were the predicted TPs
    pred_fp2 = {("a", "z")}                                # one wrong edge touching node a
    srows = {r["node"]: r for r in synonym_node_rows(clos2, recovered2, pred_tp2, pred_fp2)}
    assert srows["b|b2"]["syn_count"] == 2
    assert abs(srows["b|b2"]["node_recall"] - 1.0) < 1e-9   # both incident GT edges recovered
    assert abs(srows["a"]["node_recall"] - 0.5) < 1e-9      # a->b|b2 yes, a->c no
    assert srows["c"]["node_depth"] == 2                    # ancestors: a and b|b2
    # precision on node a: incident predicted = {a->b|b2 (TP), a->z (FP)} -> 1/2
    assert abs(srows["a"]["node_precision"] - 0.5) < 1e-9, srows["a"]
    assert abs(srows["b|b2"]["node_precision"] - 1.0) < 1e-9  # only its TP edges
    assert "z" in srows and srows["z"]["node_precision"] == 0.0  # node only in an FP edge
    print("  parse_synonyms / synonym_node_rows  OK")

    # Synonym-plot method selector.
    ms = ["OurMethod", "Our_Method_k=10", "Lexical", "TaxoLLaMA_PPL15.0"]
    assert select_methods(ms, "our") == ["OurMethod", "Our_Method_k=10"]
    assert select_methods(ms, "all") == ms and select_methods(ms, None) == ms
    assert select_methods(ms, "lexical") == ["Lexical"]
    assert select_methods(ms, "nomatch") == ms          # fall back to all, never empty
    print("  select_methods  OK")
    print("ALL SELF-TESTS PASSED.")


def main():
    ap = argparse.ArgumentParser(description="Implied-edge (closure-minus-reduction) analysis from saved results.")
    ap.add_argument("--results_dir", default="./results", help="Folder with GT_*.graphml and *_condensed_*.txt")
    ap.add_argument("--out_dir", default="./implied_analysis", help="Where to write CSVs/PNGs")
    ap.add_argument("--plot", action="store_true", help="Also render figures")
    ap.add_argument("--leverage", action="store_true", help="Also emit precision-clawback candidate edges")
    ap.add_argument("--min_support", type=int, default=20,
                    help="Buckets with fewer pooled edges are drawn faint/omitted in plots and "
                         "excluded when picking the F1-optimal synonym count")
    ap.add_argument("--synonym_method", default="our",
                    help="Which method(s) the synonym P/R/F1 plot shows: case-insensitive name "
                         "substring (default 'our' = the Our Method family); use 'all' for every method")
    ap.add_argument("--selftest", action="store_true", help="Validate logic on a synthetic example and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    out = run(args.results_dir, args.out_dir, do_leverage=args.leverage, min_support=args.min_support)
    if out and args.plot:
        summary, hop_df, syn_by_count = out
        make_plots(summary, hop_df, syn_by_count, args.out_dir,
                   min_support=args.min_support, synonym_method=args.synonym_method)


if __name__ == "__main__":
    main()
