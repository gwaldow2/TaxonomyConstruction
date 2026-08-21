"""Rescore predictions against a GT filtered to is-a edges, and compare with the original GT.

Uses the relation-type labels the audit already produced (results/relation_types.csv, GT rows)
to drop GT edges judged non-is-a, then scores the SAME predictions against both versions of the
ground truth. If those edges were genuinely unwinnable for a faithful is-a extractor, removing
them should raise recall (fewer impossible false negatives) with little precision cost.

--types_csv is REPEATABLE for the planned two-judge design: with several CSVs (e.g. one judged
by Gemma, one by Gemini), an edge is dropped only when EVERY judge that classified it says
non-is-a -- consensus. A split verdict keeps the edge, so single-judge idiosyncrasies cannot
delete ground truth, and adding a second judge can only make the filter more conservative.

    python isa_rescore.py                                        # Gemma labels, Exp_Raw scores
    python isa_rescore.py --mode Cond_Clos                       # headline metric instead
    python isa_rescore.py --types_csv results/relation_types_gemma.csv \
                          --types_csv results/relation_types_gemini.csv   # consensus

--mode defaults to Exp_Raw (direct-edge scoring, no closure): the transitive closure couples
recall failures into precision loss, so raw scoring shows the filter's effect without closure
amplification on top. Outputs vis/isa_rescore.png + results/isa_rescore_summary.csv.
"""

import os
import csv
import glob
import argparse
from collections import defaultdict

import networkx as nx

from relation_type_audit import RESULTS_DIR, VIS_DIR, find_datasets, load_gt_graph, load_pred_edges
from model_comparison import paired_ttest, _p_stars

MODES = ["Exp_Raw", "Cond_Clos", "Cond_Red", "Exp_Clos"]


def load_type_maps(paths):
    """-> one {(dataset, parent, child): type} per CSV, GT rows only."""
    maps = []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"[!] {p} not found -- run relation_type_audit.py first.")
        with open(p, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["source"] == "gt"]
        maps.append({(r["dataset"], r["parent"], r["child"]): r["type"] for r in rows})
        print(f"    {p}: {len(rows)} GT edge classifications")
    return maps


def consensus_drop(ds, edge, maps, keep_types=("is_a",)):
    """True iff every judge that classified this edge says non-is-a (and at least one did).

    An unclassified edge is never dropped -- absence of an audit is not evidence -- and a
    split verdict keeps the edge, so the filter only ever acts on consensus.
    """
    key = (ds, edge[0], edge[1])
    opinions = [m[key] for m in maps if key in m]
    return bool(opinions) and all(t not in keep_types for t in opinions)


def filter_gt(G, ds, maps):
    """-> (G_filtered, dropped_edges)."""
    drop = [e for e in G.edges() if consensus_drop(ds, e, maps)]
    G_out = G.copy()
    G_out.remove_edges_from(drop)
    return G_out, drop


def score(results_dir, ds, pred_edges, G_gt, tag):
    """-> {mode: {Precision, Recall, F1}} for the predictions against one GT version."""
    from evaluator import evaluate_all_modes
    G_pred = nx.DiGraph()
    G_pred.add_nodes_from(G_gt.nodes())
    G_pred.add_edges_from([(p, c) for p, c, _ in pred_edges])
    tmp = os.path.join(results_dir, f"_isa_rescore_{ds}_{tag}")
    res = evaluate_all_modes(G_pred, G_gt, tmp)
    for f in glob.glob(tmp + "*"):
        try:
            os.remove(f)
        except OSError:
            pass
    return res


def plot(rows, mode, out_path):
    """3 panels (P/R/F1): per-dataset bars, original GT vs is-a-only GT, paired t-test in title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = [r["dataset"] for r in rows]
    short = [d.replace("_SUB", "") for d in datasets]
    fig, axes = plt.subplots(1, 3, figsize=(6.2 * 3, 6))
    for ax, key in zip(axes, ["Precision", "Recall", "F1"]):
        orig = [r[f"orig_{key}"] for r in rows]
        filt = [r[f"filt_{key}"] for r in rows]
        x = np.arange(len(datasets))
        ax.bar(x - 0.2, orig, 0.38, label="all edge types (original GT)", color="#888888", alpha=0.8)
        ax.bar(x + 0.2, filt, 0.38, label="is-a only (judge-filtered GT)", color="#4C72B0", alpha=0.9)
        d = [f - o for o, f in zip(orig, filt)]
        t, dfree, p = paired_ttest(d)
        ax.set_title(f"{key}\ndelta={sum(d)/len(d):+.3f} p={p:.2g}{_p_stars(p)} (n={len(d)})",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(key, fontweight="bold")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Scoring against original vs is-a-filtered ground truth  [{mode}]",
                 y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Rescore against an is-a-filtered GT vs the original.")
    ap.add_argument("--types_csv", action="append", default=None,
                    help="Relation-type CSV(s) from relation_type_audit.py. Repeatable: with "
                         "several judges an edge is dropped only on consensus. "
                         "Default: results/relation_types.csv")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--pred_label", default=None, help="Substring filter for the diagnostics file.")
    ap.add_argument("--mode", choices=MODES, default="Exp_Raw",
                    help="Scoring condition to plot. Exp_Raw (default) avoids closure "
                         "amplification; Cond_Clos is the headline metric.")
    ap.add_argument("--out_png", default=os.path.join(VIS_DIR, "isa_rescore.png"))
    ap.add_argument("--out_csv", default=os.path.join(RESULTS_DIR, "isa_rescore_summary.csv"))
    ap.add_argument("--show_dropped", type=int, default=5, help="Dropped edges to print per dataset.")
    args = ap.parse_args()

    print("[*] relation-type labels:")
    maps = load_type_maps(args.types_csv or [os.path.join(RESULTS_DIR, "relation_types.csv")])
    n_judges = len(maps)

    datasets = args.datasets or find_datasets(args.results_dir)
    rows = []
    for ds in datasets:
        G = load_gt_graph(args.results_dir, ds)
        if G is None:
            continue
        pred, path = load_pred_edges(args.results_dir, ds, args.pred_label)
        if not pred:
            print(f"    [!] {ds}: no edge diagnostics -- skipping")
            continue
        G_filt, dropped = filter_gt(G, ds, maps)
        if not dropped:
            n_judged = sum(1 for e in G.edges() if any((ds, e[0], e[1]) in m for m in maps))
            print(f"\n### {ds}: 0 of {G.number_of_edges()} GT edges dropped "
                  f"({n_judged} judged) -- scores will be identical")
        else:
            print(f"\n### {ds}: dropped {len(dropped)}/{G.number_of_edges()} GT edges "
                  f"(consensus of {n_judges} judge{'s' if n_judges > 1 else ''})")
            for p, c in dropped[:args.show_dropped]:
                print(f"      - {c!r} under {p!r}")
            if len(dropped) > args.show_dropped:
                print(f"      ... and {len(dropped) - args.show_dropped} more")
        print(f"    predictions: {len(pred)} edges <- {os.path.basename(path)}")

        a = score(args.results_dir, ds, pred, G, "orig")[args.mode]
        b = score(args.results_dir, ds, pred, G_filt, "filt")[args.mode]
        rows.append({"dataset": ds, "gt_edges": G.number_of_edges(), "dropped": len(dropped),
                     "orig_Precision": a["Precision"], "orig_Recall": a["Recall"], "orig_F1": a["F1"],
                     "filt_Precision": b["Precision"], "filt_Recall": b["Recall"], "filt_F1": b["F1"]})
        print(f"    [{args.mode}] P {a['Precision']:.4f} -> {b['Precision']:.4f}  "
              f"R {a['Recall']:.4f} -> {b['Recall']:.4f}  F1 {a['F1']:.4f} -> {b['F1']:.4f}")

    if not rows:
        raise SystemExit("[!] nothing to score -- need GT graphs, diagnostics CSVs and type labels.")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.mode, args.out_png)
    print(f"\n[*] wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
