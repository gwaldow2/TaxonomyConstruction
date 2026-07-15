"""Offline prune-sweep analysis -- heuristic-only cutting vs the LLM prune, no LLM calls.

For each dataset it reconstructs the extracted graph from its raw edge-diagnostics CSV
(results/<ds>_Our_Method_edge_diagnostics.csv, which already carries the per-edge suspicion
components + an is_fp label) and the GT graph (results/GT_<ds>_eval.graphml), then sweeps
K = number of top-suspicious edges removed and evaluates the resulting **Cond. Closure**
P/R/F1 -- exactly as evaluator.compute_and_save_metrics does (set_overlap on synonyms) -- for:

  * each --rank_by heuristic (combined / leverage / neighborhood / self_agreement / salience),
    cutting the top-K by that score with NO model in the loop,
  * an ORACLE that removes the K highest-leverage TRUE false positives (the ceiling), and
  * the baseline (K=0).

Optionally overlays the LLM restructure_prune_only runs from a results JSON, so you can see
whether the heuristic-only cut ever beats the LLM prune. Also plots the false-positive MASS
concentration (how few edges generate most of the closure FPs).

    python prune_sweep_analysis.py --results_dir results --llm_results restr_method.json

Outputs vis/prune_sweep.png.
"""

import os
import re
import csv
import glob
import json
import argparse

import networkx as nx

VIS_DIR = "vis"
RANK_BYS = ("combined", "leverage", "neighborhood", "self_agreement", "salience")


# ---- pure copies of the two evaluator helpers (avoid importing data_manager/evaluator,
#      which pull in nltk/obonet/requests) ----
def parse_lemma_format(node_str):
    node_str = str(node_str).strip().lower()
    match = re.search(r'^([^(]+)\s*\((.*)\)$', node_str)
    if match:
        primary_text = match.group(1).strip()
        syns = [s.strip() for s in match.group(2).split(',') if s.strip()]
        if syns and syns[0] == primary_text:
            return syns
    if '|' in node_str:
        return [s.strip() for s in node_str.split('|') if s.strip()]
    return [node_str]


def _exploded_pairs(edges):
    pairs = set()
    for u, v in edges:
        for tu in parse_lemma_format(u):
            for tv in parse_lemma_format(v):
                pairs.add((tu, tv))
    return pairs


def _matches(u, v, pairs):
    for tu in parse_lemma_format(u):
        for tv in parse_lemma_format(v):
            if (tu, tv) in pairs:
                return True
    return False


def _pct_rank(values):
    n = len(values)
    if n == 0: return []
    if n == 1: return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for r, i in enumerate(order):
        ranks[i] = r / (n - 1)
    return ranks


def score_edges(rows, rank_by):
    """{edge: score} under rank_by, from the diagnostics components (matches edge_suspicion_scores)."""
    presets = {"combined": (1, 1, 1, 1), "leverage": (1, 0, 0, 0), "neighborhood": (0, 1, 0, 0),
               "self_agreement": (0, 0, 1, 0), "salience": (0, 0, 0, 1)}
    wl, wn, wa, ws = presets[rank_by]
    lev_p = _pct_rank([r["leverage"] for r in rows])
    sal_p = _pct_rank([r["salience"] for r in rows])
    out = {}
    for i, r in enumerate(rows):
        low_agree = 1.0 if r["votes"] <= 1 else 0.0
        out[(r["parent"], r["child"])] = (wl * lev_p[i] + wn * (1.0 - r["neighborhood_agreement"])
                                          + wa * low_agree + ws * (1.0 - sal_p[i]))
    return out


def cond_clos(G_pred, gt_pairs, gt_edges):
    """Cond. Closure (P, R, F1, n_fp) of G_pred vs the GT, set_overlap on synonyms."""
    pred_edges = list(nx.transitive_closure(G_pred).edges()) if G_pred.number_of_edges() else []
    if not pred_edges:
        return (0.0, 0.0, 0.0, 0)
    pred_pairs = _exploded_pairs(pred_edges)
    tp_pred = sum(1 for (u, v) in pred_edges if _matches(u, v, gt_pairs))
    tp_gt = sum(1 for (u, v) in gt_edges if _matches(u, v, pred_pairs))
    p = tp_pred / len(pred_edges)
    r = tp_gt / len(gt_edges) if gt_edges else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return (p, r, f1, len(pred_edges) - tp_pred)


def load_dataset(csv_path, gt_dir):
    """-> (dataset, rows, G_pred, gt_pairs, gt_edges) or None if the GT graphml is missing."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            rows.append({"parent": d["parent"], "child": d["child"],
                         "leverage": float(d["leverage"]), "neighborhood_agreement": float(d["neighborhood_agreement"]),
                         "votes": float(d["votes"]), "salience": float(d["salience"]), "is_fp": int(d["is_fp"])})
    if not rows:
        return None
    dataset = rows[0].get("dataset") if "dataset" in rows[0] else None
    # dataset from the filename: <ds>_Our_Method_edge_diagnostics.csv
    ds = re.sub(r"_Our_Method_edge_diagnostics\.csv$", "", os.path.basename(csv_path))
    gt_path = os.path.join(gt_dir, f"GT_{ds}_eval.graphml")
    if not os.path.exists(gt_path):
        print(f"    [!] missing GT graphml for {ds}: {gt_path} -- skipping")
        return None
    G_gt = nx.DiGraph(nx.read_graphml(gt_path))
    if "virtual_root" in G_gt:
        G_gt.remove_node("virtual_root")
    gt_edges = list(nx.transitive_closure(G_gt).edges())
    gt_pairs = _exploded_pairs(gt_edges)
    G_pred = nx.DiGraph()
    G_pred.add_edges_from([(r["parent"], r["child"]) for r in rows])
    return (ds, rows, G_pred, gt_pairs, gt_edges)


def sweep_dataset(rows, G_pred, gt_pairs, gt_edges, ks):
    """-> {'rank_by'->{K->(p,r,f1)}, 'oracle'->{K->(p,r,f1)}, 'fp_curve'->[(n_removed, frac_fp_gone)]}."""
    base_p, base_r, base_f1, base_fp = cond_clos(G_pred, gt_pairs, gt_edges)
    res = {"baseline": (base_p, base_r, base_f1), "by": {}, "oracle": {}}
    for rb in RANK_BYS:
        scores = score_edges(rows, rb)
        ranked = [e for e, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        res["by"][rb] = {}
        for K in ks:
            G = G_pred.copy(); G.remove_edges_from(ranked[:K])
            res["by"][rb][K] = cond_clos(G, gt_pairs, gt_edges)[:3]
    # oracle: remove the K highest-leverage TRUE FPs
    fp_by_lev = [ (r["parent"], r["child"]) for r in sorted([x for x in rows if x["is_fp"] == 1],
                                                            key=lambda x: x["leverage"], reverse=True) ]
    for K in ks:
        G = G_pred.copy(); G.remove_edges_from(fp_by_lev[:K])
        res["oracle"][K] = cond_clos(G, gt_pairs, gt_edges)[:3]
    # FP-mass concentration: fraction of baseline closure-FPs eliminated as we remove FP edges (oracle order)
    curve = []
    for n in range(0, len(fp_by_lev) + 1):
        G = G_pred.copy(); G.remove_edges_from(fp_by_lev[:n])
        _, _, _, n_fp = cond_clos(G, gt_pairs, gt_edges)
        frac_gone = 1.0 - (n_fp / base_fp) if base_fp else 0.0
        curve.append((n / len(fp_by_lev) if fp_by_lev else 0.0, frac_gone))
    res["fp_curve"] = curve
    return res


def load_llm_prune(results_path):
    """-> {(K, rank_by): (meanP, meanR, meanF1)} for restructure_prune_only runs in a results JSON."""
    if not results_path or not os.path.exists(results_path):
        return {}
    agg = {}
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    for block in data:
        for r in block.get("results", []):
            m = r.get("method", "")
            if "+restructure_prune_only" not in m:
                continue
            k = int(re.search(r"_top(\d+)", m).group(1)) if re.search(r"_top(\d+)", m) else 0
            rb = re.search(r"\+rank_(\w+)", m)
            rb = rb.group(1) if rb else "combined"
            agg.setdefault((k, rb), []).append((r.get("Cond_Clos_Precision"), r.get("Cond_Clos_Recall"), r.get("Cond_Clos_F1")))
    out = {}
    for key, vals in agg.items():
        vals = [v for v in vals if None not in v]
        if vals:
            out[key] = tuple(sum(v[i] for v in vals) / len(vals) for i in range(3))
    return out


def plot(agg, llm, ks, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    color = {rb: cmap(i) for i, rb in enumerate(RANK_BYS)}
    metric_idx = {"F1": 2, "Precision": 0, "Recall": 1}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panels = [("F1", axes[0, 0]), ("Precision", axes[0, 1]), ("Recall", axes[1, 0])]

    for name, ax in panels:
        mi = metric_idx[name]
        base_m = sum(a["baseline"][mi] for a in agg) / len(agg)
        ax.axhline(base_m, ls=":", color=".5", lw=1.5, label=f"baseline ({base_m:.3f})")
        for rb in RANK_BYS:
            ys = [sum(a["by"][rb][K][mi] for a in agg) / len(agg) for K in ks]
            ax.plot(ks, ys, "-o", ms=3, color=color[rb], label=rb, alpha=0.9)
        oy = [sum(a["oracle"][K][mi] for a in agg) / len(agg) for K in ks]
        ax.plot(ks, oy, "--", color="black", lw=2, label="oracle (cut true FPs)")
        # overlay LLM prune points
        for (k, rb), pr in (llm or {}).items():
            ax.scatter([k], [pr[mi]], marker="*", s=180, color=color.get(rb, "red"),
                       edgecolors="black", zorder=5, label=f"LLM prune [{rb}]")
        ax.set_title(f"Cond. Closure {name} vs edges removed (top-K)", fontweight="bold")
        ax.set_xlabel("K (edges cut)", fontweight="bold"); ax.set_ylabel(name, fontweight="bold")
        ax.set_ylim(0, 1.02)
        # de-dupe legend
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=2)

    # FP-mass concentration (mean curve over a common x-grid)
    axc = axes[1, 1]
    import numpy as np
    xg = np.linspace(0, 1, 21)
    curves = []
    for a in agg:
        xs = [p[0] for p in a["fp_curve"]]; ys = [p[1] for p in a["fp_curve"]]
        curves.append(np.interp(xg, xs, ys) if len(xs) > 1 else np.zeros_like(xg))
    mean_curve = np.mean(curves, axis=0)
    axc.plot(xg, mean_curve, "-o", color="#4C72B0", ms=3)
    axc.plot([0, 1], [0, 1], "--", color=".6", lw=1)   # diffuse reference (proportional)
    axc.set_title("FP-mass concentration (oracle order)\nsteep = few edges make most closure FPs",
                  fontweight="bold")
    axc.set_xlabel("fraction of FP edges removed", fontweight="bold")
    axc.set_ylabel("fraction of closure FPs eliminated", fontweight="bold")
    axc.set_xlim(0, 1); axc.set_ylim(0, 1.02)

    fig.suptitle("Prune sweep: heuristic-only cut vs LLM prune vs oracle (Cond. Closure)",
                 y=1.01, fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Offline prune-sweep: heuristic vs LLM vs oracle.")
    ap.add_argument("--results_dir", default="results",
                    help="Dir with <ds>_Our_Method_edge_diagnostics.csv and GT_<ds>_eval.graphml.")
    ap.add_argument("--llm_results", default=None, help="Results JSON to overlay LLM prune runs (optional).")
    ap.add_argument("--ks", default="0,1,2,3,5,7,10,15,20,25,30,40,50,75,100",
                    help="Comma-separated K values to sweep.")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "prune_sweep.png"))
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",")})
    csvs = sorted(glob.glob(os.path.join(args.results_dir, "*_Our_Method_edge_diagnostics.csv")))
    if not csvs:
        print(f"[!] No *_Our_Method_edge_diagnostics.csv in {args.results_dir}. "
              f"Run our_method (raw) so it writes edge diagnostics.")
        return
    agg = []
    for c in csvs:
        loaded = load_dataset(c, args.results_dir)
        if loaded is None:
            continue
        ds, rows, G_pred, gt_pairs, gt_edges = loaded
        agg.append(sweep_dataset(rows, G_pred, gt_pairs, gt_edges, ks))
        b = agg[-1]["baseline"]
        n_fp = sum(1 for r in rows if r["is_fp"] == 1)
        print(f"    {ds:26s} edges={len(rows):4d} (FP={n_fp:4d})  baseline P/R/F1={b[0]:.3f}/{b[1]:.3f}/{b[2]:.3f}")
    if not agg:
        print("[!] No datasets evaluated (missing GT graphmls?).")
        return
    llm = load_llm_prune(args.llm_results)
    if llm:
        print(f"[*] overlaying {len(llm)} LLM prune point(s): "
              + ", ".join(f"top{k}/{rb}={v[2]:.3f}F1" for (k, rb), v in sorted(llm.items())))
    # quick verdict: best heuristic F1 vs best LLM F1 (at matched K where available)
    best_h = max(sum(a["by"][rb][K][2] for a in agg) / len(agg) for rb in RANK_BYS for K in ks)
    print(f"[*] best heuristic-only mean F1 over the sweep: {best_h:.3f}  "
          f"(baseline {sum(a['baseline'][2] for a in agg)/len(agg):.3f})")
    if llm:
        print(f"    best LLM-prune mean F1: {max(v[2] for v in llm.values()):.3f}")
    plot(agg, llm, ks, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
