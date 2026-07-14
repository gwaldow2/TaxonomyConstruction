"""Compare Our Method sub-strategies within one results file, on P/R/F1.

Two groupings (``--group_by``):
  * strategy  -- Raw (clawback off) vs Clawback vs Restructure vs Ranked vs Prune-only,
                 each vs the Raw baseline. (goal: post-raw strategies vs the raw strategy)
  * rank_by   -- among restructure_prune_only runs only, split by --rank_by
                 (combined | leverage | neighborhood | self_agreement | salience), each
                 vs 'combined'. (goal: prune-only runs compared by heuristic)

For each grouping it plots Cond. Closure F1 / Precision / Recall as a box+strip per
category (distribution across datasets) with a paired t-test (by dataset) vs the reference.

    python strategy_comparison.py --results restr_method.json   --group_by strategy
    python strategy_comparison.py --results restr_heuristic.json --group_by rank_by

Outputs vis/strategy_<group_by>.png.
"""

import os
import re
import json
import math
import argparse
from collections import defaultdict

VIS_DIR = "vis"
METRICS = [("Cond_Clos_F1", "Cond Closure F1"),
           ("Cond_Clos_Precision", "Cond Closure Precision"),
           ("Cond_Clos_Recall", "Cond Closure Recall")]


def _strategy(method):
    if "+restructure_prune_only" in method: return "Prune-only"
    if "+restructure_ranked" in method:     return "Ranked"
    if "+restructure" in method:            return "Restructure"
    mm = re.search(r"clawback=(\d+)", method)
    k = int(mm.group(1)) if mm else 0
    return "Raw" if k == 0 else f"Clawback(k={k})"


def _rank_by(method):
    mm = re.search(r"\+rank_(\w+)", method)
    return mm.group(1) if mm else "combined"   # combined runs carry no +rank tag


def _strategy_sort(cat):
    base = {"Raw": 0, "Restructure": 3, "Ranked": 4, "Prune-only": 5}
    if cat in base: return (base[cat], cat)
    if cat.startswith("Clawback"): return (2, cat)
    return (9, cat)


_RB_ORDER = {c: i for i, c in enumerate(["combined", "leverage", "neighborhood", "self_agreement", "salience"])}


def load_rows(path, group_by):
    """-> list of {dataset, category, F1, Precision, Recall} for the relevant Our Method runs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for block in data:
        ds = str(block.get("dataset", "")).replace(".csv", "")
        for res in block.get("results", []):
            m = res.get("method", "")
            if "Our Method" not in m:
                continue
            if group_by == "rank_by" and "+restructure_prune_only" not in m:
                continue    # rank_by comparison is over prune-only runs only
            cat = _strategy(m) if group_by == "strategy" else _rank_by(m)
            rows.append({
                "dataset": ds, "category": cat,
                "F1": res.get("Cond_Clos_F1"), "Cond_Clos_F1": res.get("Cond_Clos_F1"),
                "Cond_Clos_Precision": res.get("Cond_Clos_Precision"),
                "Cond_Clos_Recall": res.get("Cond_Clos_Recall"),
            })
    return rows


# ---- scipy-free paired t-test (same routine as vis_benchmarks.py) ----
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3.0e-7, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS: break
    return h


def _betai(a, b, x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest(diffs):
    diffs = [d for d in diffs if d == d]
    n = len(diffs)
    if n < 2: return (float("nan"), 0, float("nan"))
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return ((float("inf") if mean != 0 else 0.0), n - 1, (0.0 if mean != 0 else 1.0))
    t = mean / math.sqrt(var / n); df = n - 1
    return (t, df, _betai(df / 2.0, 0.5, df / (df + t * t)))


def _stars(p):
    if p != p: return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def plot(rows, group_by, ref, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = {r["category"] for r in rows}
    order = (sorted(cats, key=_strategy_sort) if group_by == "strategy"
             else sorted(cats, key=lambda c: (_RB_ORDER.get(c, 99), c)))
    ref = ref or ("Raw" if group_by == "strategy" else "combined")
    if ref not in cats:
        ref = order[0]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 6))
    for ax, (key, title) in zip(axes, METRICS):
        # box+strip per category
        data = [[r[key] for r in rows if r["category"] == c and r[key] is not None] for c in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False,
                        medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0"); patch.set_alpha(0.35)
        import numpy as np
        rng = np.random.default_rng(0)
        for i, c in enumerate(order):
            ys = [r[key] for r in rows if r["category"] == c and r[key] is not None]
            ax.scatter(rng.uniform(i + 0.75, i + 1.25, len(ys)), ys, s=18, alpha=0.6, color=".25")
        # paired t-test of each category vs ref (by dataset)
        byds = defaultdict(dict)
        for r in rows:
            if r[key] is not None:
                byds[r["dataset"]][r["category"]] = r[key]
        ann = []
        for c in order:
            if c == ref: continue
            diffs = [v[c] - v[ref] for v in byds.values() if c in v and ref in v]
            t, df, p = paired_ttest(diffs)
            md = (sum(diffs) / len(diffs)) if diffs else float("nan")
            ann.append(f"{c} vs {ref}: Δ={md:+.3f} p={p:.2g}{_stars(p)} (n={len(diffs)})")
        ax.set_title(f"{title}\n" + ("\n".join(ann) if ann else ""), fontsize=8, fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)

    sub = "post-raw strategies vs Raw" if group_by == "strategy" else "prune-only by --rank_by (vs combined)"
    fig.suptitle(f"Our Method: {sub}", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare Our Method sub-strategies on P/R/F1.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--group_by", choices=["strategy", "rank_by"], default="strategy")
    ap.add_argument("--ref", default=None, help="Reference category (default: Raw / combined).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.results, args.group_by)
    if not rows:
        print(f"[!] No matching Our Method runs in {args.results} for --group_by {args.group_by}.")
        return
    cats = sorted({r["category"] for r in rows})
    print(f"[*] {len(rows)} rows across {len({r['dataset'] for r in rows})} datasets | categories: {', '.join(cats)}")
    out = args.out or os.path.join(VIS_DIR, f"strategy_{args.group_by}.png")
    plot(rows, args.group_by, args.ref, out)
    print(f"[*] wrote {out}")


if __name__ == "__main__":
    main()
