"""Two-judge agreement and consensus contamination for the relation-type audit.

Joins two (or more) relation_type_audit.py output CSVs on (dataset, source, parent, child)
and reports, for the edges every judge classified:

  * inter-judge agreement on the full twelve-type label and on the binary is-a / non-is-a
    decision (raw percent agreement + Cohen's kappa), with the GT confusion matrix;
  * the consensus contamination table: per dataset, each judge's non-is-a rate and the
    consensus rate (non-is-a only when EVERY judge says non-is-a -- the same unanimity rule
    isa_rescore.py filters with), with Wilson 95% CIs on the consensus rate;
  * the votes robustness join: among predicted edges scored as false positives, the share
    judged is_a per self-agreement bucket (votes 1 vs 2), per judge and under both-judge
    consensus.

Rows with an empty justification are failed judge calls (the endpoint returned nothing and
'unrelated' is a parser fallback, not a verdict); they are skipped with a warning, and a CSV
made only of such rows is refused.

    python judge_agreement.py --types_csv results/relation_types.csv \
                              --types_csv results/relation_types_gemini.csv

Outputs results/judge_agreement.csv (pairwise stats), results/judge_consensus_contamination.csv
(the consensus table), results/judge_votes_join.csv and vis/judge_agreement.png.
"""

import os
import csv
import glob
import math
import argparse
from collections import Counter, defaultdict

from relation_type_audit import RESULTS_DIR, VIS_DIR, CANONICAL_TYPES

KEEP_TYPES = ("is_a",)


def wilson(k, n, z=1.96):
    """Wilson score 95% interval for a binomial proportion -> (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(pairs):
    """pairs: [(label_a, label_b)] -> (raw percent agreement, Cohen's kappa)."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / (n * n)
    return po, 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def load_judged(path):
    """-> ({(dataset, source, parent, child): row}, judge model name).

    Failed rows (empty justification) are skipped: the judge's response was empty, so the
    stored 'unrelated' is a parser fallback, not a judgment.
    """
    if not os.path.exists(path):
        raise SystemExit(f"[!] {path} not found -- run relation_type_audit.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good = [r for r in rows if r.get("justification", "").strip()]
    if len(good) < len(rows):
        print(f"    [!] {path}: skipping {len(rows) - len(good)} failed rows "
              f"(empty justification)")
    if not good:
        raise SystemExit(f"[!] {path}: every row is a failed judge call -- re-run the audit.")
    model = Counter(r.get("model", "?") for r in good).most_common(1)[0][0]
    return {(r["dataset"], r["source"], r["parent"], r["child"]): r for r in good}, model


def join_judged(maps, datasets=None):
    """-> {key: [type per judge]} for the keys EVERY judge classified."""
    keys = set(maps[0])
    for m in maps[1:]:
        keys &= set(m)
    if datasets:
        keys = {k for k in keys if k[0] in datasets}
    return {k: [m[k]["type"] for m in maps] for k in sorted(keys)}


def consensus_nonisa(types):
    """The unanimity rule isa_rescore.py filters with: non-is-a only if every judge agrees."""
    return all(t not in KEEP_TYPES for t in types)


def consensus_table(joined):
    """GT-edge consensus contamination -> ordered [(dataset, stats dict)], POOLED last."""
    per = defaultdict(list)
    for (ds, source, _, _), types in joined.items():
        if source == "gt":
            per[ds].append(types)
    out = []
    for ds in sorted(per) + ["POOLED"]:
        rows = [t for d in per for t in per[d]] if ds == "POOLED" else per[ds]
        n = len(rows)
        n_judges = len(rows[0]) if rows else 0
        by_judge = [sum(t[j] not in KEEP_TYPES for t in rows) for j in range(n_judges)]
        k = sum(consensus_nonisa(t) for t in rows)
        lo, hi = wilson(k, n)
        out.append((ds, {"n": n, "by_judge": by_judge, "consensus": k,
                         "rate": k / n if n else 0.0, "ci_lo": lo, "ci_hi": hi}))
    return out


def load_votes(results_dir, dataset, label=None):
    """{(parent, child): votes} from the same diagnostics file the audit sampled from."""
    hits = [p for p in sorted(glob.glob(os.path.join(results_dir,
                                                     f"{dataset}_*_edge_diagnostics.csv")))
            if not label or label in os.path.basename(p)]
    if not hits:
        return {}
    with open(hits[-1], newline="", encoding="utf-8") as f:
        return {(r["parent"], r["child"]): int(float(r["votes"]))
                for r in csv.DictReader(f) if r.get("votes", "") != ""}


def votes_join(joined, maps, results_dir, pred_label=None):
    """Among predicted FPs, share judged is_a per votes bucket -> {bucket: stats}."""
    votes_by_ds = {}
    buckets = defaultdict(list)
    for key, types in joined.items():
        ds, source, p, c = key
        if source != "pred" or maps[0][key].get("is_fp", "") not in ("1", 1):
            continue
        if ds not in votes_by_ds:
            votes_by_ds[ds] = load_votes(results_dir, ds, pred_label)
        v = votes_by_ds[ds].get((p, c))
        if v is not None:
            buckets[v].append(types)
    out = {}
    for v in sorted(buckets):
        rows = buckets[v]
        n = len(rows)
        out[v] = {"n": n,
                  "by_judge": [sum(t[j] in KEEP_TYPES for t in rows) / n
                               for j in range(len(rows[0]))],
                  "all_judges": sum(all(t in KEEP_TYPES for t in types)
                                    for types in rows) / n}
    return out


def plot(table, joined, models, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    names = [ds for ds, _ in table]
    x = np.arange(len(names))
    w = 0.26
    for j in range(len(models)):
        ax1.bar(x + (j - 1) * w, [s["by_judge"][j] / s["n"] for _, s in table], w,
                label=models[j].split("/")[-1])
    cons = [s["rate"] for _, s in table]
    err = [[s["rate"] - s["ci_lo"] for _, s in table],
           [s["ci_hi"] - s["rate"] for _, s in table]]
    ax1.bar(x + (len(models) - 1) * w, cons, w, yerr=err, capsize=3,
            label="consensus", color="0.25")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("_SUB", "").replace("LLMs4OL_", "") for n in names],
                        rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("GT edges judged non-is-a", fontweight="bold")
    ax1.set_title("Contamination: per judge vs consensus (Wilson 95% CI)")
    ax1.legend(fontsize=8)

    gt_pairs = [(t[0], t[1]) for (_, s, _, _), t in joined.items() if s == "gt"]
    present = [t for t in CANONICAL_TYPES
               if any(a == t or b == t for a, b in gt_pairs)]
    mat = np.zeros((len(present), len(present)))
    idx = {t: i for i, t in enumerate(present)}
    for a, b in gt_pairs:
        mat[idx[a], idx[b]] += 1
    ax2.imshow(np.log1p(mat), cmap="Blues")
    for i in range(len(present)):
        for j in range(len(present)):
            if mat[i, j]:
                ax2.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7,
                         color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax2.set_xticks(range(len(present)))
    ax2.set_xticklabels(present, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(present)))
    ax2.set_yticklabels(present, fontsize=7)
    ax2.set_xlabel(models[1].split("/")[-1], fontweight="bold")
    ax2.set_ylabel(models[0].split("/")[-1], fontweight="bold")
    ax2.set_title("GT label confusion")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Judge agreement + consensus contamination.")
    ap.add_argument("--types_csv", action="append", default=None,
                    help="Relation-type CSVs from relation_type_audit.py; give one per judge. "
                         "Default: results/relation_types.csv + results/relation_types_gemini.csv")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Restrict to these datasets (default: everything all judges share).")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--pred_label", default=None,
                    help="Substring filter for the diagnostics file (votes join).")
    ap.add_argument("--agree_csv", default=os.path.join(RESULTS_DIR, "judge_agreement.csv"))
    ap.add_argument("--out_csv",
                    default=os.path.join(RESULTS_DIR, "judge_consensus_contamination.csv"))
    ap.add_argument("--votes_csv", default=os.path.join(RESULTS_DIR, "judge_votes_join.csv"))
    ap.add_argument("--out_png", default=os.path.join(VIS_DIR, "judge_agreement.png"))
    args = ap.parse_args()

    paths = args.types_csv or [os.path.join(RESULTS_DIR, "relation_types.csv"),
                               os.path.join(RESULTS_DIR, "relation_types_gemini.csv")]
    print("[*] loading judge CSVs:")
    loaded = [load_judged(p) for p in paths]
    maps = [m for m, _ in loaded]
    models = [name for _, name in loaded]
    if len(set(models)) < len(models):
        print(f"    [!] duplicate judge model across CSVs: {models} -- is one file stale?")

    joined = join_judged(maps, args.datasets)
    if not joined:
        raise SystemExit("[!] no edges classified by every judge -- nothing to compare.")
    gt = {k: t for k, t in joined.items() if k[1] == "gt"}
    print(f"[*] {len(joined)} edges judged by all {len(maps)} judges "
          f"({len(gt)} GT, {len(joined) - len(gt)} predicted)")

    # -- agreement (first two judges) --------------------------------------------------
    pairs = [(t[0], t[1]) for t in gt.values()]
    po_full, k_full = cohen_kappa(pairs)
    po_bin, k_bin = cohen_kappa([(a in KEEP_TYPES, b in KEEP_TYPES) for a, b in pairs])
    print(f"\n[*] GT-edge agreement, {models[0]} vs {models[1]} (n={len(pairs)}):")
    print(f"    12-type label : {po_full:.1%} raw, kappa={k_full:.3f}")
    print(f"    is-a / non-is-a: {po_bin:.1%} raw, kappa={k_bin:.3f}")
    conf = Counter(pairs)
    print("    disagreement cells (judge1 -> judge2):")
    for (a, b), n in conf.most_common():
        if a != b:
            print(f"        {a:16s} -> {b:16s} {n:4d}")
    with open(args.agree_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["judge1", "judge2", "n_gt", "raw_full", "kappa_full",
                    "raw_binary", "kappa_binary"])
        w.writerow([models[0], models[1], len(pairs), f"{po_full:.4f}", f"{k_full:.4f}",
                    f"{po_bin:.4f}", f"{k_bin:.4f}"])

    # -- consensus contamination table -------------------------------------------------
    table = consensus_table(joined)
    print("\n[*] consensus contamination (non-is-a only when every judge agrees):")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n"] + [f"nonisa_{m.split('/')[-1]}" for m in models]
                   + ["nonisa_consensus", "rate", "ci_lo", "ci_hi"])
        for ds, s in table:
            w.writerow([ds, s["n"]] + s["by_judge"]
                       + [s["consensus"], f"{s['rate']:.4f}",
                          f"{s['ci_lo']:.4f}", f"{s['ci_hi']:.4f}"])
            per_judge = " / ".join(f"{b}" for b in s["by_judge"])
            print(f"    {ds:24s} n={s['n']:4d}  judges {per_judge}  "
                  f"consensus {s['consensus']:3d} ({s['rate']:.1%} "
                  f"[{s['ci_lo']:.1%}, {s['ci_hi']:.1%}])")

    # -- votes robustness join ---------------------------------------------------------
    vj = votes_join(joined, maps, args.results_dir, args.pred_label)
    if vj:
        print("\n[*] predicted FPs judged is_a, by self-agreement bucket:")
        with open(args.votes_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["votes", "n_fp"] + [f"isa_{m.split('/')[-1]}" for m in models]
                       + ["isa_all_judges"])
            for v, s in vj.items():
                w.writerow([v, s["n"]] + [f"{x:.4f}" for x in s["by_judge"]]
                           + [f"{s['all_judges']:.4f}"])
                per_judge = " / ".join(f"{x:.1%}" for x in s["by_judge"])
                print(f"    votes={v}: n={s['n']:5d}  per judge {per_judge}  "
                      f"both {s['all_judges']:.1%}")
    else:
        print("\n[!] votes join skipped: no diagnostics matched the joined predicted FPs")

    try:
        plot(table, joined, models, args.out_png)
        print(f"\n[*] wrote {args.out_png}")
    except Exception as e:
        print(f"[!] figure skipped: {e}")
    print(f"[*] wrote {args.agree_csv}, {args.out_csv}"
          + (f", {args.votes_csv}" if vj else ""))


if __name__ == "__main__":
    main()
