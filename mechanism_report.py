"""Clean per-mechanism data for the paper, from benchmark results files.

The earlier mechanism figures were degenerate because the PLOTTING layer existed but the
sweeps feeding it were never run (batch-size k: one value ever; clawback: only K=0 survived
in the results file; reasoning effort: a server-side knob the stack silently ignored). This
tool is the dedicated analysis layer for the two mechanism experiments that ARE cleanly
runnable, and it fails loudly when the data it needs is missing instead of plotting a
one-point "sweep".

Modes:

  compare  -- the CALL-STRUCTURE decomposition. Takes three results files for one model:
              the structured method, the best-practice single call, and the matched-semantics
              single call (main.py --sc_style matched). Reports three paired contrasts over
              the shared datasets:
                structured vs matched      -> call structure, prompt content held fixed
                matched    vs bestpractice -> prompt content, call count held fixed
                structured vs bestpractice -> the original bundled comparison
              python mechanism_report.py compare --structured control_gemma4N.json \
                     --bestpractice sc_gemma4_prim_merge.json --matched sc_matched_gemma4.json \
                     --tag gemma4

  ksweep   -- reasoning-per-decision dose response. Takes results file(s) holding structured
              runs at several --chunk_size values (labels carry "(k=<K>,"), plots F1 vs k per
              dataset with the largest k as reference, and refuses to report a "sweep" of one.
              python mechanism_report.py ksweep --results ksweep_gemma4.json \
                     --results control_gemma4N.json --tag gemma4

Outputs land in results/mechanisms/: <mode>_<tag>.csv and <mode>_<tag>.png.
"""

import os
import re
import json
import argparse
from collections import defaultdict

OUT_DIR = os.path.join("results", "mechanisms")
METRIC = "Cond_Clos_F1"
K_RE = re.compile(r"\(k=(\d+)")


# ----------------------------------------------------------------------------
# Loading and pure helpers (covered by test_mechanism_report)
# ----------------------------------------------------------------------------

def load_results(path):
    """-> {dataset: {method_label: row}} from one benchmark results file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = defaultdict(dict)
    for block in data:
        for row in block.get("results", []):
            out[block["dataset"]][row.get("method", "?")] = row
    return dict(out)


def pick_row(rows_by_label, want):
    """Select one method row per dataset.

    want='structured'   -> an 'Our Method' row without restructure/variant tags
    want='singlecall'   -> a 'Single-Call Baseline' row WITHOUT the [matched] tag
    want='matched'      -> the 'Single-Call Baseline [matched]' row
    Returns None when absent; ambiguity resolves to the shortest label (the plain run).
    """
    def ok(lbl):
        if want == "structured":
            return lbl.startswith("Our Method") and "+" not in lbl and "[" not in lbl
        if want == "matched":
            return lbl.startswith("Single-Call Baseline [matched]")
        return lbl.startswith("Single-Call Baseline") and "[matched]" not in lbl
    hits = sorted((lbl for lbl in rows_by_label if ok(lbl)), key=len)
    return rows_by_label[hits[0]] if hits else None


def extract_k(label):
    """-> chunk size k from a structured-method label, or None."""
    m = K_RE.search(label)
    return int(m.group(1)) if m else None


def paired_stats(deltas):
    """-> (mean, t_p, wilcoxon_p, n_up) for a list of per-dataset deltas."""
    from scipy import stats
    mean = sum(deltas) / len(deltas)
    t_p = stats.ttest_1samp(deltas, 0).pvalue if len(deltas) > 1 else float("nan")
    try:
        w_p = stats.wilcoxon(deltas).pvalue
    except ValueError:
        w_p = float("nan")
    return mean, t_p, w_p, sum(1 for d in deltas if d > 0)


# ----------------------------------------------------------------------------
# compare mode
# ----------------------------------------------------------------------------

def run_compare(args):
    structured = load_results(args.structured)
    best = load_results(args.bestpractice)
    matched = load_results(args.matched)

    rows = []
    for ds in sorted(set(structured) & set(best) & set(matched)):
        s = pick_row(structured[ds], "structured")
        b = pick_row(best[ds], "singlecall")
        m = pick_row(matched[ds], "matched")
        if not (s and b and m):
            print(f"    [!] {ds}: missing a row (structured={bool(s)} best={bool(b)} "
                  f"matched={bool(m)}) -- skipped")
            continue
        rows.append({"dataset": ds, "structured": s[args.metric],
                     "matched": m[args.metric], "bestpractice": b[args.metric]})
    if len(rows) < 3:
        raise SystemExit(f"[!] only {len(rows)} datasets have all three runs -- not enough "
                         f"for a paired comparison. Run the missing conditions first.")

    print(f"\n=== call-structure decomposition [{args.metric}], n={len(rows)} datasets ===")
    print(f"  {'dataset':26s} {'struct':>7s} {'matched':>8s} {'bestpr':>7s}")
    for r in rows:
        print(f"  {r['dataset']:26s} {r['structured']:7.3f} {r['matched']:8.3f} "
              f"{r['bestpractice']:7.3f}")

    contrasts = [("structured_vs_matched", "structured", "matched", "call structure (content held fixed)"),
                 ("matched_vs_bestpractice", "matched", "bestpractice", "prompt content (one call both)"),
                 ("structured_vs_bestpractice", "structured", "bestpractice", "the original bundle")]
    stat_rows = []
    for name, a, b, meaning in contrasts:
        d = [r[a] - r[b] for r in rows]
        mean, t_p, w_p, up = paired_stats(d)
        stat_rows.append({"contrast": name, "isolates": meaning, "mean_delta": round(mean, 4),
                          "t_p": round(t_p, 4), "wilcoxon_p": round(w_p, 4),
                          "datasets_up": f"{up}/{len(d)}"})
        print(f"  {name:28s} mean d={mean:+.4f}  t-p={t_p:.4f}  w-p={w_p:.4f}  ({up}/{len(d)} up)")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"compare_{args.tag}")
    _write_csv(base + ".csv", rows + [{}] + stat_rows)
    _plot_compare(rows, args.metric, base + ".png")
    print(f"[*] wrote {base}.csv and {base}.png")


def _plot_compare(rows, metric, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(rows)), 5.5))
    for off, key, color in [(-0.27, "structured", "#4C72B0"), (0.0, "matched", "#DD8452"),
                            (0.27, "bestpractice", "#55A868")]:
        ax.bar(x + off, [r[key] for r in rows], 0.25, label=key, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([r["dataset"].replace("_SUB", "") for r in rows], rotation=25, ha="right")
    ax.set_ylabel(metric, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Call-structure decomposition: structured vs matched-semantics vs "
                 "best-practice single call", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# ksweep mode
# ----------------------------------------------------------------------------

def collect_ksweep(result_maps, metric=METRIC):
    """-> {dataset: {k: score}} across one or more loaded results maps."""
    table = defaultdict(dict)
    for rm in result_maps:
        for ds, rows in rm.items():
            for lbl, row in rows.items():
                if not lbl.startswith("Our Method") or "+" in lbl or "[" in lbl:
                    continue
                k = extract_k(lbl)
                if k is not None and metric in row:
                    table[ds][k] = row[metric]
    return dict(table)


def run_ksweep(args):
    table = collect_ksweep([load_results(p) for p in args.results], args.metric)
    ks = sorted({k for v in table.values() for k in v})
    if len(ks) < 2:
        raise SystemExit(f"[!] found only k={ks} across the input files -- that is not a "
                         f"sweep. Run main.py --method our_method --chunk_size <K> for more "
                         f"values first (this is exactly how the old figure went wrong).")
    datasets = sorted(ds for ds, v in table.items() if len(v) >= 2)
    print(f"\n=== k sweep [{args.metric}]: k values {ks}, {len(datasets)} datasets ===")
    ref = max(ks)
    stat_rows = []
    for k in ks:
        pair = [(table[ds][k], table[ds][ref]) for ds in datasets
                if k in table[ds] and ref in table[ds]]
        if k == ref or len(pair) < 3:
            continue
        d = [a - b for a, b in pair]
        mean, t_p, w_p, up = paired_stats(d)
        stat_rows.append({"k": k, "vs_ref_k": ref, "mean_delta": round(mean, 4),
                          "t_p": round(t_p, 4), "wilcoxon_p": round(w_p, 4),
                          "datasets_up": f"{up}/{len(d)}"})
        print(f"  k={k:4d} vs k={ref}: mean d={mean:+.4f}  t-p={t_p:.4f}  ({up}/{len(d)} up)")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"ksweep_{args.tag}")
    flat = [{"dataset": ds, "k": k, args.metric: v}
            for ds, kv in sorted(table.items()) for k, v in sorted(kv.items())]
    _write_csv(base + ".csv", flat + [{}] + stat_rows)
    _plot_ksweep(table, ks, args.metric, base + ".png")
    print(f"[*] wrote {base}.csv and {base}.png")


def _plot_ksweep(table, ks, metric, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ds, kv in sorted(table.items()):
        xs = sorted(kv)
        ax.plot(xs, [kv[k] for k in xs], marker="o", alpha=0.5, lw=1,
                label=ds.replace("_SUB", ""))
    means = {k: sum(v[k] for v in table.values() if k in v) /
                max(1, sum(1 for v in table.values() if k in v)) for k in ks}
    ax.plot(sorted(means), [means[k] for k in sorted(means)], marker="s", color="black",
            lw=2.5, label="mean")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("chunk size k (candidates per call; smaller k = more calls, "
                  "more reasoning per candidate)", fontweight="bold")
    ax.set_ylabel(metric, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Reasoning-per-decision dose response", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------

def _write_csv(path, rows):
    import csv
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Clean per-mechanism analysis from results files.")
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("compare", help="call-structure decomposition")
    c.add_argument("--structured", required=True)
    c.add_argument("--bestpractice", required=True)
    c.add_argument("--matched", required=True)
    c.add_argument("--metric", default=METRIC)
    c.add_argument("--tag", required=True)
    c.add_argument("--out_dir", default=OUT_DIR)

    k = sub.add_parser("ksweep", help="chunk-size dose response")
    k.add_argument("--results", action="append", required=True,
                   help="Results file(s); repeatable so the k=1000 reference can come from "
                        "an existing control file.")
    k.add_argument("--metric", default=METRIC)
    k.add_argument("--tag", required=True)
    k.add_argument("--out_dir", default=OUT_DIR)

    args = ap.parse_args()
    (run_compare if args.mode == "compare" else run_ksweep)(args)


if __name__ == "__main__":
    main()
