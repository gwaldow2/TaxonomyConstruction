"""Compare a method's run between two models (e.g. gpt-oss-120b vs gemma).

Run once per model -- swapping the model loaded in vLLM and pointing each run at its own
--results_file -- then this reads both files, takes the run in each dataset block (no
method-name filtering; feed files that contain only the runs you want to compare), and
plots Cond. Closure F1 / Precision / Recall / Runtime side by side, with a paired t-test
(by dataset) of the model difference.

    python model_comparison.py \
        --a results_gptoss.json --name_a gpt-oss-120b \
        --b results_gemma.json  --name_b gemma

Outputs vis/model_comparison.png.
"""

import os
import json
import math
import argparse

VIS_DIR = "vis"

METRICS = [("Cond_Clos_F1", "Cond Closure F1"),
           ("Cond_Clos_Precision", "Cond Closure Precision"),
           ("Cond_Clos_Recall", "Cond Closure Recall"),
           ("Runtime_sec", "Runtime (s)")]


def load_standard(path, label=None):
    """-> {dataset: result_dict}, one run per dataset block.

    No method-name filtering: feed a results file that contains ONLY the runs you want to
    compare (one per dataset). If a block happens to hold more than one run, the last is
    used; pass ``label`` to restrict to methods whose label contains that substring.
    """
    out, multi = {}, []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for block in data:
        ds = str(block.get("dataset", "")).replace(".csv", "")
        results = block.get("results", [])
        if label is not None:
            results = [r for r in results if label in r.get("method", "")]
        if not results:
            continue
        if len(results) > 1:
            multi.append(ds)
        out[ds] = results[-1]
    if multi:
        print(f"    [i] {os.path.basename(path)}: multiple runs in {multi}; used the last "
              f"(pass --label to pick a specific one).")
    return out


def all_method_labels(path):
    """All distinct method labels present in a results JSON (for diagnosing a no-match)."""
    labels = set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for block in data:
        for res in block.get("results", []):
            labels.add(res.get("method", ""))
    return sorted(labels)


# ---- scipy-free paired t-test (same routine as vis_benchmarks.py) ----
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3.0e-7, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
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
    """Two-sided paired t-test from paired differences. Returns (t, df, p)."""
    diffs = [d for d in diffs if d == d]
    n = len(diffs)
    if n < 2:
        return (float("nan"), 0, float("nan"))
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return ((float("inf") if mean != 0 else 0.0), n - 1, (0.0 if mean != 0 else 1.0))
    t = mean / math.sqrt(var / n)
    df = n - 1
    return (t, df, _betai(df / 2.0, 0.5, df / (df + t * t)))


def _p_stars(p):
    if p != p: return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def plot(a, b, name_a, name_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = sorted(set(a) & set(b))
    if not datasets:
        print("[!] No datasets are present in BOTH files. Nothing to compare.")
        return
    x = np.arange(len(datasets))
    w = 0.38
    ca, cb = "#4C72B0", "#DD8452"

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 6))
    for ax, (key, title) in zip(axes, METRICS):
        va = [float(a[d].get(key, float("nan"))) for d in datasets]
        vb = [float(b[d].get(key, float("nan"))) for d in datasets]
        ax.bar(x - w / 2, va, w, label=name_a, color=ca)
        ax.bar(x + w / 2, vb, w, label=name_b, color=cb)

        diffs = [bb - aa for aa, bb in zip(va, vb) if aa == aa and bb == bb]
        t, dfree, p = paired_ttest(diffs)
        md = sum(diffs) / len(diffs) if diffs else float("nan")
        ax.set_title(f"{title}\n{name_b}-{name_a}: Δ={md:+.3g} p={p:.2g}{_p_stars(p)} (n={len(diffs)})",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(title, fontweight="bold")
        if key != "Runtime_sec":
            ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle(f"Our Method (standard, clawback off): {name_a} vs {name_b}",
                 y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare Our Method's standard run between two models.")
    ap.add_argument("--a", required=True, help="Results JSON for model A")
    ap.add_argument("--b", required=True, help="Results JSON for model B")
    ap.add_argument("--name_a", default="model A")
    ap.add_argument("--name_b", default="model B")
    ap.add_argument("--label", default=None,
                    help="Optional: restrict to methods whose label CONTAINS this substring "
                         "(only needed if a dataset block holds more than one run).")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "model_comparison.png"))
    args = ap.parse_args()

    a, b = load_standard(args.a, args.label), load_standard(args.b, args.label)
    # If a file yielded nothing, show what IS in it (empty file, or --label filtered all out).
    for name, path, found in [(args.name_a, args.a, a), (args.name_b, args.b, b)]:
        if not found:
            print(f"[!] No runs found in {path} for {name}. Method labels present:")
            for lab in all_method_labels(path):
                print(f"      {lab!r}")
            print("    -> is the file empty / has empty result blocks, or did --label filter it all out?")
    both = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    print(f"[*] {args.name_a}: {len(a)} datasets | {args.name_b}: {len(b)} datasets | shared: {len(both)}")
    if only_a: print(f"    [!] only in {args.name_a} (skipped): {', '.join(only_a)}")
    if only_b: print(f"    [!] only in {args.name_b} (skipped): {', '.join(only_b)}")
    for d in both:
        fa, fb = a[d].get("Cond_Clos_F1", float("nan")), b[d].get("Cond_Clos_F1", float("nan"))
        win = args.name_a if fa > fb else args.name_b if fb > fa else "tie"
        print(f"    {d:24s} F1  {args.name_a}={fa:.3f}  {args.name_b}={fb:.3f}  -> {win}")
    plot(a, b, args.name_a, args.name_b, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
