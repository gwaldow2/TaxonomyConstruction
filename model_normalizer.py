"""Does Our Method act as an F1 normalizer across models?

Compares the CROSS-MODEL F1 gap of a weak baseline (llm_zero) vs Our Method, per dataset.
If Our Method makes two very different models score more alike, its per-dataset gap
|F1(model A) - F1(model B)| is smaller than llm_zero's.

Run BOTH methods for BOTH models, one results file per model (each file holding llm_zero
AND the raw Our Method run), then:

    python model_normalizer.py --a results_gemma.json --name_a gemma \
                               --b results_gptoss.json --name_b gpt-oss-120b

Outputs vis/model_normalizer.png.
"""

import os
import json
import math
import argparse

VIS_DIR = "vis"


def _is_raw_our(m):
    return "Our Method" in m and "clawback=0" in m and "+" not in m and "[" not in m


def extract(path, pred):
    """-> {dataset: Cond_Clos_F1} for the first result whose method satisfies pred."""
    out = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for block in data:
        ds = str(block.get("dataset", "")).replace(".csv", "")
        for res in block.get("results", []):
            if pred(res.get("method", "")) and res.get("Cond_Clos_F1") is not None:
                out[ds] = res["Cond_Clos_F1"]
                break
    return out


def all_labels(path):
    labels = set()
    with open(path, encoding="utf-8") as f:
        for block in json.load(f):
            for res in block.get("results", []):
                labels.add(res.get("method", ""))
    return sorted(labels)


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


def gaps(a_file, b_file, our_pred, zero_pred):
    """Per shared dataset: {dataset: {'zero':(fa,fb,gap), 'our':(fa,fb,gap)}}."""
    zero_a, zero_b = extract(a_file, zero_pred), extract(b_file, zero_pred)
    our_a, our_b = extract(a_file, our_pred), extract(b_file, our_pred)
    out = {}
    for ds in sorted(set(zero_a) & set(zero_b) & set(our_a) & set(our_b)):
        out[ds] = {
            "zero": (zero_a[ds], zero_b[ds], abs(zero_a[ds] - zero_b[ds])),
            "our":  (our_a[ds], our_b[ds], abs(our_a[ds] - our_b[ds])),
        }
    return out, dict(zero_a=zero_a, zero_b=zero_b, our_a=our_a, our_b=our_b)


def plot(g, name_a, name_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = list(g.keys())
    gz = [g[d]["zero"][2] for d in datasets]
    go = [g[d]["our"][2] for d in datasets]
    diffs = [o - z for o, z in zip(go, gz)]     # negative = Our Method tighter
    t, df, p = paired_ttest(diffs)
    mz, mo = (sum(gz) / len(gz), sum(go) / len(go)) if gz else (float("nan"), float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Panel 1: per-dataset cross-model F1 gap, llm_zero vs Our Method
    ax = axes[0]
    x = np.arange(len(datasets)); w = 0.38
    ax.bar(x - w / 2, gz, w, label="llm_zero", color="#DD8452")
    ax.bar(x + w / 2, go, w, label="Our Method", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"|F1({name_a}) - F1({name_b})|", fontweight="bold")
    ax.set_title(f"Cross-model F1 gap per dataset (lower = more consistent)\n"
                 f"mean gap: llm_zero={mz:.3f}  Our Method={mo:.3f}", fontsize=10, fontweight="bold")
    ax.legend()

    # Panel 2: gap_zero vs gap_our; below the y=x line => Our Method tightened that dataset
    ax = axes[1]
    lim = max([0.01] + gz + go) * 1.1
    ax.plot([0, lim], [0, lim], "--", color=".5", linewidth=1)
    ax.scatter(gz, go, s=60, alpha=0.8, color="#4C72B0", edgecolors="black", linewidths=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("llm_zero cross-model gap", fontweight="bold")
    ax.set_ylabel("Our Method cross-model gap", fontweight="bold")
    below = sum(1 for z, o in zip(gz, go) if o < z)
    ax.set_title(f"Below the diagonal = Our Method is more consistent\n"
                 f"{below}/{len(datasets)} datasets tighter | mean Δgap={sum(diffs)/len(diffs):+.3f} "
                 f"p={p:.2g}{_stars(p)}", fontsize=10, fontweight="bold")

    fig.suptitle(f"Is Our Method an F1 normalizer across models?  ({name_a} vs {name_b})",
                 y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Does Our Method tighten cross-model F1 vs llm_zero?")
    ap.add_argument("--a", required=True, help="Results JSON for model A (has llm_zero + Our Method)")
    ap.add_argument("--b", required=True, help="Results JSON for model B")
    ap.add_argument("--name_a", default="model A")
    ap.add_argument("--name_b", default="model B")
    ap.add_argument("--method", default=None,
                    help="Substring for the Our Method run (default: the raw clawback=0 run).")
    ap.add_argument("--baseline", default="LLM Zero-Shot",
                    help="Substring for the weak baseline method label (default: 'LLM Zero-Shot').")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "model_normalizer.png"))
    args = ap.parse_args()

    our_pred = (lambda m: args.method in m) if args.method else _is_raw_our
    zero_pred = lambda m: args.baseline in m
    g, raw = gaps(args.a, args.b, our_pred, zero_pred)
    if not g:
        print("[!] No datasets have BOTH methods in BOTH files. Present labels:")
        for nm, pth in [(args.name_a, args.a), (args.name_b, args.b)]:
            print(f"    {nm} ({pth}):")
            for lab in all_labels(pth):
                print(f"        {lab!r}")
        print("    -> ensure each file has the baseline and Our Method run for the SAME datasets.")
        return
    print(f"[*] {len(g)} shared datasets. Cross-model F1 gap (|{args.name_a}-{args.name_b}|):")
    for ds in g:
        print(f"    {ds:24s} llm_zero={g[ds]['zero'][2]:.3f}  Our Method={g[ds]['our'][2]:.3f}")
    print(f"    mean gap: llm_zero={sum(g[d]['zero'][2] for d in g)/len(g):.3f}  "
          f"Our Method={sum(g[d]['our'][2] for d in g)/len(g):.3f}")
    plot(g, args.name_a, args.name_b, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
