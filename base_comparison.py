"""Compare any number of base models on the untuned method, as PAIRED DELTAS vs a reference.

Neither existing comparison script fits this: model_comparison.py is strictly pairwise, and
lora_comparison.py needs tuned conditions to delta against a control -- with only untuned runs
it correctly refuses. This takes repeated --run LABEL:PATH and plots, per metric, each model's
RAW per-dataset scores as a box per model, with the paired t-test against the reference kept
in each panel title. The per-dataset deltas themselves are still printed to the console.

    python base_comparison.py \
        --run Gemma4-31B:control_gemma4N.json \
        --run Qwen2.5-7B:control_qwenN.json \
        --run "Gemini 3.1 Pro:control_gemini.json"

The FIRST --run is the reference unless --ref names another label. Deltas are computed only on
datasets present in both files of a pair, and each panel is annotated with a paired t-test.
Outputs vis/base_comparison.png.
"""

import os
import json
import argparse

from model_comparison import paired_ttest, _p_stars

VIS_DIR = "vis"
METRICS = [("Cond_Clos_F1", "Cond Closure F1"),
           ("Cond_Clos_Precision", "Cond Closure Precision"),
           ("Cond_Clos_Recall", "Cond Closure Recall")]


def load_runs(specs):
    """-> ordered [(label, {dataset: result})]. Spec is LABEL:PATH; label may contain spaces."""
    out = []
    for spec in specs:
        label, sep, path = spec.partition(":")   # first colon: paths may contain ":" (C:/...)
        if not sep or not label:
            raise SystemExit(f"[!] --run must be LABEL:PATH, got {spec!r}")
        if not os.path.exists(path):
            raise SystemExit(f"[!] {path} not found (from --run {spec})")
        per_ds = {}
        for block in json.load(open(path, encoding="utf-8")):
            ds = str(block.get("dataset", "")).replace(".csv", "")
            results = block.get("results", [])
            if results:
                per_ds[ds] = results[-1]
        models = sorted({r.get("model") for r in per_ds.values() if r.get("model")})
        shown = models[0] if len(models) == 1 else (" + ".join(models) if models else "unknown")
        print(f"    {label:16s} {len(per_ds):2d} dataset(s)  model={shown}  <- {path}")
        out.append((label, per_ds))
    return out


def deltas_vs_ref(runs, ref_label, key):
    """-> [(label, shared_datasets, [delta...])] for every non-reference run."""
    ref = dict(runs)[ref_label]
    out = []
    for label, cur in runs:
        if label == ref_label:
            continue
        shared = sorted(d for d in ref if d in cur
                        and ref[d].get(key) is not None and cur[d].get(key) is not None)
        out.append((label, shared, [cur[d][key] - ref[d][key] for d in shared]))
    return out


def plot(runs, ref_label, out_path):
    """Boxes are RAW per-dataset scores, one box per model, ALL models including the
    reference. The paired t-test vs the reference stays in each panel title, so the figure
    shows the absolute level of every model while the annotation carries the significance
    of the differences -- the deltas themselves are printed by main()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.2 * len(METRICS), 6))
    colors = plt.cm.tab10.colors
    for ax, (key, title) in zip(axes, METRICS):
        labels = [label for label, _ in runs]
        data = [[r[key] for r in per_ds.values() if r.get(key) is not None]
                for _, per_ds in runs]
        if any(data):
            bp = ax.boxplot([d or [0] for d in data], patch_artist=True, showfliers=False,
                            medianprops={"color": "black"})
            ax.set_xticks(range(1, len(data) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[i % 10])
                patch.set_alpha(0.35)
            rng = np.random.default_rng(0)
            for i, d in enumerate(data):
                if d:
                    ax.scatter(rng.uniform(i + 0.78, i + 1.22, len(d)), d, s=22, alpha=0.7, color=".25")
        ann = []
        for label, shared, d in deltas_vs_ref(runs, ref_label, key):
            if d:
                t, dfree, p = paired_ttest(d)
                ann.append(f"{label} vs {ref_label}: Δ={sum(d)/len(d):+.3f} p={p:.2g}{_p_stars(p)} (n={len(d)})")
            else:
                ann.append(f"{label}: no shared datasets with {ref_label}")
        ax.set_title(f"{title}\n" + "\n".join(ann), fontsize=9, fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_ylim(0, 1)

    fig.suptitle("Untuned base method: raw per-dataset scores "
                 f"(paired t-tests vs '{ref_label}' in titles)",
                 y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Paired-delta comparison of base models (untuned).")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:PATH",
                    help="Repeatable. The first is the reference unless --ref is given.")
    ap.add_argument("--ref", default=None, help="Label of the reference model.")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "base_comparison.png"))
    args = ap.parse_args()

    print("[*] loading runs:")
    runs = load_runs(args.run)
    if len(runs) < 2:
        raise SystemExit("[!] need at least two --run entries to compare.")
    labels = [l for l, _ in runs]
    ref = args.ref or labels[0]
    if ref not in labels:
        raise SystemExit(f"[!] --ref {ref!r} is not among: {', '.join(labels)}")

    print(f"[*] reference: {ref}")
    for key, title in METRICS:
        for label, shared, d in deltas_vs_ref(runs, ref, key):
            if d:
                t, dfree, p = paired_ttest(d)
                print(f"    {title:24s} {label:16s} delta={sum(d)/len(d):+.4f}  p={p:.3g}{_p_stars(p)}  n={len(d)}")
            else:
                print(f"    {title:24s} {label:16s} (no shared datasets)")
    plot(runs, ref, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
