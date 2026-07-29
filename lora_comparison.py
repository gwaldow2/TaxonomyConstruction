"""Compare LoRA fine-tuning across base models AND conditions in one figure.

Answers the two questions the pairwise model_comparison.py can't put side by side:
  1. Does tuning help at all?          -- each base's tuned runs vs ITS OWN untuned control
  2. Does the starting model matter?   -- the size of that lift, base vs base

Row 1 plots absolute scores (how good is each combination) and row 2 the per-dataset PAIRED
DELTA vs that base's control (did tuning help), with a paired t-test per group. The delta row
is the one that answers "did fine-tuning work": absolute scores mostly reflect base-model
strength, so a 7B and a 31B are not comparable on row 1 alone -- but their lifts are.

Each run is passed as BASE:CONDITION:PATH, so any number of bases and conditions work:

    python lora_comparison.py \
        --run qwen:control:control_qwen.json \
        --run qwen:in-domain:indomain_qwen.json \
        --run qwen:cross-domain:crossdomain_qwen.json \
        --run gemma4:control:control_gemma4.json \
        --run gemma4:in-domain:indomain_gemma4.json \
        --run gemma4:cross-domain:crossdomain_gemma4.json

Outputs vis/lora_comparison.png. The control condition name is --control (default 'control').
"""

import os
import json
import argparse
from collections import defaultdict

from model_comparison import paired_ttest, _p_stars

VIS_DIR = "vis"
METRICS = [("Cond_Clos_F1", "Cond Closure F1"),
           ("Cond_Clos_Precision", "Cond Closure Precision"),
           ("Cond_Clos_Recall", "Cond Closure Recall")]


def load_runs(specs, label=None):
    """-> {(base, condition): {dataset: metrics}} from BASE:CONDITION:PATH strings."""
    out = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 3:
            raise SystemExit(f"[!] --run must be BASE:CONDITION:PATH, got {spec!r}")
        base, condition, path = parts[0], parts[1], ":".join(parts[2:])
        if not os.path.exists(path):
            raise SystemExit(f"[!] {path} not found (from --run {spec})")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        per_ds = {}
        for block in data:
            ds = str(block.get("dataset", "")).replace(".csv", "")
            results = block.get("results", [])
            if label:
                results = [r for r in results if label in r.get("method", "")]
            if results:
                per_ds[ds] = results[-1]
        if not per_ds:
            print(f"    [!] {path} contributed no datasets"
                  + (f" matching --label {label!r}" if label else ""))
        out[(base, condition)] = per_ds

        # Which model actually produced these numbers. Runs made before main.py started
        # recording it have no 'model' key -- reported as 'unknown' rather than failing, since
        # those results stay valid and are identifiable by provenance.
        models = {r.get("model") for r in per_ds.values()}
        known = sorted(m for m in models if m)
        if not known:
            shown = "unknown (predates model logging)"
        elif len(known) == 1 and None not in models:
            shown = known[0]
        else:
            shown = " + ".join(known + (["unknown"] if None in models else []))
        print(f"    {base:10s} {condition:14s} {len(per_ds):2d} dataset(s)  model={shown}  <- {path}")
        if len(known) > 1 or (known and None in models):
            print(f"      [!] this file mixes models -- it was written by more than one run. "
                  f"Each base/condition should have its own --results_file.")
    return out


def plot(runs, control, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bases = sorted({b for b, _ in runs})
    conditions = sorted({c for _, c in runs}, key=lambda c: (c != control, c))
    tuned = [c for c in conditions if c != control]
    if not tuned:
        raise SystemExit(f"[!] only the control condition {control!r} was supplied -- nothing to compare.")

    fig, axes = plt.subplots(2, len(METRICS), figsize=(6.2 * len(METRICS), 10))
    if len(METRICS) == 1:
        axes = axes.reshape(2, 1)
    colors = plt.cm.tab10.colors

    for col, (key, title) in enumerate(METRICS):
        # ---- row 1: absolute scores, mean over datasets ----
        ax = axes[0][col]
        width = 0.8 / max(1, len(conditions))
        for ci, cond in enumerate(conditions):
            xs, ys = [], []
            for bi, base in enumerate(bases):
                vals = [r.get(key) for r in runs.get((base, cond), {}).values() if r.get(key) is not None]
                xs.append(bi + (ci - (len(conditions) - 1) / 2) * width)
                ys.append(sum(vals) / len(vals) if vals else float("nan"))
            ax.bar(xs, ys, width * 0.92, label=cond, color=colors[ci % 10],
                   alpha=0.85 if cond != control else 0.55,
                   hatch="//" if cond == control else None)
        ax.set_xticks(range(len(bases))); ax.set_xticklabels(bases)
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title} — absolute (mean over datasets)", fontsize=10, fontweight="bold")
        if col == 0:
            ax.legend(fontsize=8)

        # ---- row 2: paired delta vs each base's OWN control ----
        ax = axes[1][col]
        groups, labels, ann = [], [], []
        for base in bases:
            ctrl = runs.get((base, control), {})
            for cond in tuned:
                cur = runs.get((base, cond), {})
                shared = sorted(set(ctrl) & set(cur))
                deltas = [cur[d][key] - ctrl[d][key] for d in shared
                          if cur[d].get(key) is not None and ctrl[d].get(key) is not None]
                groups.append(deltas)
                labels.append(f"{base}\n{cond}")
                if deltas:
                    t, dfree, p = paired_ttest(deltas)
                    md = sum(deltas) / len(deltas)
                    ann.append(f"{base}/{cond}: Δ={md:+.3f} p={p:.2g}{_p_stars(p)} (n={len(deltas)})")
                else:
                    ann.append(f"{base}/{cond}: no shared datasets with its control")

        if any(groups):
            # Tick labels are set separately rather than passed to boxplot(): matplotlib
            # dropped the `labels=` kwarg (renamed `tick_labels=`), and set_xticklabels works
            # on every version, old and new.
            bp = ax.boxplot([g or [0] for g in groups], patch_artist=True,
                            showfliers=False, medianprops={"color": "black"})
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels(labels)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[(i // max(1, len(tuned))) % 10]); patch.set_alpha(0.35)
            rng = np.random.default_rng(0)
            for i, g in enumerate(groups):
                if g:
                    ax.scatter(rng.uniform(i + 0.78, i + 1.22, len(g)), g, s=20, alpha=0.7, color=".25")
        ax.axhline(0.0, ls=":", color="crimson", lw=1.6)
        ax.set_ylabel(f"Δ {title} vs own control", fontweight="bold")
        ax.set_title(f"Δ {title} vs '{control}'  (>0 = tuning helped)\n" + "\n".join(ann),
                     fontsize=8, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("LoRA fine-tuning by base model: absolute scores (top) and lift over each "
                 "base's own untuned control (bottom)", y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare LoRA results across base models and conditions.")
    ap.add_argument("--run", action="append", required=True, metavar="BASE:CONDITION:PATH",
                    help="Repeatable. e.g. --run gemma4:in-domain:indomain_gemma4.json")
    ap.add_argument("--control", default="control", help="Condition name treated as the control.")
    ap.add_argument("--label", default=None,
                    help="Only use runs whose method label contains this substring.")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "lora_comparison.png"))
    args = ap.parse_args()

    print("[*] loading runs:")
    runs = load_runs(args.run, args.label)
    bases = sorted({b for b, _ in runs})
    missing = [b for b in bases if (b, args.control) not in runs]
    if missing:
        raise SystemExit(f"[!] no '{args.control}' run for base(s): {', '.join(missing)}. "
                         f"Each base needs its own untuned control -- a control from a different "
                         f"base measures the model, not the tuning.")
    plot(runs, args.control, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
