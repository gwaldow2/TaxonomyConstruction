"""Delta figure: how is-a-filtered rescoring moves P/R/F1 under self, cross, consensus judges."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os
_TB = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
BASE = os.environ.get("TAXO_RESCORE", f"{_TB}/results/corrected_core/rescore")
OUT = f"{_TB}/paper_figures/isa_rescore_deltas.png"

FILTERS = [("self", "self judge\n(Gemma labels, {n} edges)", "C0"),
           ("cross", "cross judge\n(Gemini labels, {n} edges)", "C1"),
           ("consensus", "consensus\n(unanimous, {n} edges)", "0.25")]
METRICS = [("Precision", "$\\Delta$ Precision"), ("Recall", "$\\Delta$ Recall"),
           ("F1", "$\\Delta$ F1")]

data = {}
_named = []
for key, name, col in FILTERS:
    rows = list(csv.DictReader(open(f"{BASE}/rescore_{key}.csv", encoding="utf-8")))
    data[key] = {m: [float(r[f"filt_{m}"]) - float(r[f"orig_{m}"]) for r in rows]
                 for m, _ in METRICS}
    dropped = sum(int(r["dropped"]) for r in rows)
    _named.append((key, name.format(n=dropped), col))
FILTERS = _named

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
rng = np.random.default_rng(3)
for ax, (m, label) in zip(axes, METRICS):
    ax.axvline(0, color="0.55", lw=1)
    for yi, (key, name, col) in enumerate(FILTERS):
        d = np.array(data[key][m])
        t, p = stats.ttest_1samp(d, 0)
        jitter = rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(d, np.full(len(d), yi) + jitter, s=26, color=col, alpha=0.45, zorder=2)
        mean = d.mean()
        ax.plot([0, mean], [yi, yi], color=col, lw=2.2, zorder=3)
        ax.scatter([mean], [yi], s=130, color=col, marker="D", zorder=4,
                   edgecolor="white", linewidth=1.2)
        star = "*" if p < .05 else ""
        ax.annotate(f"{mean:+.3f}  p={p:.2f}{star}" if p >= .005 else
                    f"{mean:+.3f}  p={p:.3f}{star}",
                    (mean, yi + 0.38), ha="center", fontsize=8.5,
                    color=col if col != "0.25" else "0.15", fontweight="bold")
    ax.set_xlabel(label, fontweight="bold")
    ax.set_xlim(-0.11, 0.11)
    ax.grid(axis="x", color="0.92", zorder=0)
axes[0].set_yticks(range(len(FILTERS)))
axes[0].set_yticklabels([n for _, n, _ in FILTERS], fontsize=9)
axes[0].set_ylim(2.72, -0.45)
fig.suptitle("Is-a-filtered rescoring, change versus the unfiltered reference "
             "(Exp$\\_$Raw; Gemma-4 structured predictions, seven-dataset core; "
             "dots: datasets, diamonds: means)", fontsize=10.5, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
for key, _, _ in FILTERS:
    print(key, {m: f"{np.mean(v):+.4f}" for m, v in data[key].items()})
