"""Three-way k-hop recall figure: structured vs direct-only vs matched single call."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
# historical pre-fix figure (frozen battery CSV); the corrected-core three-way
# version is khop_core.py's paper_figures/khop_recall.png
OUT = f"{BASE}/paper_figures/khop_recall_prefix_refs.png"

data = {}
for r in csv.DictReader(open(f"{BASE}/results/battery_20260925/khop_setoverlap.csv", encoding="utf-8")):
    data.setdefault((r["tag"], r["method"]), []).append((int(r["hop"]), float(r["recall"]), int(r["total"])))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
PANELS = [("gemma4b", "Gemma-4-31B"), ("gemini31b", "Gemini 3.1 Pro")]
STYLE = [("structured", "C0", "-", "o", "structured (ours)"),
         ("single-call", "C1", "--", "s", "single-call (direct-only prompt)"),
         ("single-call-matched", "C2", ":", "^", "single-call (matched ancestor prompt)")]
for ax, (tag, title) in zip(axes, PANELS):
    for meth, col, ls, mk, label in STYLE:
        pts = sorted(p for p in data[(tag, meth)] if p[2] >= 10)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=col, lw=2.4, ms=6, label=label)
    ax.set_xlabel("hop distance $k$ in ground-truth reduction", fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0.25, 1.03)
    ax.set_xticks(range(1, 9))
    ax.grid(color="0.92")
    ax.legend(fontsize=8.5, loc="upper left" if "Gemini" in title else "lower right",
              framealpha=0.95)
axes[0].set_ylabel("recall of $k$-hop closure pairs", fontweight="bold")
fig.suptitle("Recall by ancestor distance: the sag is the one-call structure, not the prompt content ---\n"
             "the matched call licenses ancestor edges yet sags like (or below) the direct-only call\n"
             "(seven-dataset core, synonym-aware matching; $n$ per hop: 689, 576, 447, 311, 191, 92, 36, 10)",
             fontsize=10.5, fontweight="bold", y=1.12)
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
