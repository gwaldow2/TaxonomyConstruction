"""Method comparison bar chart on the seven-dataset core (Cond. Closure F1).
Reads the merged corrected-core jsons (build with merge_corrected_core.py first).
Also prints the tab:methods numbers.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
CC = f"{BASE}/results/corrected_core"
OUT = f"{BASE}/paper_figures/method_comparison.png"


def grab(path, method_sub):
    out = {}
    for b in json.load(open(path, encoding="utf-8")):
        for r in b["results"]:
            if method_sub in r["method"]:
                out[b["dataset"]] = r["Cond_Clos_F1"]
    return out


methods = {
    "Structured (ours)": grab(f"{CC}/gemma4_structured.json", "Our Method"),
    "Single-Call": grab(f"{CC}/gemma4_singlecall.json", "Single-Call"),
}
basel = defaultdict(dict)
for b in json.load(open(f"{CC}/baselines.json", encoding="utf-8")):
    for r in b["results"]:
        basel[r["method"].split(" (")[0]][b["dataset"]] = r["Cond_Clos_F1"]
for m in ("SBU LLM Batch", "TaxoLLaMA", "Lexical", "Vector", "SBU Embedding"):
    if m in basel:
        methods[m] = basel[m]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
names = list(methods)
means = [np.mean(list(methods[m].values())) for m in names]
order = np.argsort(means)[::-1]
for rank, idx in enumerate(order):
    m = names[idx]
    vals = list(methods[m].values())
    color = "C0" if "ours" in m else ("C1" if m == "Single-Call" else "0.6")
    ax.bar(rank, np.mean(vals), color=color, width=0.65)
    ax.plot([rank] * len(vals), vals, ".", color="k", ms=5, alpha=0.55)
    ax.annotate(f"{np.mean(vals):.3f}", (rank, np.mean(vals)), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([names[i] for i in order], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("mean Cond. Closure F1", fontweight="bold")
ax.set_title("Method comparison on the seven-dataset core\n"
             "(Gemma-4-31B serving every LLM method; dots: individual datasets)",
             fontsize=10.5, fontweight="bold")
ax.grid(axis="y", color="0.92")
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
print("\ntab:methods (Cond_Clos F1, seven-dataset core):")
for idx in order:
    m = names[idx]
    print(f"  {m:18s} {np.mean(list(methods[m].values())):.3f}  (n={len(methods[m])})")
