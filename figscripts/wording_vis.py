"""Prompt-wording ablation figure: per-variant Cond_Clos F1 vs the 'full' control."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
OUT = f"{BASE}/paper_figures/prompt_wording_ablation.png"

per = defaultdict(dict)
for f in ["prompt_ablation.json", "variants.json", "ablation.json"]:
    for b in json.load(open(f"{BASE}/results/results_json/{f}", encoding="utf-8")):
        if b.get("use_synsets") or b.get("explode_nodes"):
            continue
        for r in b["results"]:
            tag = r["method"].split("[")[-1].split("]")[0] if "[" in r["method"] else None
            if tag:
                per[tag].setdefault(b["dataset"], r["Cond_Clos_F1"])

DESC = {"full": "full (control)", "full_json": "JSON output\n(format)",
        "oneway": "one direction\nonly", "no_conversational": "ban prose",
        "isa": "\u201cis a kind of\u201d", "no_example": "no example",
        "no_restriction": "no scoping\nrestriction", "direct": "\u201cparent/child\nconcept\u201d",
        "no_quantifier": "quantifier\ndropped"}
full = per["full"]
order = ["full"] + sorted([v for v in DESC if v != "full"],
                          key=lambda v: -np.mean(list(per[v].values())))

fig, ax = plt.subplots(figsize=(11, 4.6))
xs = np.arange(len(order))
for i, v in enumerate(order):
    ds = sorted(set(per[v]) & set(full))
    vals = [per[v][d] for d in ds]
    color = "C0" if v == "full" else ("0.75" if v != "no_quantifier" else "#d99")
    bp = ax.boxplot([vals], positions=[i], widths=0.55, patch_artist=True,
                    showfliers=False, medianprops={"color": "black"})
    bp["boxes"][0].set_facecolor(color)
    ax.scatter(np.full(len(vals), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(vals)),
               vals, s=14, color="0.35", zorder=3)
    m = np.mean(vals)
    ax.annotate(f"{m:.3f}", (i, 0.955), ha="center", fontsize=8.5, fontweight="bold")
    if v != "full":
        a = [full[d] for d in ds]
        t, p = stats.ttest_rel(vals, a)
        star = "*" if p < .05 else ""
        ax.annotate(f"d={m - np.mean(a):+.3f}\np={p:.2f}{star}" if p >= .005
                    else f"d={m - np.mean(a):+.3f}\np={p:.3f}{star}",
                    (i, 0.885), ha="center", fontsize=7.5,
                    color="#a33" if p < .05 else "0.3")
ax.axhline(np.mean(list(full.values())), color="C0", ls="--", lw=1, alpha=0.7)
ax.set_xticks(xs)
ax.set_xticklabels([DESC[v] for v in order], fontsize=8.5)
ax.set_ylabel("Cond. Closure F1", fontweight="bold")
ax.set_ylim(0.3, 1.0)
ax.set_title("Prompt-wording ablation: each variant changes one element of the structured prompt\n"
             "(gpt-oss-120b exploratory campaign, 10 datasets, paired $t$-tests vs. 'full'; only dropping the "
             "extensional quantifier moves the score)", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
