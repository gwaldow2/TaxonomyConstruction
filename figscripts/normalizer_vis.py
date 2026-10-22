"""Two-panel figure: what structured prompting normalizes across models, and what it reveals."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
OUT = f"{BASE}/paper_figures/method_normalization.png"
CORE = ["WordNetFood_SUB", "CellOntology_SUB", "SemEvalFood_SUB", "LLMs4OL_OBI_SUB",
        "LLMs4OL_PO_SUB", "LLMs4OL_SchemaOrg_SUB", "LLMs4OL_MatOnto_SUB"]


def load(p, tag):
    out = {}
    for b in json.load(open(f"{BASE}/{p}", encoding="utf-8")):
        for r in b["results"]:
            if tag in r["method"]:
                out[b["dataset"]] = r["Cond_Clos_F1"]
    return out


st = {"Gemini 3.1 Pro": load("results/results_json/control_gemini.json", "Our Method"),
      "gpt-oss-120B": load("results/gptoss_20260910/control_gptoss.json", "Our Method"),
      "Gemma-4-31B": load("results/results_json/control_gemma4N.json", "Our Method"),
      "Qwen2.5-7B": load("results/results_json/control_qwenN.json", "Our Method")}
sc = {"Gemini 3.1 Pro": load("results/results_json/sc_gemini_prim_merge.json", "Single-Call"),
      "gpt-oss-120B": load("results/gptoss_retry_20260917/sc_gptoss_prim_merge.json", "Single-Call"),
      "Gemma-4-31B": load("results/results_json/sc_gemma4_prim_merge.json", "Single-Call"),
      "Qwen2.5-7B": load("results/results_json/sc_qwen_prim_merge.json", "Single-Call")}
COL = {"Gemini 3.1 Pro": "C3", "gpt-oss-120B": "C1", "Gemma-4-31B": "C0", "Qwen2.5-7B": "C2"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# --- Panel A: slope chart of mean F1, single-call -> structured -------------
NAME_OFF = {"Gemini 3.1 Pro": 9, "Gemma-4-31B": -4, "gpt-oss-120B": 0, "Qwen2.5-7B": 0}
END_OFF = {"Gemini 3.1 Pro": 0, "gpt-oss-120B": 7, "Gemma-4-31B": -7, "Qwen2.5-7B": 0}
for m in st:
    a = np.mean([sc[m][d] for d in CORE])
    b = np.mean([st[m][d] for d in CORE])
    ax1.plot([0, 1], [a, b], "o-", color=COL[m], lw=2.4, ms=7)
    ax1.annotate(f"{m}  {a:.3f}", (0, a), textcoords="offset points",
                 xytext=(-8, NAME_OFF[m]), ha="right", va="center", fontsize=9,
                 color=COL[m], fontweight="bold")
    ax1.annotate(f"{b:.3f}", (1, b), textcoords="offset points",
                 xytext=(8, END_OFF[m]), ha="left", va="center", fontsize=9, color=COL[m])
ax1.set_xlim(-0.55, 1.35)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["single-call", "structured"], fontsize=10, fontweight="bold")
ax1.set_ylabel("mean Cond. Closure F1 (7-dataset core)", fontweight="bold")
ax1.set_title("Mean scores: one call compresses capable models\ntoward its ceiling; structure resolves capability",
              fontsize=9.5, fontweight="bold")
ax1.grid(axis="y", color="0.93")

# --- Panel B: per-dataset between-model SD (3 capable models) ---------------
cap = ["Gemini 3.1 Pro", "gpt-oss-120B", "Gemma-4-31B"]
sd_sc = [np.std([sc[m][d] for m in cap]) for d in CORE]
sd_st = [np.std([st[m][d] for m in cap]) for d in CORE]
for a, b in zip(sd_sc, sd_st):
    ax2.plot([0, 1], [a, b], "o-", color="0.65", lw=1.1, ms=4, zorder=2)
ax2.plot([0, 1], [np.mean(sd_sc), np.mean(sd_st)], "D-", color="0.1", lw=2.8, ms=9, zorder=3)
ax2.annotate(f"mean {np.mean(sd_sc):.3f}", (0, np.mean(sd_sc)), textcoords="offset points",
             xytext=(-8, 0), ha="right", fontsize=9, fontweight="bold")
ax2.annotate(f"mean {np.mean(sd_st):.3f}", (1, np.mean(sd_st)), textcoords="offset points",
             xytext=(8, 0), ha="left", fontsize=9, fontweight="bold")
ax2.set_xlim(-0.5, 1.5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["single-call", "structured"], fontsize=10, fontweight="bold")
ax2.set_ylabel("between-model SD of F1, per dataset", fontweight="bold")
ax2.set_title("Per-dataset spread among the three capable models:\nlower under structure on 6 of 7 datasets",
              fontsize=9.5, fontweight="bold")
ax2.grid(axis="y", color="0.93")

fig.suptitle("What structured prompting normalizes: scale and dataset idiosyncrasy --- not capability",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
