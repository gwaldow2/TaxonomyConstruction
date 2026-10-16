"""V2 (full-source reference) figure set -> paper_figures/v2/."""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
P1 = f"{BASE}/results/v2phase1_20261016/results"
P2 = f"{BASE}/results/v2phase2_20261016/results"
OUT = f"{BASE}/paper_figures/v2"
os.makedirs(OUT, exist_ok=True)

V2 = ["SchemaOrgV2", "POV2", "OBIV2", "MatOntoV2"]
SHORT = {"SchemaOrgV2": "SchemaOrg", "POV2": "PO", "OBIV2": "OBI", "MatOntoV2": "MatOnto"}
LEGMAP = {"SchemaOrgV2": "LLMs4OL_SchemaOrg", "POV2": "LLMs4OL_PO",
          "OBIV2": "LLMs4OL_OBI", "MatOntoV2": "LLMs4OL_MatOnto"}
LEG3 = ["WordNetFood", "CellOntology", "SemEvalFood"]


def grab(path, methsub):
    out = {}
    for b in json.load(open(path, encoding="utf-8")):
        ds = b["dataset"].replace("_SUB", "")
        for r in b["results"]:
            if methsub in r["method"]:
                out[ds] = (r["Cond_Clos_Precision"], r["Cond_Clos_Recall"], r["Cond_Clos_F1"])
    return out


gm_leg = grab(f"{BASE}/results/results_json/control_gemma4N.json", "Our Method")
gi_leg = grab(f"{BASE}/results/results_json/control_gemini.json", "Our Method")
gm_v2 = grab(f"{P1}/v2_phase1.json", "Our Method")
gi_v2 = grab(f"{P2}/v2_phase2_gemini.json", "Our Method")

# ================= Figure 1: the reference repair =================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, (name, leg, v2) in zip(axes, (("Gemma-4-31B", gm_leg, gm_v2),
                                      ("Gemini 3.1 Pro", gi_leg, gi_v2))):
    xs = np.arange(4)
    for i, ds in enumerate(V2):
        lp, lr, lf = leg[LEGMAP[ds]]
        vp, vr, vf = v2[ds]
        ax.annotate("", xy=(i - 0.18, vp), xytext=(i - 0.18, lp),
                    arrowprops=dict(arrowstyle="-|>", color="C0", lw=2.4))
        ax.annotate("", xy=(i + 0.18, vf), xytext=(i + 0.18, lf),
                    arrowprops=dict(arrowstyle="-|>", color="C2", lw=2.4))
        ax.plot(i - 0.18, lp, "o", color="C0", ms=6, alpha=0.5)
        ax.plot(i + 0.18, lf, "o", color="C2", ms=6, alpha=0.5)
        ax.annotate(f"+{vp-lp:.2f}", (i - 0.18, vp), textcoords="offset points",
                    xytext=(-2, 6), ha="center", fontsize=8, color="C0", fontweight="bold")
        ax.annotate(f"+{vf-lf:.2f}", (i + 0.18, vf), textcoords="offset points",
                    xytext=(2, 6), ha="center", fontsize=8, color="C2", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[d] for d in V2])
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_ylim(0.35, 1.05)
    ax.grid(axis="y", color="0.92")
axes[0].set_ylabel("structured method, Cond. Closure", fontweight="bold")
axes[0].plot([], [], "o-", color="C0", label="precision (legacy $\\to$ full-source ref.)")
axes[0].plot([], [], "o-", color="C2", label="F1 (legacy $\\to$ full-source ref.)")
axes[0].legend(fontsize=8.5, loc="lower right", framealpha=0.95)
fig.suptitle("Repairing the reference: rebuilding the four LLMs4OL-derived ground truths from their full\n"
             "canonical sources lifts precision by $+$0.10 to $+$0.30 at roughly unchanged recall (same method, same model)",
             fontsize=10.5, fontweight="bold", y=1.06)
fig.tight_layout()
fig.savefig(f"{OUT}/reference_repair.png", dpi=250, bbox_inches="tight")
print("wrote reference_repair.png")

# ================= Figure 2: method comparison on the new 7-core =================
gm_sc_leg = grab(f"{BASE}/results/results_json/sc_gemma4_prim_merge.json", "Single-Call")
gm_sc_v2 = grab(f"{P1}/v2_phase1.json", "Single-Call")

def basel(path):
    out = defaultdict(dict)
    for b in json.load(open(path, encoding="utf-8")):
        ds = b["dataset"].replace("_SUB", "")
        for r in b["results"]:
            out[r["method"].split(" (")[0]][ds] = r["Cond_Clos_F1"]
    return out

b_leg = basel(f"{BASE}/results/baselines_20261002/results/baselines_7core.json")
b_v2 = basel(f"{P2}/v2_baselines.json")

methods = {
    "Structured (ours)": {**{d: gm_leg[d][2] for d in LEG3}, **{d: gm_v2[d][2] for d in V2}},
    "Single-Call": {**{d: gm_sc_leg[d][2] for d in LEG3}, **{d: gm_sc_v2[d][2] for d in V2}},
}
for m_old in ("SBU LLM Batch", "TaxoLLaMA", "Lexical", "Vector", "SBU Embedding"):
    methods[m_old] = {**{d: b_leg[m_old][d] for d in LEG3 if d in b_leg[m_old]},
                      **{d: b_v2[m_old][d] for d in V2 if d in b_v2[m_old]}}

fig2, ax = plt.subplots(figsize=(9.5, 4.6))
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
ax.set_ylabel("mean Cond. Closure F1 (new 7-core)", fontweight="bold")
ax.set_title("Method comparison on the repaired 7-dataset core (Gemma-4 serving every LLM method;\n"
             "dots: individual datasets; V2 full-source references for SchemaOrg/PO/OBI/MatOnto)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", color="0.92")
fig2.tight_layout()
fig2.savefig(f"{OUT}/method_comparison_v2.png", dpi=250, bbox_inches="tight")
print("wrote method_comparison_v2.png")
print("\nnew tab:methods (Gemma, new 7-core, Cond_Clos F1):")
for idx in order:
    m = names[idx]
    print(f"  {m:18s} {np.mean(list(methods[m].values())):.3f}  (n={len(methods[m])})")

# ================= Figure 3: V2 audit (contamination + FP verdicts) =================
def gt_lbl(path):
    return {(r["dataset"], r["parent"], r["child"]): r["type"]
            for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("source") == "gt"}

gm_lbl = gt_lbl(f"{P2}/rt_gemmaJ_v2_gemma4P.csv")
gi_lbl = gt_lbl(f"{P2}/rt_geminiJ_v2_gemma4P.csv")
common = sorted(set(gm_lbl) & set(gi_lbl))
n = len(common)
rates = {"Gemma judge": sum(gm_lbl[k] != "is_a" for k in common) / n,
         "Gemini judge": sum(gi_lbl[k] != "is_a" for k in common) / n,
         "Consensus": sum(gm_lbl[k] != "is_a" and gi_lbl[k] != "is_a" for k in common) / n}
legacy_rates = {"Gemma judge": 78 / 692, "Gemini judge": 50 / 692, "Consensus": 24 / 692}

def fp_rows(path):
    return {(r["dataset"], r["parent"], r["child"]): r["type"]
            for r in csv.DictReader(open(path, encoding="utf-8"))
            if r.get("source") == "pred" and r.get("is_fp") == "1"}

s = fp_rows(f"{P2}/rt_geminiJ_v2_gemini31P.csv")
c = fp_rows(f"{P2}/rt_gemmaJ_v2_gemini31P.csv")
com = sorted(set(s) & set(c))
fp = {"self judge\n(Gemini)": sum(s[k] == "is_a" for k in com) / len(com),
      "cross judge\n(Gemma)": sum(c[k] == "is_a" for k in com) / len(com),
      "both agree": sum(s[k] == "is_a" and c[k] == "is_a" for k in com) / len(com)}

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
x = np.arange(3)
ax1.bar(x - 0.18, [legacy_rates[k] for k in rates], width=0.36, color="0.65",
        label=f"legacy references (692 edges)")
ax1.bar(x + 0.18, [rates[k] for k in rates], width=0.36, color="C0",
        label=f"full-source V2 references ({n} edges)")
for i, k in enumerate(rates):
    ax1.annotate(f"{100*legacy_rates[k]:.1f}%", (i - 0.18, legacy_rates[k]),
                 textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    ax1.annotate(f"{100*rates[k]:.1f}%", (i + 0.18, rates[k]),
                 textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
ax1.set_xticks(x); ax1.set_xticklabels(list(rates))
ax1.set_ylabel("share of GT edges judged non-is-a", fontweight="bold")
ax1.set_title("Reference contamination replicates on the\nrepaired references (two-judge audit)",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=8.5, framealpha=0.95)
ax1.grid(axis="y", color="0.92")

ax2.bar(range(3), list(fp.values()), color=["C3", "C0", "0.5"], width=0.6)
for i, (k, v) in enumerate(fp.items()):
    ax2.annotate(f"{100*v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(range(3)); ax2.set_xticklabels(list(fp))
ax2.set_ylabel("share of metric FPs judged is-a", fontweight="bold")
ax2.set_title(f"Even against full-source references, most remaining\n"
              f"Gemini FPs are judged genuinely is-a (n={len(com)})",
              fontsize=10, fontweight="bold")
ax2.grid(axis="y", color="0.92")
fig3.suptitle("Two-judge audit of the repaired (V2) benchmarks", fontsize=11, fontweight="bold", y=1.02)
fig3.tight_layout()
fig3.savefig(f"{OUT}/audit_v2.png", dpi=250, bbox_inches="tight")
print("wrote audit_v2.png")
