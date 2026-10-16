"""Scaling figures: F1 vs N (both models) and runtime vs N with power-law fits."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
CAMP = f"{BASE}/results/scaling_20261001/results"
OUT_F1 = f"{BASE}/paper_figures/scaling_f1.png"
OUT_COST = f"{BASE}/paper_figures/scaling_cost.png"

ST = r"Our Method \(k=1000"
SC = r"Single-Call Baseline$"


def rows(path, scale, methpat, field):
    out = []
    for b in json.load(open(path, encoding="utf-8")):
        if not b["dataset"].endswith("_" + scale):
            continue
        for r in b["results"]:
            if re.search(methpat, r["method"]):
                out.append(r[field])
    return out


def legacy_n100(path, methpat, field):
    """N=100 point from the original campaign files (datasets end in _SUB)."""
    out = []
    for b in json.load(open(path, encoding="utf-8")):
        if not b["dataset"].endswith("_SUB"):
            continue
        core = ("WordNetFood", "CellOntology", "SemEvalFood", "LLMs4OL_OBI",
                "LLMs4OL_PO", "LLMs4OL_SchemaOrg", "LLMs4OL_MatOnto")
        if b["dataset"].rsplit("_SUB", 1)[0] not in core:
            continue
        for r in b["results"]:
            if re.search(methpat, r["method"]):
                out.append(r[field])
    return out


MODELS = [
    ("Gemma-4-31B", f"{CAMP}/scaling_gemma4.json",
     f"{BASE}/results/results_json/control_gemma4N.json",
     f"{BASE}/results/results_json/sc_gemma4_prim_merge.json"),
    ("Gemini 3.1 Pro", f"{CAMP}/scaling_gemini.json",
     f"{BASE}/results/results_json/control_gemini.json",
     f"{BASE}/results/results_json/sc_gemini_prim_merge.json"),
]
NS = [25, 50, 100, 200]

# ---------------- F1 figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
for ax, (name, scal, ctrl100, sc100) in zip(axes, MODELS):
    for methpat, pat100, col, ls, mk, label in [
            (ST, r"Our Method", "C0", "-", "o", "structured (ours)"),
            (SC, r"Single-Call", "C1", "--", "s", "single-call")]:
        means, spreads = [], []
        for n in NS:
            if n == 100:
                src = ctrl100 if methpat == ST else sc100
                v = legacy_n100(src, pat100, "Cond_Clos_F1")
                means.append(np.mean(v)); spreads.append([np.mean(v)])
            else:
                v = rows(scal, f"SUB{n}", methpat, "Cond_Clos_F1")
                reps = []
                for sd in (1, 2):   # Gemma-only seeded reruns
                    sv = rows(scal, f"SUB{n}S{sd}", methpat, "Cond_Clos_F1")
                    if sv:
                        reps.append(np.mean(sv))
                means.append(np.mean(v)); spreads.append(reps)
        means = np.array(means)
        ax.plot(NS, means, ls, marker=mk, color=col, lw=2.4, ms=7, label=label)
        for n, sv in zip(NS, spreads):
            if sv:
                ax.plot([n * 1.06] * len(sv), sv, "D", color=col, ms=3.5,
                        alpha=0.6, zorder=1, mew=0)
        for n, m in zip(NS, means):
            up = methpat == ST
            ax.annotate(f"{m:.3f}", (n, m), textcoords="offset points",
                        xytext=(-2 if n == 25 else 0, 8 if up else -14),
                        ha="left" if n == 25 else "center", fontsize=7.5, color=col)
    ax.set_xscale("log")
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("benchmark size $N$ (nodes)", fontweight="bold")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_ylim(0.26, 0.90)
    ax.grid(color="0.92")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
axes[0].set_ylabel("mean Cond. Closure F1 (7-dataset core)", fontweight="bold")
fig.suptitle("Taxonomy size: the structured--single-call gap opens with scale, as the knee predicts ---\n"
             "near zero at $N{=}25$ ($\\sim$600 judgments in one call), catastrophic at $N{=}200$ ($\\sim$40k)\n"
             "(seven-dataset core; small dots: Gemma seed replicates at $N{=}25,50$; $N{=}100$ = original campaigns)",
             fontsize=10.5, fontweight="bold", y=1.09)
fig.tight_layout()
fig.savefig(OUT_F1, dpi=250, bbox_inches="tight")
print("wrote", OUT_F1)

# ---------------- cost figure ----------------
fig2, ax = plt.subplots(figsize=(7.2, 4.8))
STYLE = [("Gemma-4-31B", ST, "C0", "-", "o", "Gemma structured"),
         ("Gemma-4-31B", SC, "C0", "--", "s", "Gemma single-call"),
         ("Gemini 3.1 Pro", ST, "C3", "-", "o", "Gemini structured"),
         ("Gemini 3.1 Pro", SC, "C3", "--", "s", "Gemini single-call")]
fits = {}
for name, methpat, col, ls, mk, label in STYLE:
    scal, ctrl100, sc100 = next((m[1], m[2], m[3]) for m in MODELS if m[0] == name)
    pat100 = r"Our Method" if methpat == ST else r"Single-Call"
    ys = []
    for n in NS:
        if n == 100:
            src = ctrl100 if methpat == ST else sc100
            ys.append(np.mean(legacy_n100(src, pat100, "Runtime_sec")))
        else:
            ys.append(np.mean(rows(scal, f"SUB{n}", methpat, "Runtime_sec")))
    ys = np.array(ys)
    b, loga = np.polyfit(np.log(NS), np.log(ys), 1)
    fits[label] = (np.exp(loga), b, ys)
    ax.plot(NS, ys, ls, marker=mk, color=col, lw=2.2, ms=6,
            label=f"{label}  ($T \\propto N^{{{b:.2f}}}$)")
    xx = np.array([200, 2350])
    ax.plot(xx, np.exp(loga) * xx ** b, ls, color=col, lw=1.0, alpha=0.4)
ax.axvline(2350, color="0.6", lw=0.8, ls=":")
ax.annotate("full WordNetFood\n(N=2,350)", (2350, ax.get_ylim()[0] * 1.5),
            fontsize=8, ha="right", color="0.35")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("benchmark size $N$ (nodes)", fontweight="bold")
ax.set_ylabel("mean runtime per dataset (s)", fontweight="bold")
ax.set_title("Measured runtime vs. taxonomy size, with power-law fits\n"
             "(faint lines: extrapolation to full-ontology scale)",
             fontsize=10, fontweight="bold")
ax.grid(color="0.92", which="both")
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
fig2.tight_layout()
fig2.savefig(OUT_COST, dpi=250, bbox_inches="tight")
print("wrote", OUT_COST)

print("\n=== power-law fits and full-graph extrapolations ===")
for label, (a, b, ys) in fits.items():
    t2350 = a * 2350 ** b
    t7255 = a * 7255 ** b
    print(f"{label:22s} T = {a:8.4f} * N^{b:.2f} | measured@200={ys[-1]:8.0f}s | "
          f"N=2350: {t2350/3600:7.1f} h | N=7255: {t7255/3600:8.1f} h")
