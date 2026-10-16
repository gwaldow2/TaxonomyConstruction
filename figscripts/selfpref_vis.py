"""Two-panel judge self-preference figure from the 2x2 audit CSVs."""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction") + "/results/x2_20260923"
OUT = "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction/paper_figures/judge_selfpreference.png"


def load(p):
    return {(r["dataset"], r["parent"], r["child"]): r
            for r in csv.DictReader(open(p, encoding="utf-8")) if r["source"] == "pred"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


cells = {}
for pred in ["gemma4P", "gemini31P"]:
    gm = load(f"{BASE}/rt_gemmaJ_{pred}.csv")
    gi = load(f"{BASE}/rt_geminiJ_{pred}.csv")
    keys = [k for k in gm if k in gi and gm[k]["is_fp"] == "1"]
    a = np.array([gm[k]["type"] == "is_a" for k in keys])
    b = np.array([gi[k]["type"] == "is_a" for k in keys])
    cells[pred] = {"n": len(keys), "gemmaJ": a.mean(), "geminiJ": b.mean(),
                   "both": (a & b).mean(),
                   "onlyGemma": int((a & ~b).sum()), "onlyGemini": int((b & ~a).sum()),
                   "kGm": int(a.sum()), "kGi": int(b.sum()), "kBoth": int((a & b).sum())}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                               gridspec_kw={"width_ratios": [1.5, 1]})

# ---- Panel A: the 2x2 with both-judge bars ---------------------------------
labels = ["Gemma-4 predictions\n(849 FPs)", "Gemini predictions\n(780 FPs)"]
x = np.arange(2)
w = 0.26
COL = {"gemmaJ": "C0", "geminiJ": "C1", "both": "0.25"}
for i, (key, name) in enumerate([("gemmaJ", "Gemma judge"),
                                 ("geminiJ", "Gemini judge"),
                                 ("both", "both judges")]):
    vals = [cells["gemma4P"][key], cells["gemini31P"][key]]
    ks = [cells[p]["kGm" if key == "gemmaJ" else "kGi" if key == "geminiJ" else "kBoth"]
          for p in ["gemma4P", "gemini31P"]]
    ns = [cells[p]["n"] for p in ["gemma4P", "gemini31P"]]
    err = np.array([[v - wilson(k, n)[0], wilson(k, n)[1] - v]
                    for v, k, n in zip(vals, ks, ns)]).T
    bars = ax1.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3,
                   color=COL[key], label=name)
    for j, bar in enumerate(bars):
        ax1.annotate(f"{vals[j]:.1%}", (bar.get_x() + bar.get_width() / 2,
                                        bar.get_height()),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5)
# mark the self cells
for xi, key in [(0 - w, "gemmaJ"), (1, "geminiJ")]:
    pred = "gemma4P" if key == "gemmaJ" else "gemini31P"
    off = {"gemmaJ": -w, "geminiJ": 0}[key]
    xc = (0 if pred == "gemma4P" else 1) + off
    ax1.annotate("self", (xc, 0.035), ha="center", fontsize=9,
                 fontweight="bold", color="white")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9.5)
ax1.set_ylabel("share of metric-FPs judged is-a", fontweight="bold")
ax1.set_ylim(0, 1.02)
ax1.legend(fontsize=8.5, loc="upper left")
ax1.set_title("Each judge favors its own model's edges\n(identical edge sets per group; Wilson 95% CI)",
              fontsize=10, fontweight="bold")

# ---- Panel B: discordant verdicts (the McNemar evidence) -------------------
y = np.arange(2)
og = [cells["gemma4P"]["onlyGemma"], cells["gemini31P"]["onlyGemma"]]
oi = [cells["gemma4P"]["onlyGemini"], cells["gemini31P"]["onlyGemini"]]
h = 0.32
ax2.barh(y + h / 2, og, h, color="C0", label="only Gemma judge says is-a")
ax2.barh(y - h / 2, oi, h, color="C1", label="only Gemini judge says is-a")
for yi, (a, b) in enumerate(zip(og, oi)):
    ax2.annotate(str(a), (a, yi + h / 2), textcoords="offset points",
                 xytext=(4, -3), fontsize=9)
    ax2.annotate(str(b), (b, yi - h / 2), textcoords="offset points",
                 xytext=(4, -3), fontsize=9)
ax2.set_yticks(y)
ax2.set_yticklabels(["Gemma-4\npredictions", "Gemini\npredictions"], fontsize=9.5)
ax2.set_xlabel("disagreeing verdicts (count)", fontweight="bold")
ax2.set_title("Disagreements are directional\n(McNemar $p<10^{-4}$ in both rows)",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8.5, loc="lower right")
ax2.set_xlim(0, 145)

fig.suptitle("Judge self-preference in the predicted-edge audit ($2\\times2$: both judges over both models' full edge sets)",
             fontsize=11, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
for p, c in cells.items():
    print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
