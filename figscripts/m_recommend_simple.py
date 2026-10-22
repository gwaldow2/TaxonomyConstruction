"""Decluttered Gemma batching-prescription figure: rule + payoff only."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")

pts = []
for path, N in (("results/mech_20260909/density_gemma4.json", 100),
                ("results/scaling_20261001/results/density_sub200.json", 200)):
    agg = {}
    for b in json.load(open(f"{BASE}/{path}", encoding="utf-8")):
        for r in b["results"]:
            mm = re.search(r"density m=(\d+)\]", r["method"])
            if mm:
                agg.setdefault(int(mm.group(1)), []).append(r["Runtime_sec"])
    for m, v in agg.items():
        pts.append((m * (N - 1), np.mean(v) / -(-N // m)))
jj = np.array([p[0] for p in pts]); tt = np.array([p[1] for p in pts])
d, logc = np.polyfit(np.log(jj), np.log(tt), 1)
tcall = lambda j: np.exp(logc) * j ** d
BUDGET = 500

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# ---- Panel A: just the rule ----
Ns = np.logspace(np.log10(10), np.log10(3000), 300)
ax1.plot(Ns, Ns * (Ns - 1), "-", color="C1", lw=2.4, label=r"single call ($m{=}N$)")
ax1.plot(Ns, Ns - 1, "-", color="C0", lw=2.4, label=r"per-target ($m{=}1$)")
ax1.axhspan(400, 600, color="#F5C97B", alpha=0.65, zorder=0)
ax1.annotate(r"quality budget: $\sim$500 judgments per call (measured knee)",
             (10.5, 680), fontsize=9, color="0.2", va="bottom", fontweight="bold")
ax1.axhline(500, color="#C8922A", lw=1.2, zorder=1)
for i, (N, lab) in enumerate(((50, r"$m^*{=}10$"), (100, r"$m^*{=}5$"),
                              (200, r"$m^*{=}2$"), (400, r"$m^*{=}1$"))):
    ax1.annotate(lab, (N, 500), textcoords="offset points",
                 xytext=(0, -13 if i % 2 == 0 else -26),
                 ha="center", fontsize=8.5, fontweight="bold", color="0.1")
    ax1.plot([N, N], [N - 1, 500], color="0.45", lw=1.2, ls=":", zorder=1)
ax1.axvspan(501, 3000, color="0.93", zorder=0)
ax1.annotate("no safe $m$:\nbudget exceeded\neven at $m{=}1$", (1150, 60),
             fontsize=8.5, color="0.35", ha="center")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("vocabulary size $N$", fontweight="bold")
ax1.set_ylabel("judgments per call", fontweight="bold")
ax1.set_title("The rule: batch up from $m{=}1$ until the budget\n"
              "(dotted verticals: the headroom the budget allows)",
              fontsize=9.5, fontweight="bold")
ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
ax1.grid(color="0.93", which="both")

# ---- Panel B: the payoff ----
Ns2 = np.unique(np.logspace(np.log10(25), np.log10(500), 150).astype(int))
t_m1 = 0.4123 * Ns2 ** 1.29
t_rec = []
for N in Ns2:
    m = max(1, int(BUDGET // (N - 1)))
    t_rec.append(-(-N // m) * tcall(m * (N - 1)))
ax2.plot(Ns2, t_m1, "-", color="C0", lw=2.4, label=r"per-target $m{=}1$")
ax2.plot(Ns2, t_rec, "-", color="C2", lw=2.4, label=r"recommended $m^*$")
for N, m, t in ((100, 5, 37.7), (200, 2, 160.8), (100, 1, 115.9), (200, 1, 266.9)):
    ax2.plot(N, t, "o", color="C2" if m > 1 else "C0", ms=8, mec="k", mew=0.6)
ax2.annotate("$3.1\\times$ cheaper, $-$0.077 F1\n(measured, $N{=}100$)", (100, 37.7),
             textcoords="offset points", xytext=(12, -24), fontsize=9, color="C2")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("vocabulary size $N$", fontweight="bold")
ax2.set_ylabel("wall-clock per taxonomy (s, local)", fontweight="bold")
ax2.set_title("The payoff: 2--3$\\times$ less wall-clock\n(dots: measured checkpoints)",
              fontsize=9.5, fontweight="bold")
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
ax2.grid(color="0.93", which="both")
from matplotlib.ticker import NullFormatter, FixedLocator
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.xaxis.set_major_locator(FixedLocator([25, 50, 100, 200, 400]))
ax2.set_xticklabels(["25", "50", "100", "200", "400"])

fig.suptitle("Batching prescription (Gemma-4): choose the largest $m$ with $m(N{-}1) \\leq 500$,\n"
             "trading $\\Delta$F1 $\\approx$ $-$0.06 to $-$0.09 for 2.4--3.1$\\times$ less wall-clock",
             fontsize=11, fontweight="bold", y=1.06)
fig.tight_layout()
fig.savefig(f"{BASE}/paper_figures/m_recommendation_gemma_simple.png", dpi=250, bbox_inches="tight")
print("wrote simple figure")
