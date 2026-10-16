"""Density knee at N=100 vs N=200, on the judgments-per-call axis."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
OUT = f"{BASE}/paper_figures/density_knee_scale.png"


def curve(path, n_nodes):
    agg = {}
    for b in json.load(open(path, encoding="utf-8")):
        for r in b["results"]:
            m = re.search(r"density m=(\d+)\]", r["method"])
            if m:
                agg.setdefault(int(m.group(1)), []).append(r["Cond_Clos_F1"])
    ms = sorted(agg)
    return (np.array([m * (n_nodes - 1) for m in ms]),
            np.array([np.mean(agg[m]) for m in ms]), ms)


x100, y100, m100 = curve(f"{BASE}/results/mech_20260909/density_gemma4.json", 100)
x200, y200, m200 = curve(f"{BASE}/results/scaling_20261001/results/density_sub200.json", 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, (xs, ys, ms, col, lab) in [
        (ax1, (x100, y100, m100, "C0", "$N{=}100$ (99 to 9,900 judg/call)")),
        (ax1, (x200, y200, m200, "C3", "$N{=}200$ (199 to 39,800 judg/call)"))]:
    ax.plot(xs, ys, "o-", color=col, lw=2.3, ms=6, label=lab)
    up = col == "C0"
    for x, y, m in zip(xs, ys, ms):
        if not up and m == 1:
            continue
        ax.annotate(f"m={m}", (x, y), textcoords="offset points",
                    xytext=(-8 if (not up and m == 200) else 0, 7 if up else -13),
                    ha="center", fontsize=7, color=col)
ax1.set_ylabel("mean Cond. Closure F1", fontweight="bold")
ax1.set_title("Absolute F1 (gray band: $\sim$500 judgments per call)", fontsize=10, fontweight="bold")

ax2.plot(x100, y100 / y100[0], "o-", color="C0", lw=2.3, ms=6, label="$N{=}100$, relative to own $m{=}1$")
ax2.plot(x200, y200 / y200[0], "o-", color="C3", lw=2.3, ms=6, label="$N{=}200$, relative to own $m{=}1$")
ax2.set_ylabel("F1 relative to the sweep's $m{=}1$ anchor", fontweight="bold")
ax2.set_title("Shape comparison: the knee sits at the same\nabsolute judgment count, not the same $m$",
              fontsize=10, fontweight="bold")

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.axvspan(400, 600, color="0.85", alpha=0.6, zorder=0)
    ax.set_xlabel("judgments per call  ($m \\times (N{-}1)$)", fontweight="bold")
    ax.margins(y=0.12)
    ax.grid(color="0.92", which="both")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)

fig.suptitle("The overload knee is a fixed per-call budget: doubling $N$ halves the $m$ at which quality breaks,\n"
             "keeping the knee at $\\sim$500 judgments per call (Gemma-4, seven-dataset means)",
             fontsize=10.5, fontweight="bold", y=1.09)
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
for ms, xs, ys, tag in ((m100, x100, y100, "N=100"), (m200, x200, y200, "N=200")):
    print(tag, {m: round(y, 3) for m, y in zip(ms, ys)})
