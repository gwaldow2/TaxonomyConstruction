"""Per-model prescriptive batching figures: Gemma (measured budget + payoff) and
Gemini (bracketed budget from scaling points)."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")

# severity colors for dF1 vs structured anchor
def sev(d):
    return "#2E9E4F" if d <= 0.03 else ("#F39C3D" if d <= 0.09 else "#D23B3B")

# ---- measured sweep data (Gemma): (judgments/call, dF1, s/call) at each N ----
sweeps = {}
for path, N in (("results/mech_20260909/density_gemma4.json", 100),
                ("results/scaling_20261001/results/density_sub200.json", 200)):
    f1, rt = {}, {}
    for b in json.load(open(f"{BASE}/{path}", encoding="utf-8")):
        for r in b["results"]:
            mm = re.search(r"density m=(\d+)\]", r["method"])
            if mm:
                f1.setdefault(int(mm.group(1)), []).append(r["Cond_Clos_F1"])
                rt.setdefault(int(mm.group(1)), []).append(r["Runtime_sec"])
    anchor = np.mean(f1[1])
    sweeps[N] = [(m * (N - 1), anchor - np.mean(f1[m]),
                  np.mean(rt[m]) / -(-N // m)) for m in sorted(f1)]

pts = [(j, t) for N in sweeps for j, d, t in sweeps[N]]
jj = np.array([p[0] for p in pts]); tt = np.array([p[1] for p in pts])
dexp, logc = np.polyfit(np.log(jj), np.log(tt), 1)
tcall = lambda j: np.exp(logc) * j ** dexp

BUDGET = 500

def recommend(N):
    if N * (N - 1) <= BUDGET:
        return N, 1, N * (N - 1), "one-call"
    if (N - 1) <= BUDGET:
        m = max(1, int(BUDGET // (N - 1)))
        return m, -(-N // m), m * (N - 1), "batch"
    return 1, N * -(-(N - 1) // BUDGET), BUDGET, "chunk"

def loadmap(ax, legend_loc="lower right"):
    Ns = np.logspace(np.log10(10), np.log10(3000), 300)
    ax.plot(Ns, Ns * (Ns - 1), "-", color="C1", lw=2.2, label="single call: $N(N{-}1)$ judgments")
    ax.plot(Ns, Ns - 1, "-", color="C0", lw=2.2, label="per-target ($m{=}1$): $N{-}1$ judgments")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("vocabulary size $N$", fontweight="bold")
    ax.set_ylabel("judgments per call", fontweight="bold")
    ax.grid(color="0.93", which="both")
    ax.legend(fontsize=8, loc=legend_loc, framealpha=0.95)

# ========================= GEMMA FIGURE =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
loadmap(ax1)
ax1.axhspan(400, 600, color="0.82", zorder=0)
ax1.axhline(200, color="0.45", ls="--", lw=1.2)
ax1.annotate("measured knee $\\approx$500 judg/call", (10.5, 700), fontsize=8.5,
             color="0.2", va="bottom")
ax1.annotate("conservative 200 ($\\Delta$F1 $\\leq$ 0.03)", (10.5, 132), fontsize=8, color="0.35")
# density-sweep evidence: dots on verticals at N=100 and N=200, colored by dF1
for N in (100, 200):
    for j, d, _ in sweeps[N]:
        ax1.plot(N, j, "o", color=sev(d), ms=5.5, mec="k", mew=0.4, zorder=5)
ax1.annotate("density-sweep points\n(green $\\Delta$F1$\\leq$.03 / orange $\\leq$.09 / red $>$.09):\n"
             "color flips crossing the band", (10.5, 130000), fontsize=8, color="0.25", va="bottom")
# single-call runs with Gemma's own gaps
for N, dq in ((25, 0.051), (50, 0.054), (100, 0.062), (200, 0.066)):
    ax1.plot(N, N * (N - 1), "s", color=sev(dq), ms=8, mec="k", mew=0.6, zorder=5)
for N, lab in ((50, "$m^*{=}10$"), (100, "$m^*{=}5$"), (200, "$m^*{=}2$"), (420, "$m^*{=}1$")):
    ax1.annotate(lab, (N, 390), textcoords="offset points", xytext=(0, -12),
                 ha="center", fontsize=8.5, fontweight="bold")
ax1.axvspan(501, 3000, color="0.93", zorder=0)
ax1.annotate("$m{=}1$ over budget:\nchunking required\n(quality untested)", (900, 30000),
             fontsize=8, color="0.35", ha="center")
ax1.set_title("Load map with measured evidence\n(squares: Gemma's single-call runs, colored by F1 gap)",
              fontsize=9.5, fontweight="bold")

Ns2 = np.unique(np.logspace(np.log10(25), np.log10(2350), 200).astype(int))
t_m1 = 0.4123 * Ns2 ** 1.29
rec = [recommend(N) for N in Ns2]
t_rec = np.array([c * tcall(j) for _, c, j, _ in rec])
chunk = np.array([r[3] == "chunk" for r in rec])
ax2.plot(Ns2, t_m1, "-", color="C0", lw=2.2, label="per-target $m{=}1$ (measured fit, $N^{1.29}$)")
ax2.plot(Ns2[~chunk], t_rec[~chunk], "-", color="C2", lw=2.2, label="recommended $m^*$ (budget 500)")
ax2.plot(Ns2[chunk], t_rec[chunk], ":", color="C2", lw=2.2, label="chunked regime (untested; costs MORE time)")
for N, m, t in ((100, 5, 37.7), (200, 2, 160.8), (100, 1, 115.9), (200, 1, 266.9)):
    ax2.plot(N, t, "o", color="C2" if m > 1 else "C0", ms=7, mec="k", mew=0.6)
ax2.annotate("measured: $3.1\\times$ cheaper at $-$0.077 F1\n($N{=}100$, $m{=}5$)", (100, 37.7),
             textcoords="offset points", xytext=(12, -24), fontsize=8, color="C2")
ax2.axvline(2350, color="0.6", lw=0.8, ls=":")
ax2.annotate("full\nWordNetFood", (2200, 15), fontsize=8, ha="right", color="0.35")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("vocabulary size $N$", fontweight="bold")
ax2.set_ylabel("wall-clock per taxonomy (s, local)", fontweight="bold")
ax2.set_title("The payoff at the recommended $m^*$\n(dots: measured sweep checkpoints)",
              fontsize=9.5, fontweight="bold")
ax2.legend(fontsize=8, loc="upper left", framealpha=0.95)
ax2.grid(color="0.93", which="both")
fig.suptitle("Gemma-4-31B batching prescription: choose the largest $m$ with $m(N{-}1) \\leq 500$ judgments per call\n"
             "($\\Delta$F1 $\\approx -0.06$ to $-0.09$ for 2.4--3.1$\\times$ less wall-clock; "
             "the 200-judgment line costs $\\leq$0.03 F1)",
             fontsize=10.5, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(f"{BASE}/paper_figures/m_recommendation_gemma.png", dpi=250, bbox_inches="tight")
print("wrote gemma figure")

# ========================= GEMINI FIGURE =========================
fig2, ax = plt.subplots(figsize=(7.6, 5.4))
loadmap(ax, legend_loc="lower right")
# bracket band: safe through N=50 single call (2,450 j), broken by N=100 (9,900 j)
ax.axhspan(2450, 9900, color="C3", alpha=0.12, zorder=0)
ax.axhline(2450, color="C3", lw=1.0, ls="--")
ax.axhline(9900, color="C3", lw=1.0, ls="--")
ax.annotate("knee bracket: 2,450--9,900 judg/call\n(no density sweep run for Gemini;\n"
            "bracketed by the single-call points)", (10.5, 11500), fontsize=8.5,
            color="C3", va="bottom")
for N, dq in ((25, 0.013), (50, 0.027), (100, 0.155), (200, 0.365)):
    ax.plot(N, N * (N - 1), "s", color=sev(dq), ms=9, mec="k", mew=0.7, zorder=5)
    ax.annotate(f"$-${dq:.2f}", (N, N * (N - 1)), textcoords="offset points",
                xytext=(9, -4), fontsize=8.5, color="0.15")
for N in (25, 50, 100, 200):
    ax.plot(N, N - 1, "o", color="#2E9E4F", ms=6, mec="k", mew=0.5, zorder=5)
ax.annotate("per-target runs: all fine\n(199 judg/call $\\ll$ bracket)", (300, 90),
            fontsize=8, color="#2E9E4F")
ax.annotate("batch below the bracket's LOWER edge:\npast budget, Gemini drops entire terms\n"
            "(entity triage) --- the unrecoverable mode", (400, 8800), fontsize=8.5,
            color="0.25", va="top")
ax.set_title("Single-call runs colored by F1 gap vs structured\n"
             "(green $\\leq$0.03, orange $\\leq$0.09, red $>$0.09)",
             fontsize=9.5, fontweight="bold")
fig2.suptitle("Gemini 3.1 Pro: the per-call budget is only bracketed --- batch conservatively\n"
              "(a small $m$-sweep would pin the knee; until then use the bracket's lower edge)",
              fontsize=10.5, fontweight="bold", y=1.03)
fig2.tight_layout()
fig2.savefig(f"{BASE}/paper_figures/m_recommendation_gemini.png", dpi=250, bbox_inches="tight")
print("wrote gemini figure")
