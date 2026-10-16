"""Two-mechanism decomposition of single-call k-hop failures: chain-broken vs term-dropped,
plus conditional recall showing the mechanisms separate cleanly."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
OUT = f"{BASE}/paper_figures/khop_mechanism.png"

rows = {}
for r in csv.DictReader(open(f"{BASE}/results/scaling_20261001/khop_mechanism.csv", encoding="utf-8")):
    rows[(r["model"], int(r["N"]), int(r["hop"]))] = (
        int(r["total"]), int(r["covered"]), int(r["chain_broken"]),
        int(r["term_dropped"]), int(r["both_touched"]))

MODELS = [("gemma", "Gemma-4-31B"), ("gemini", "Gemini 3.1 Pro")]
C_COV, C_CB, C_TD = "#3B7EC0", "#F39C3D", "#9E9E9E"

fig = plt.figure(figsize=(12.5, 7.8))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.15], hspace=0.55, wspace=0.3,
                       top=0.80, bottom=0.08)

# ---- top row: stacked failure anatomy, one panel per model x N ----
panels = [("gemma", 100), ("gemma", 200), ("gemini", 100), ("gemini", 200)]
for i, (model, N) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    ks = sorted(k for (m, n, k) in rows if m == model and n == N
                and rows[(m, n, k)][0] >= 10)
    cov = [rows[(model, N, k)][1] / rows[(model, N, k)][0] for k in ks]
    cb = [rows[(model, N, k)][2] / rows[(model, N, k)][0] for k in ks]
    td = [rows[(model, N, k)][3] / rows[(model, N, k)][0] for k in ks]
    ax.bar(ks, cov, color=C_COV, width=0.8)
    ax.bar(ks, cb, bottom=cov, color=C_CB, width=0.8)
    ax.bar(ks, td, bottom=[a + b for a, b in zip(cov, cb)], color=C_TD, width=0.8)
    name = dict(MODELS)[model]
    ax.set_title(f"{name}\n$N={N}$", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xticks(ks)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.set_ylabel("share of $k$-hop pairs", fontweight="bold", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("hop $k$", fontsize=8.5)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_COV, C_CB, C_TD)],
           labels=["covered", "chain-broken (terms mentioned, no path)",
                   "term-dropped (endpoint in no asserted edge)"],
           loc="upper center", bbox_to_anchor=(0.5, 0.875), ncol=3, fontsize=9,
           framealpha=0.95)

# ---- bottom row: unconditional vs conditional recall ----
for j, (model, name) in enumerate(MODELS):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    for N, col in ((100, "0.45"), (200, "C3")):
        ks = sorted(k for (m, n, k) in rows if m == model and n == N
                    and rows[(m, n, k)][0] >= 10)
        unc = [rows[(model, N, k)][1] / rows[(model, N, k)][0] for k in ks]
        con = [rows[(model, N, k)][1] / rows[(model, N, k)][4] for k in ks]
        ax.plot(ks, unc, "o-", color=col, lw=2.0, ms=5,
                label=f"$N={N}$ recall (all pairs)")
        ax.plot(ks, con, "^--", color=col, lw=2.0, ms=5, alpha=0.85,
                label=f"$N={N}$ recall | terms mentioned")
    ax.set_title(name, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("hop distance $k$ in ground-truth reduction", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(color="0.92")
    if model == "gemma":
        ax.legend(fontsize=7.8, loc="lower left", framealpha=0.95)
    if j == 0:
        ax.set_ylabel("recall of $k$-hop closure pairs", fontweight="bold", fontsize=9)

fig.suptitle("Two failure modes of single-call extraction, separated: conditioning on term mention\n"
             "restores Gemini's $N{=}200$ curve to its $N{=}100$ shape (collapse = term omission), while\n"
             "Gemma's conditional recall still falls with $N$ (degradation = link thinning)",
             fontsize=10.5, fontweight="bold", y=0.985)
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
