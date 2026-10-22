"""Three-panel judgment-density figure: knee, failure mode, compute collapse."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
SRC = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction") + "/results/mech_20260909/density_gemma4.json"
OUT = "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction/paper_figures/density_gemma4.png"
ONE_CALL = {"single-call (best practice)": 0.605, "single-call (matched)": 0.598}

per_ds = defaultdict(dict)          # ds -> m -> row
for b in json.load(open(SRC, encoding="utf-8")):
    for r in b["results"]:
        if "density" in r["method"]:
            m = int(r["method"].split("m=")[1].rstrip("])"))
            per_ds[b["dataset"]][m] = r

ms = sorted({m for d in per_ds.values() for m in d})
def mean(key, m):
    vals = [per_ds[ds][m][key] for ds in per_ds if m in per_ds[ds]]
    return sum(vals) / len(vals)
jpc = {m: mean("judgments_per_call", m) for m in ms}
x = [jpc[m] for m in ms]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))

# --- Panel A: the knee -------------------------------------------------------
for ds in sorted(per_ds):
    xs = [jpc[m] for m in ms if m in per_ds[ds]]
    ys = [per_ds[ds][m]["Cond_Clos_F1"] for m in ms if m in per_ds[ds]]
    ax1.plot(xs, ys, color="0.75", lw=0.8, zorder=1)
f1 = [mean("Cond_Clos_F1", m) for m in ms]
ax1.plot(x, f1, "o-", color="C0", lw=2.5, ms=6, zorder=3, label="mean (7 datasets)")
lo = min(ONE_CALL.values()); hi = max(ONE_CALL.values())
ax1.axhspan(lo, hi, color="C3", alpha=0.15, zorder=0)
ax1.axhline((lo + hi) / 2, color="C3", ls="--", lw=1.2,
            label="one-call baselines (0.598-0.605)")
for m, xi, yi in zip(ms, x, f1):
    ax1.annotate(f"m={m}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=7.5)
ax1.set_xscale("log")
ax1.set_xlabel("judgments per call", fontweight="bold")
ax1.set_ylabel("Cond. Closure F1", fontweight="bold")
ax1.set_title("Accuracy falls to the one-call level", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8, loc="lower left")

# --- Panel B: failure mode is omission --------------------------------------
p = [mean("Cond_Clos_Precision", m) for m in ms]
r = [mean("Cond_Clos_Recall", m) for m in ms]
ax2.plot(x, r, "o-", color="C1", lw=2.2, ms=5, label="recall")
ax2.plot(x, p, "s-", color="C2", lw=2.2, ms=5, label="precision")
ax2.set_xscale("log")
ax2.set_xlabel("judgments per call", fontweight="bold")
ax2.set_ylabel("Cond. Closure P / R", fontweight="bold")
ax2.set_title("Failure mode: omission, not fabrication", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)

# --- Panel C: compute per judgment collapses ---------------------------------
spc = []
for m in ms:
    spc.append(sum(per_ds[ds][m]["Runtime_sec"] / per_ds[ds][m]["n_calls"]
                   for ds in per_ds if m in per_ds[ds]) / len(per_ds))
mspj = [1000 * s / jpc[m] for s, m in zip(spc, ms)]
ax3.plot(x, mspj, "o-", color="C4", lw=2.2, ms=5, label="ms per judgment")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("judgments per call", fontweight="bold")
ax3.set_ylabel("compute per judgment (ms)", fontweight="bold")
ax3b = ax3.twinx()
ax3b.plot(x, spc, "s--", color="0.5", lw=1.5, ms=4, label="seconds per call")
ax3b.set_ylabel("seconds per call", color="0.4")
ax3b.tick_params(axis="y", colors="0.4")
ax3.annotate(f"{mspj[0]:.1f} ms", (x[0], mspj[0]), textcoords="offset points",
             xytext=(8, 4), fontsize=8, color="C4")
ax3.annotate(f"{mspj[-1]:.1f} ms", (x[-1], mspj[-1]), textcoords="offset points",
             xytext=(-8, -12), ha="right", fontsize=8, color="C4")
h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
ax3.set_title("Reasoning does not scale with load", fontsize=10, fontweight="bold")

fig.suptitle("Judgment density (targets per call $m$, full entity list every call; "
             "Gemma-4, seven-dataset means)", fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print("wrote", OUT)
