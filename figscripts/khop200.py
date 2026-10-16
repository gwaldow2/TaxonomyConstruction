"""K-hop recall at SUB200 (set-overlap matching), structured vs single-call, both models."""
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import os
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
CAMP = f"{BASE}/results/scaling_20261001"
OUT_CSV = f"{CAMP}/khop_sub200.csv"
OUT_PNG = f"{BASE}/paper_figures/khop_sub200.png"

CORE = ["WordNetFood", "CellOntology", "SemEvalFood", "LLMs4OL_OBI",
        "LLMs4OL_PO", "LLMs4OL_SchemaOrg", "LLMs4OL_MatOnto"]


def parse_lemma(s):
    s = str(s).strip().lower()
    if "(" in s and s.endswith(")"):
        head, inner = s.split("(", 1)
        syns = [t.strip() for t in inner[:-1].split(",") if t.strip()]
        if syns and syns[0] == head.strip():
            return set(syns)
    return {s}


def strip_vr(G):
    H = G.copy()
    for n in list(H.nodes()):
        if str(n) == "virtual_root":
            H.remove_node(n)
    return H


def load_gt(ds):
    G = strip_vr(nx.read_graphml(f"{CAMP}/benchmark_sets/{ds}_SUB200.graphml"))
    red = nx.transitive_reduction(G)
    pairs = {}  # (u,v) -> hop distance in reduction
    for u in red.nodes():
        for v, d in nx.single_source_shortest_path_length(red, u).items():
            if d > 0:
                pairs[(u, v)] = d
    return pairs


def covered_fn(pred_path):
    P = strip_vr(nx.read_graphml(pred_path))
    clos = set()
    for a in P.nodes():
        for b in nx.descendants(P, a):
            clos.add((a, b))
    by_term = {}
    for n in P.nodes():
        for t in parse_lemma(n):
            by_term.setdefault(t, set()).add(n)

    def covered(u, v):
        cu = set().union(*(by_term.get(t, set()) for t in parse_lemma(u)))
        cv = set().union(*(by_term.get(t, set()) for t in parse_lemma(v)))
        return any((a, b) in clos for a in cu for b in cv)
    return covered


RUNS = [("gemma4b", "structured", "Our_Method_k1000_clawback0_gemma4_s200"),
        ("gemma4b", "single-call", "SingleCall_gemma4_s200"),
        ("gemini31b", "structured", "Our_Method_k1000_clawback0_gemini31_s200"),
        ("gemini31b", "single-call", "SingleCall_gemini31_s200")]

rows = []
for ds in CORE:
    gt_pairs = load_gt(ds)
    for tag, meth, suffix in RUNS:
        paths = glob.glob(f"{CAMP}/results/{ds}_SUB200_{suffix}_pred.graphml")
        if not paths:
            print(f"[!] missing pred: {ds} {tag} {meth}")
            continue
        cov = covered_fn(paths[0])
        for (u, v), k in gt_pairs.items():
            rows.append((tag, meth, ds, k, 1 if cov(u, v) else 0))

agg = {}
for tag, meth, ds, k, c in rows:
    key = (tag, meth, k)
    tot, hit = agg.get(key, (0, 0))
    agg[key] = (tot + 1, hit + c)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["tag", "method", "hop", "total", "covered", "recall"])
    for (tag, meth, k), (tot, hit) in sorted(agg.items()):
        w.writerow([tag, meth, k, tot, hit, round(hit / tot, 4)])
print("wrote", OUT_CSV)

ns = {}
for (tag, meth, k), (tot, hit) in agg.items():
    ns.setdefault(k, 0)
    if (tag, meth) == ("gemma4b", "structured"):
        ns[k] = tot
n_note = ", ".join(str(ns[k]) for k in sorted(ns) if ns[k] >= 10)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
PANELS = [("gemma4b", "Gemma-4-31B"), ("gemini31b", "Gemini 3.1 Pro")]
STYLE = [("structured", "C0", "-", "o", "structured (ours)"),
         ("single-call", "C1", "--", "s", "single-call")]
for ax, (tag, title) in zip(axes, PANELS):
    for meth, col, ls, mk, label in STYLE:
        pts = sorted((k, hit / tot) for (t, m, k), (tot, hit) in agg.items()
                     if t == tag and m == meth and tot >= 10)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                color=col, lw=2.4, ms=6, label=label)
    ax.set_xlabel("hop distance $k$ in ground-truth reduction", fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(color="0.92")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
axes[0].set_ylabel("recall of $k$-hop closure pairs", fontweight="bold")
fig.suptitle("Recall by ancestor distance at $N{=}200$: deeper graphs widen the regime split\n"
             f"(seven-dataset core, synonym-aware matching; $n$ per hop: {n_note})",
             fontsize=10.5, fontweight="bold", y=1.08)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
print("wrote", OUT_PNG)

for tag, _ in PANELS:
    for meth, *_ in STYLE:
        pts = sorted((k, round(hit / tot, 3), tot) for (t, m, k), (tot, hit) in agg.items()
                     if t == tag and m == meth)
        print(tag, meth, pts)
