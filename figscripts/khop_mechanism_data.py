"""Build khop_mechanism.csv: per (model, N, hop), classify every GT closure pair as
covered / chain-broken (both terms asserted somewhere, no path) / term-dropped
(an endpoint absent from every asserted edge). Input to khop_mechanism_vis.py.

Reads single-call prediction graphs from the battery (N=100) and scaling (N=200)
campaign archives; synonym-aware matching throughout.
"""
import csv
import glob
import os

import networkx as nx

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
OUT = f"{BASE}/results/scaling_20261001/khop_mechanism.csv"

CORE = ["WordNetFood", "CellOntology", "SemEvalFood", "LLMs4OL_OBI",
        "LLMs4OL_PO", "LLMs4OL_SchemaOrg", "LLMs4OL_MatOnto"]

CONFIGS = [
    ("gemini", 100, "{B}/results/battery_20260925/preds/GT_{ds}_SUB_eval.graphml",
                    "{B}/results/battery_20260925/preds/{ds}_SUB_SingleCall_gemini31b_pred.graphml"),
    ("gemini", 200, "{B}/results/scaling_20261001/benchmark_sets/{ds}_SUB200.graphml",
                    "{B}/results/scaling_20261001/results/{ds}_SUB200_SingleCall_gemini31_s200_pred.graphml"),
    ("gemma",  100, "{B}/results/battery_20260925/preds/GT_{ds}_SUB_eval.graphml",
                    "{B}/results/battery_20260925/preds/{ds}_SUB_SingleCall_gemma4b_pred.graphml"),
    ("gemma",  200, "{B}/results/scaling_20261001/benchmark_sets/{ds}_SUB200.graphml",
                    "{B}/results/scaling_20261001/results/{ds}_SUB200_SingleCall_gemma4_s200_pred.graphml"),
]


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
    if "virtual_root" in H:
        H.remove_node("virtual_root")
    return H


def main():
    agg = {}
    for model, N, gt_pat, pred_pat in CONFIGS:
        for ds in CORE:
            gts = glob.glob(gt_pat.format(B=BASE, ds=ds))
            prs = glob.glob(pred_pat.format(B=BASE, ds=ds))
            if not gts or not prs:
                print(f"[!] missing {model} N={N} {ds}")
                continue
            GT = strip_vr(nx.read_graphml(gts[0]))
            red = nx.transitive_reduction(GT)
            P = strip_vr(nx.read_graphml(prs[0]))
            clos = {(a, b) for a in P.nodes() for b in nx.descendants(P, a)}
            by_term = {}
            for n in P.nodes():
                if P.degree(n) > 0:
                    for t in parse_lemma(n):
                        by_term.setdefault(t, set()).add(n)

            def cands(u):
                hits = [by_term[t] for t in parse_lemma(u) if t in by_term]
                return set().union(*hits) if hits else set()

            for u in red.nodes():
                cu = cands(u)
                for v, k in nx.single_source_shortest_path_length(red, u).items():
                    if k == 0:
                        continue
                    cv = cands(v)
                    key = (model, N, k)
                    tot, cov, cb, td, bt = agg.get(key, (0, 0, 0, 0, 0))
                    tot += 1
                    if not cu or not cv:
                        td += 1
                    else:
                        bt += 1
                        if any((a, b) in clos for a in cu for b in cv):
                            cov += 1
                        else:
                            cb += 1
                    agg[key] = (tot, cov, cb, td, bt)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "N", "hop", "total", "covered", "chain_broken",
                    "term_dropped", "both_touched"])
        for (model, N, k), (tot, cov, cb, td, bt) in sorted(agg.items()):
            w.writerow([model, N, k, tot, cov, cb, td, bt])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
