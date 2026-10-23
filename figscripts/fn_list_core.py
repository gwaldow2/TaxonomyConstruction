"""Export every false-negative pair of the structured method on the seven-dataset core.

A false negative is a reference ancestor pair (parent above child, at any hop depth)
that the prediction's transitive closure fails to cover under synonym set-overlap --
the same coverage test the k-hop recall machinery uses, so the per-hop totals here
reproduce corrected_core/khop_core.csv exactly.

Each row carries the failure anatomy:
  failure_mode   term_dropped  = an endpoint never appears in any asserted edge
                 chain_broken  = both endpoints asserted somewhere, but no path
  pred_parents_of_child        = where the model DID file the child (its direct
                                 parents in the predicted graph)

Sources are the same runs as fp_list_core.py (see that script's provenance note:
the legacy-three Gemini rows come from the battery rerun, not the table's run).

Output: results/corrected_core/fn_structured_core.csv
"""
import csv
import os
import sys

import networkx as nx

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_manager import parse_lemma_format  # noqa: E402

CAMP = os.environ.get("TAXO_V2_CAMP", f"{BASE}/results/v2r2_20261021")
BATT = f"{BASE}/results/battery_20260925/preds"
OUT = f"{BASE}/results/corrected_core/fn_structured_core.csv"

LEG3 = ["WordNetFood", "CellOntology", "SemEvalFood"]
REBUILT = {"SchemaOrgV2": "SchemaOrg", "POV2": "PO", "OBIV2": "OBI", "MatOntoV2": "MatOnto"}
MODELS = {"gemma4": "Gemma-4-31B", "gemini31": "Gemini 3.1 Pro"}


def strip_vr(G):
    G = G.copy()
    for n in list(G.nodes()):
        if str(n) == "virtual_root":
            G.remove_node(n)
    return G


def pred_graph(model, ds):
    if ds in LEG3:
        if model == "gemma4":
            path = f"{BASE}/results/x2_20260923/{ds}_SUB_gemma4_diag.csv"
            rows = csv.DictReader(open(path, encoding="utf-8"))
            return (nx.DiGraph((r["parent"], r["child"]) for r in rows),
                    "control_gemma4N (x2_20260923 diag)")
        P = nx.read_graphml(f"{BATT}/{ds}_SUB_Our_Method_k1000_clawback0_gemini31b_pred.graphml")
        return strip_vr(P), "battery_20260925 rerun (NOT the table's control_gemini run)"
    P = nx.read_graphml(f"{CAMP}/results/{ds}_SUB_Our_Method_k1000_clawback0_{model}_v2_pred.graphml")
    return strip_vr(P), f"{os.path.basename(CAMP)} {model}_v2"


def gt_graph(ds):
    path = (f"{BATT}/GT_{ds}_SUB_eval.graphml" if ds in LEG3
            else f"{CAMP}/results/GT_{ds}_SUB_eval.graphml")
    return strip_vr(nx.read_graphml(path))


def main():
    out = []
    print(f"{'model':16s} {'dataset':13s} {'GT pairs':>9s} {'missed':>7s} {'term_dropped':>13s} {'chain_broken':>13s}")
    for model, model_name in MODELS.items():
        for ds in LEG3 + list(REBUILT):
            G = gt_graph(ds)
            red = nx.transitive_reduction(G)
            P, src = pred_graph(model, ds)
            clos = {(a, b) for a in P.nodes() for b in nx.descendants(P, a)}
            by_term = {}
            for n in P.nodes():
                if P.degree(n) > 0:
                    for t in parse_lemma_format(n):
                        by_term.setdefault(t, set()).add(n)

            def cands(u):
                hits = [by_term[t] for t in parse_lemma_format(u) if t in by_term]
                return set().union(*hits) if hits else set()

            def pred_parents(node):
                return " | ".join(sorted({p for h in cands(node) for p in P.predecessors(h)}))

            name = REBUILT.get(ds, ds)
            total = missed = td = cb = 0
            for u in red.nodes():
                cu = cands(u)
                for v, hop in nx.single_source_shortest_path_length(red, u).items():
                    if hop == 0:
                        continue
                    total += 1
                    cv = cands(v)
                    if cu and cv and any((a, b) in clos for a in cu for b in cv):
                        continue
                    missed += 1
                    mode = "chain_broken" if (cu and cv) else "term_dropped"
                    td += mode == "term_dropped"
                    cb += mode == "chain_broken"
                    out.append({
                        "model": model_name, "dataset": name, "parent": u, "child": v,
                        "hop": hop, "parent_mentioned": "yes" if cu else "no",
                        "child_mentioned": "yes" if cv else "no", "failure_mode": mode,
                        "pred_parents_of_child": pred_parents(v) if cv else "",
                        "source_run": src, "your_label": "", "notes": ""})
            print(f"{model_name:16s} {name:13s} {total:9d} {missed:7d} {td:13d} {cb:13d}")
    for i, row in enumerate(out, 1):
        row["fn_id"] = f"FN{i:04d}"
    cols = ["fn_id", "model", "dataset", "parent", "child", "hop", "parent_mentioned",
            "child_mentioned", "failure_mode", "pred_parents_of_child", "source_run",
            "your_label", "notes"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT}: {len(out)} false negatives")


if __name__ == "__main__":
    main()
