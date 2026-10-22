"""Export every false-positive edge of the structured method on the seven-dataset core.

An edge is a false positive when its parent->child relation is absent from the
reference's transitive closure (synonym set-overlap) -- the scorer's own definition
(evaluator.gt_closure_term_pairs / edge_is_correct), recomputed here for every
dataset so all rows share one semantics. One row per asserted edge.

Sources (the run behind each row is named in the source_run column):
  rebuilt four, both models   $TAXO_V2_CAMP prediction graphs (the merged core's runs)
  legacy three, Gemma         x2_20260923 diagnostics: the control_gemma4N run itself
                              (re-scoring them reproduces the control scores exactly)
  legacy three, Gemini        battery_20260925 predictions: a RERUN of the control
                              configuration; the control_gemini run's predictions were
                              not archived, so these rows do not match table scores

Output: results/corrected_core/fp_structured_core.csv
"""
import csv
import os
import sys

import networkx as nx

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import edge_is_correct, gt_closure_term_pairs  # noqa: E402
from data_manager import parse_lemma_format  # noqa: E402

CAMP = os.environ.get("TAXO_V2_CAMP", f"{BASE}/results/v2r2_20261021")
BATT = f"{BASE}/results/battery_20260925/preds"
OUT = f"{BASE}/results/corrected_core/fp_structured_core.csv"

LEG3 = ["WordNetFood", "CellOntology", "SemEvalFood"]
REBUILT = {"SchemaOrgV2": "SchemaOrg", "POV2": "PO", "OBIV2": "OBI", "MatOntoV2": "MatOnto"}
MODELS = {"gemma4": "Gemma-4-31B", "gemini31": "Gemini 3.1 Pro"}


def strip_vr(G):
    G = G.copy()
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


def pred_edges(model, ds):
    """-> (edges, source_run) for the structured run behind this cell."""
    if ds in LEG3:
        if model == "gemma4":
            path = f"{BASE}/results/x2_20260923/{ds}_SUB_gemma4_diag.csv"
            rows = csv.DictReader(open(path, encoding="utf-8"))
            return [(r["parent"], r["child"]) for r in rows], "control_gemma4N (x2_20260923 diag)"
        P = nx.read_graphml(f"{BATT}/{ds}_SUB_Our_Method_k1000_clawback0_gemini31b_pred.graphml")
        return list(strip_vr(P).edges()), "battery_20260925 rerun (NOT the table's control_gemini run)"
    P = nx.read_graphml(f"{CAMP}/results/{ds}_SUB_Our_Method_k1000_clawback0_{model}_v2_pred.graphml")
    return list(strip_vr(P).edges()), f"{os.path.basename(CAMP)} {model}_v2"


def gt_graph(ds):
    path = (f"{BATT}/GT_{ds}_SUB_eval.graphml" if ds in LEG3
            else f"{CAMP}/results/GT_{ds}_SUB_eval.graphml")
    return nx.read_graphml(path)


def main():
    out = []
    print(f"{'model':16s} {'dataset':13s} {'edges':>6s} {'FPs':>5s} {'edge-level precision':>21s}")
    for model, model_name in MODELS.items():
        for ds in LEG3 + list(REBUILT):
            G = gt_graph(ds)
            pairs = gt_closure_term_pairs(G)
            red = nx.transitive_reduction(strip_vr(G))
            by_term = {}
            for n in red.nodes():
                for t in parse_lemma_format(n):
                    by_term.setdefault(t, set()).add(n)

            def ref_parents(node):
                hits = set().union(*(by_term.get(t, set()) for t in parse_lemma_format(node)))
                if not hits:
                    return None
                return " | ".join(sorted({p for h in hits for p in red.predecessors(h)})) or "(root)"

            edges, src = pred_edges(model, ds)
            name = REBUILT.get(ds, ds)
            n_fp = 0
            for parent, child in edges:
                if edge_is_correct(parent, child, pairs):
                    continue
                n_fp += 1
                rc, rp = ref_parents(child), ref_parents(parent)
                out.append({
                    "model": model_name, "dataset": name, "parent": parent, "child": child,
                    "child_in_reference": "yes" if rc is not None else "no",
                    "reference_parents_of_child": rc or "",
                    "reference_parents_of_parent": rp or "",
                    "source_run": src, "your_label": "", "notes": ""})
            print(f"{model_name:16s} {name:13s} {len(edges):6d} {n_fp:5d} {1 - n_fp / len(edges):21.3f}")
    for i, row in enumerate(out, 1):
        row["fp_id"] = f"FP{i:04d}"
    cols = ["fp_id", "model", "dataset", "parent", "child", "child_in_reference",
            "reference_parents_of_child", "reference_parents_of_parent", "source_run",
            "your_label", "notes"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT}: {len(out)} false positives")


if __name__ == "__main__":
    main()
