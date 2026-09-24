"""Alternative (advisor-suite) metrics computed from saved prediction graphs.

Runs ENTIRELY on saved artifacts -- no LLM, no server. Inputs are the per-run
prediction graphs the evaluator now writes (``results/<ds>_<label>[_<tag>]_pred.graphml``,
carrying model/tag/timestamp provenance as graph attributes) or, for legacy runs that
predate graph saving, an edge-diagnostics CSV (``--from_diag``).

Per prediction it reports, against ``results/GT_<dataset>_eval.graphml``:

  * all-edges recall      -- share of GT transitive-closure pairs covered by the
                             prediction's closure (synonym-normalized node matching);
  * k-hop recall          -- the same, stratified by the pair's distance in the GT
                             transitive reduction (k=1 are direct edges); a chain
                             method loses high-k pairs whenever a mid-chain link is
                             missing, so this locates WHERE recall is lost;
  * direct / indirect recall -- k=1 vs k>1 roll-ups;
  * precision (closure-aware) -- a predicted direct edge counts as false only if it
                             is absent from the GT closure (skip edges to true
                             ancestors are never penalized);
  * F1                    -- harmonic mean of closure-aware precision and all-edges
                             recall;
  * accuracy              -- over all ordered node pairs, the fraction whose
                             edge-presence (in closure) is predicted correctly,
                             reported NEXT TO the trivial predict-nothing baseline,
                             which the class imbalance places near 1.0.

Synonym-recovery metrics are intentionally NOT duplicated here: see
implied_edge_analysis.py (per-synonym-count P/R/F1 from report txts) and
synonym_reconstruction.py (direct cluster recovery).

    python alt_metrics.py                                   # all *_pred.graphml in results/
    python alt_metrics.py --pred_glob "*gemma4b_pred.graphml"
    python alt_metrics.py --from_diag --pred_glob "*_gemma4_edge_diagnostics.csv"
    python alt_metrics.py --selftest

Outputs: alt_metrics_summary.csv (one row per prediction, provenance included) and
alt_metrics_by_hop.csv (one row per prediction x hop distance).
"""

import os
import csv
import glob
import argparse

import networkx as nx

from data_manager import get_primary_term

RESULTS_DIR = "results"


def load_gt(results_dir, dataset):
    p = os.path.join(results_dir, f"GT_{dataset}_eval.graphml")
    if not os.path.exists(p):
        return None
    G = nx.DiGraph(nx.read_graphml(p))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


def map_edges_to_gt(edges, G_gt):
    """Map predicted (parent, child) name pairs onto GT nodes via primary-term
    normalization (handles lemma-format vs primary-term naming). -> (DiGraph over
    GT nodes, n_unmapped_edges)."""
    pmap = {get_primary_term(n): n for n in G_gt.nodes()}
    P = nx.DiGraph()
    P.add_nodes_from(G_gt.nodes())
    unmapped = 0
    for p, c in edges:
        gp, gc = pmap.get(get_primary_term(p)), pmap.get(get_primary_term(c))
        if gp is not None and gc is not None and gp != gc:
            P.add_edge(gp, gc)
        else:
            unmapped += 1
    return P, unmapped


def alt_metrics(pred_edges, G_gt):
    """-> (summary dict, {k: (hits, total)}) for one prediction against one GT."""
    P, unmapped = map_edges_to_gt(pred_edges, G_gt)
    red = nx.transitive_reduction(G_gt)
    clos = nx.transitive_closure(G_gt)
    Pc = nx.transitive_closure(P)

    by_k = {}
    for u, v in clos.edges():
        k = nx.shortest_path_length(red, u, v)
        h, t = by_k.get(k, (0, 0))
        by_k[k] = (h + int(Pc.has_edge(u, v)), t + 1)

    hits = sum(h for h, _ in by_k.values())
    total = sum(t for _, t in by_k.values())
    recall_all = hits / total if total else 0.0
    d_h, d_t = by_k.get(1, (0, 0))
    i_h = hits - d_h
    i_t = total - d_t

    # closure-aware precision over the predicted DIRECT edges as given (unmapped
    # edges are predictions that match nothing in the GT vocabulary: false)
    tp_direct = sum(1 for u, v in P.edges() if clos.has_edge(u, v))
    n_pred = P.number_of_edges() + unmapped
    precision = tp_direct / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall_all / (precision + recall_all)
          if precision + recall_all else 0.0)

    n = G_gt.number_of_nodes()
    pairs = n * (n - 1)
    fp_pairs = sum(1 for u, v in Pc.edges() if u != v and not clos.has_edge(u, v))
    accuracy = (pairs - fp_pairs - (total - hits)) / pairs if pairs else 0.0
    trivial = 1 - total / pairs if pairs else 0.0

    return ({"n_pred_edges": n_pred, "n_unmapped": unmapped,
             "recall_all": recall_all, "recall_direct": (d_h / d_t if d_t else 0.0),
             "recall_indirect": (i_h / i_t if i_t else 0.0),
             "precision_closure_aware": precision, "f1_alt": f1,
             "accuracy": accuracy, "accuracy_trivial_baseline": trivial},
            by_k)


def dataset_from_filename(path):
    """results/<dataset>_<rest> -> dataset, matched against existing GT files."""
    base = os.path.basename(path)
    d = os.path.dirname(path) or RESULTS_DIR
    cands = [os.path.basename(g)[len("GT_"):-len("_eval.graphml")]
             for g in glob.glob(os.path.join(d, "GT_*_eval.graphml"))]
    for ds in sorted(cands, key=len, reverse=True):
        if base.startswith(ds + "_"):
            return ds
    return None


def load_pred_file(path, from_diag):
    """-> (edge list, provenance dict, label)."""
    base = os.path.basename(path)
    if from_diag:
        with open(path, newline="", encoding="utf-8") as f:
            edges = [(r["parent"], r["child"]) for r in csv.DictReader(f)]
        return edges, {}, base.replace("_edge_diagnostics.csv", "")
    G = nx.DiGraph(nx.read_graphml(path))
    prov = {k: G.graph.get(k, "") for k in ("model", "tag", "timestamp", "results_file")}
    return list(G.edges()), prov, base.replace("_pred.graphml", "")


def selftest():
    # chain a <- b <- c plus synset node; every case hand-checkable
    G = nx.DiGraph([("a", "b"), ("b", "c (c, c-alias)")])
    perfect, _ = alt_metrics([("a", "b"), ("b", "c")], G)
    assert perfect["recall_all"] == 1.0 and perfect["precision_closure_aware"] == 1.0
    assert perfect["accuracy"] == 1.0, perfect
    skip_only, by_k = alt_metrics([("a", "c")], G)          # true skip edge, k=2 pair only
    assert by_k[2] == (1, 1) and by_k[1] == (0, 2)
    assert skip_only["precision_closure_aware"] == 1.0      # in closure: never false
    broken, by_k = alt_metrics([("a", "b")], G)             # missing mid-chain link
    assert by_k[1] == (1, 2) and by_k[2] == (0, 1)          # k=2 lost with the chain
    wrong, _ = alt_metrics([("c", "a")], G)                 # reversed: not in closure
    assert wrong["precision_closure_aware"] == 0.0
    nothing, _ = alt_metrics([], G)
    assert nothing["accuracy"] == nothing["accuracy_trivial_baseline"]
    print("alt_metrics selftest OK")


def main():
    ap = argparse.ArgumentParser(description="Advisor-suite metrics from saved predictions.")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--pred_glob", default="*_pred.graphml",
                    help="Glob (within --results_dir) of prediction files to score.")
    ap.add_argument("--from_diag", action="store_true",
                    help="Treat matched files as edge-diagnostics CSVs (legacy runs).")
    ap.add_argument("--out_prefix", default=os.path.join(RESULTS_DIR, "alt_metrics"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.pred_glob)))
    if not paths:
        raise SystemExit(f"[!] nothing matches {args.pred_glob} in {args.results_dir}/")
    rows, hop_rows = [], []
    for path in paths:
        ds = dataset_from_filename(path)
        if ds is None:
            print(f"    [!] {os.path.basename(path)}: no matching GT graph -- skipped")
            continue
        G_gt = load_gt(args.results_dir, ds)
        edges, prov, label = load_pred_file(path, args.from_diag)
        summary, by_k = alt_metrics(edges, G_gt)
        row = {"dataset": ds, "run": label, **prov, **{k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in summary.items()}}
        rows.append(row)
        for k in sorted(by_k):
            h, t = by_k[k]
            hop_rows.append({"dataset": ds, "run": label, "model": prov.get("model", ""),
                             "hop": k, "hits": h, "total": t, "recall": round(h / t, 4)})
        print(f"    {label:55s} R_all={summary['recall_all']:.3f} "
              f"P={summary['precision_closure_aware']:.3f} F1={summary['f1_alt']:.3f} "
              f"acc={summary['accuracy']:.3f} (trivial {summary['accuracy_trivial_baseline']:.3f})")

    for suffix, data in [("_summary.csv", rows), ("_by_hop.csv", hop_rows)]:
        if not data:
            continue
        with open(args.out_prefix + suffix, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"[*] wrote {args.out_prefix}{suffix} ({len(data)} rows)")


if __name__ == "__main__":
    main()
