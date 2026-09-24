"""Alternative (advisor-suite) metrics computed from saved prediction graphs.

Runs ENTIRELY on saved artifacts -- no LLM, no server. Inputs are the per-run
prediction graphs the evaluator writes (``results/<ds>_<label>[_<tag>]_pred.graphml``,
carrying model/tag/timestamp provenance as graph attributes) or, for legacy runs that
predate graph saving, an edge-diagnostics CSV (``--from_diag``).

Node matching is SYNONYM-AWARE by default (``--match overlap``), mirroring the
evaluator's condensed set_overlap semantics: every node name expands to its synonym
set via the lemma format (``primary (syn1, syn2)``), and a predicted node matches a
ground-truth node whenever their sets intersect. This matters because post-processing
merges reciprocal assertions into cluster nodes: under strict primary-term matching a
merged cluster earns credit for only ONE of its members, which deflates merge-heavy
models (measured on Gemma-4: all-edges recall 0.602 strict vs 0.863 overlap, enough to
flip a method ranking). ``--match strict`` retains the old behavior for exactly that
comparison.

Per prediction it reports, against ``results/GT_<dataset>_eval.graphml``:

  * all-edges recall      -- share of GT transitive-closure pairs covered by the
                             prediction's closure;
  * k-hop recall          -- the same, stratified by the pair's distance in the GT
                             transitive reduction (k=1 are direct edges); a chain
                             method loses high-k pairs whenever a mid-chain link is
                             missing, so this locates WHERE recall is lost;
  * direct / indirect recall -- k=1 vs k>1 roll-ups;
  * precision (closure-aware) -- a predicted direct edge counts as false only if it
                             matches no pair in the GT closure (skip edges to true
                             ancestors are never penalized);
  * F1                    -- harmonic mean of closure-aware precision and all-edges
                             recall;
  * accuracy              -- over all ordered GT node pairs, the fraction whose
                             edge-presence (in closure) is predicted correctly,
                             reported NEXT TO the trivial predict-nothing baseline,
                             which the class imbalance places near 1.0.

Synonym-recovery metrics are intentionally NOT duplicated here: see
implied_edge_analysis.py (per-synonym-count P/R/F1 from report txts) and
synonym_reconstruction.py (direct cluster recovery).

    python alt_metrics.py                                   # all *_pred.graphml in results/
    python alt_metrics.py --pred_glob "*gemma4b_pred.graphml"
    python alt_metrics.py --from_diag --pred_glob "*_gemma4_edge_diagnostics.csv"
    python alt_metrics.py --match strict                    # legacy exact-primary matching
    python alt_metrics.py --selftest

Outputs: alt_metrics_summary.csv (one row per prediction, provenance included) and
alt_metrics_by_hop.csv (one row per prediction x hop distance).
"""

import os
import csv
import glob
import argparse

import networkx as nx

from data_manager import get_primary_term, parse_lemma_format

RESULTS_DIR = "results"


def load_gt(results_dir, dataset):
    p = os.path.join(results_dir, f"GT_{dataset}_eval.graphml")
    if not os.path.exists(p):
        return None
    G = nx.DiGraph(nx.read_graphml(p))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


def term_set(name):
    return set(parse_lemma_format(name))


def build_pred_graph(edges):
    """Predicted direct edges as a DiGraph over their own (possibly lemma-format)
    node names; self-loops after name identity are dropped."""
    P = nx.DiGraph()
    for p, c in edges:
        if p != c:
            P.add_edge(p, c)
    return P


def map_edges_to_gt(edges, G_gt):
    """STRICT mapping (legacy): predicted names onto GT nodes via primary-term
    equality. Kept for the --match strict path and for measuring the gap the
    overlap semantics close. -> (DiGraph over GT nodes, n_unmapped_edges)."""
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


def _metrics_from_cover(by_k, tp_direct, n_pred, fp_pairs, n_nodes):
    hits = sum(h for h, _ in by_k.values())
    total = sum(t for _, t in by_k.values())
    recall_all = hits / total if total else 0.0
    d_h, d_t = by_k.get(1, (0, 0))
    i_h, i_t = hits - d_h, total - d_t
    precision = tp_direct / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall_all / (precision + recall_all)
          if precision + recall_all else 0.0)
    pairs = n_nodes * (n_nodes - 1)
    accuracy = (pairs - fp_pairs - (total - hits)) / pairs if pairs else 0.0
    trivial = 1 - total / pairs if pairs else 0.0
    return {"n_pred_edges": n_pred,
            "recall_all": recall_all, "recall_direct": (d_h / d_t if d_t else 0.0),
            "recall_indirect": (i_h / i_t if i_t else 0.0),
            "precision_closure_aware": precision, "f1_alt": f1,
            "accuracy": accuracy, "accuracy_trivial_baseline": trivial}


def alt_metrics(pred_edges, G_gt, match="overlap"):
    """-> (summary dict, {k: (hits, total)}) for one prediction against one GT."""
    red = nx.transitive_reduction(G_gt)
    clos = nx.transitive_closure(G_gt)

    if match == "strict":
        P, unmapped = map_edges_to_gt(pred_edges, G_gt)
        Pc = nx.transitive_closure(P)
        by_k = {}
        for u, v in clos.edges():
            k = nx.shortest_path_length(red, u, v)
            h, t = by_k.get(k, (0, 0))
            by_k[k] = (h + int(Pc.has_edge(u, v)), t + 1)
        tp = sum(1 for u, v in P.edges() if clos.has_edge(u, v))
        n_pred = P.number_of_edges() + unmapped
        fp_pairs = sum(1 for u, v in Pc.edges() if u != v and not clos.has_edge(u, v))
        out = _metrics_from_cover(by_k, tp, n_pred, fp_pairs, G_gt.number_of_nodes())
        out["n_unmapped"] = unmapped
        return out, by_k

    # ---- synonym-aware set-overlap matching (default) --------------------------
    P = build_pred_graph(pred_edges)
    Pc = nx.transitive_closure(P)
    gt_terms = {n: term_set(n) for n in G_gt.nodes()}
    p_terms = {n: term_set(n) for n in P.nodes()}
    # GT node -> predicted nodes whose synonym sets intersect it (a merged cluster
    # may match several GT nodes; that is the point of overlap matching)
    matches = {g: [p for p, pt in p_terms.items() if pt & gts]
               for g, gts in gt_terms.items()}

    def covered(u, v):
        return any(Pc.has_edge(pu, pv)
                   for pu in matches[u] for pv in matches[v] if pu != pv)

    by_k = {}
    for u, v in clos.edges():
        k = nx.shortest_path_length(red, u, v)
        h, t = by_k.get(k, (0, 0))
        by_k[k] = (h + int(covered(u, v)), t + 1)

    clos_pairs = list(clos.edges())
    tp = sum(1 for pu, pv in P.edges()
             if any(p_terms[pu] & gt_terms[u] and p_terms[pv] & gt_terms[v]
                    for u, v in clos_pairs))
    n_pred = P.number_of_edges()
    unmapped = sum(1 for pu, pv in P.edges()
                   if not any(p_terms[pu] & g for g in gt_terms.values())
                   or not any(p_terms[pv] & g for g in gt_terms.values()))
    # accuracy over ordered GT node pairs: a pair is asserted iff some matching
    # predicted pair is in the predicted closure
    fp_pairs = sum(1 for u in G_gt.nodes() for v in G_gt.nodes()
                   if u != v and not clos.has_edge(u, v) and covered(u, v))
    out = _metrics_from_cover(by_k, tp, n_pred, fp_pairs, G_gt.number_of_nodes())
    out["n_unmapped"] = unmapped
    return out, by_k


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
    # chain a <- b <- c(+alias); every case hand-checkable, both match modes
    G = nx.DiGraph([("a", "b"), ("b", "c (c, c-alias)")])
    for match in ("overlap", "strict"):
        perfect, _ = alt_metrics([("a", "b"), ("b", "c")], G, match)
        assert perfect["recall_all"] == 1.0 and perfect["precision_closure_aware"] == 1.0
        assert perfect["accuracy"] == 1.0, (match, perfect)
        skip_only, by_k = alt_metrics([("a", "c")], G, match)   # true skip edge
        assert by_k[2] == (1, 1) and by_k[1] == (0, 2)
        assert skip_only["precision_closure_aware"] == 1.0
        broken, by_k = alt_metrics([("a", "b")], G, match)      # missing mid-chain link
        assert by_k[1] == (1, 2) and by_k[2] == (0, 1)
        wrong, _ = alt_metrics([("c", "a")], G, match)          # reversed
        assert wrong["precision_closure_aware"] == 0.0
        nothing, _ = alt_metrics([], G, match)
        assert nothing["accuracy"] == nothing["accuracy_trivial_baseline"]

    # the case the overlap semantics exist for: a merged cluster node must earn
    # credit for EVERY GT node it contains, not just its primary term
    G2 = nx.DiGraph([("root", "germ cell"), ("root", "germ line cell")])
    cluster_edges = [("root", "germ cell (germ cell, germ line cell)")]
    ov, by_k = alt_metrics(cluster_edges, G2, "overlap")
    assert by_k[1] == (2, 2), by_k                      # both GT edges covered
    assert ov["precision_closure_aware"] == 1.0 and ov["accuracy"] == 1.0
    st, by_k = alt_metrics(cluster_edges, G2, "strict")
    assert by_k[1] == (1, 2), by_k                      # strict credits only the primary
    # alias naming on the GT side must also match a plain predicted name
    ov2, _ = alt_metrics([("a", "c-alias")], nx.DiGraph([("a", "c (c, c-alias)")]), "overlap")
    assert ov2["recall_all"] == 1.0 and ov2["n_unmapped"] == 0
    print("alt_metrics selftest OK")


def main():
    ap = argparse.ArgumentParser(description="Advisor-suite metrics from saved predictions.")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--pred_glob", default="*_pred.graphml",
                    help="Glob (within --results_dir) of prediction files to score.")
    ap.add_argument("--from_diag", action="store_true",
                    help="Treat matched files as edge-diagnostics CSVs (legacy runs).")
    ap.add_argument("--match", choices=["overlap", "strict"], default="overlap",
                    help="Node matching: synonym-aware set overlap (default, mirrors the "
                         "condensed metrics) or legacy strict primary-term equality.")
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
        summary, by_k = alt_metrics(edges, G_gt, args.match)
        row = {"dataset": ds, "run": label, "match": args.match, **prov,
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}}
        rows.append(row)
        for k in sorted(by_k):
            h, t = by_k[k]
            hop_rows.append({"dataset": ds, "run": label, "match": args.match,
                             "model": prov.get("model", ""),
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
