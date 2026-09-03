"""Self-checks for synonym_reconstruction's clustering and scoring. No GPU, no LLM, no data.

    python test_synonym_reconstruction.py
"""

import networkx as nx

from synonym_reconstruction import (mutual_components, pair_set, score_partition,
                                    gt_cluster_report, pred_cluster_report)


def _graph(edges, nodes=()):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def test_mutual_components_merge_only_reciprocal():
    G = _graph([("a", "b"), ("b", "a"), ("b", "c"), ("x", "y")], nodes="abcxy")
    comps, n_recip = mutual_components(G)
    assert n_recip == 1
    assert frozenset("ab") in comps                     # merged via mutual edges
    assert frozenset("c") in comps and frozenset("x") in comps   # one-way edges never merge


def test_mutual_components_chain_transitively():
    """a<->b and b<->c chain into one component -- the mega-cluster mechanism."""
    G = _graph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")])
    comps, n_recip = mutual_components(G)
    assert frozenset("abc") in comps and n_recip == 2


def test_pairwise_scores_exact_values():
    #  GT: {a,b,c} {d}   PRED: {a,b} {c} {d}  -> tp={ab}, pred pairs 1, gt pairs 3
    s = score_partition([frozenset("ab"), frozenset("c"), frozenset("d")],
                        [frozenset("abc"), frozenset("d")])
    assert (s["pair_tp"], s["pair_pred"], s["pair_gt"]) == (1, 1, 3)
    assert s["precision"] == 1.0 and abs(s["recall"] - 1 / 3) < 1e-9


def test_vacuous_precision_and_recall():
    s = score_partition([frozenset("a"), frozenset("b")], [frozenset("ab")])
    assert s["precision"] == 1.0 and s["recall"] == 0.0      # nothing merged, one pair missed
    s = score_partition([frozenset("ab")], [frozenset("a"), frozenset("b")])
    assert s["recall"] == 1.0 and s["precision"] == 0.0      # merged where GT has no pairs


def test_gt_cluster_statuses():
    gt = {"ab-node": frozenset("ab"), "cd-node": frozenset("cd"), "ef-node": frozenset("ef"),
          "solo": frozenset("s")}
    pred = [frozenset("ab"),            # exact
            frozenset("cdx"),           # merged: kept together plus an outsider
            frozenset("e"), frozenset("f"),  # split
            frozenset("s"), frozenset("x")]
    rows = {r["cluster"]: r for r in gt_cluster_report(gt, pred)}
    assert rows["ab-node"]["status"] == "exact"
    assert rows["cd-node"]["status"] == "merged"
    assert rows["ef-node"]["status"] == "split" and rows["ef-node"]["n_pred_components"] == 2
    assert "solo" not in rows                                # singletons are not scored


def test_pred_report_flags_cross_cluster_merges():
    pred = [frozenset("abx"), frozenset("c")]
    t2c = {"a": "A", "b": "A", "x": "X", "c": "C"}
    rows = pred_cluster_report(pred, t2c)
    assert len(rows) == 1                                    # singletons omitted
    assert rows[0]["n_gt_clusters_spanned"] == 2 and rows[0]["size"] == 3


def test_pair_set_is_unordered_and_dedup():
    assert pair_set([frozenset("ab")]) == pair_set([frozenset("ba")])
    assert len(pair_set([frozenset("abc")])) == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll synonym_reconstruction checks passed.")
