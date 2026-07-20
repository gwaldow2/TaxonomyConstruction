"""Self-checks for lora_data_prep. No GPU, no server, no benchmark files needed.

    python test_lora_prep.py
"""

import networkx as nx

from lora_data_prep import (leakage_report, make_examples, build_full_prompt,
                            train_pairs_from_graph)


def test_closure_leakage():
    """A train pair that is only an INDIRECT ancestor pair in eval still leaks the answer."""
    G = nx.DiGraph([("animal", "mammal"), ("mammal", "dog")])
    assert not G.has_edge("animal", "dog")
    viol, _ = leakage_report([{"parent": "animal", "child": "dog"}], G)
    assert viol == [("animal", "dog")], viol


def test_synonym_leakage():
    G = nx.DiGraph([("animal", "mammal"), ("mammal", "dog")])
    viol, _ = leakage_report([{"parent": "animal|beast", "child": "dog"}], G)
    assert viol, "synonym-aware match missed a leaked pair"


def test_clean_split_passes():
    G = nx.DiGraph([("animal", "mammal"), ("mammal", "dog")])
    viol, overlap = leakage_report([{"parent": "vehicle", "child": "car"}], G)
    assert viol == [] and overlap == 0.0, (viol, overlap)


TRAIN = [{"parent": "food", "child": "fruit"}, {"parent": "fruit", "child": "apple"},
         {"parent": "food", "child": "vegetable"}, {"parent": "vegetable", "child": "carrot"}]


def test_completions_use_closure():
    by = {e["target"]: e for e in make_examples(TRAIN, 99)}
    assert len(by) == 5
    # apple's grandparent 'food' must appear -- the prompt asks for ancestors, not just parents
    assert set(by["apple"]["completion"].split("\n")) == {"apple <= fruit", "apple <= food"}
    assert set(by["food"]["completion"].split("\n")) == {
        "fruit <= food", "apple <= food", "vegetable <= food", "carrot <= food"}


def test_candidate_budget():
    for e in make_examples(TRAIN, candidates_per_example=2):
        assert len(e["candidates"]) == 2, e
        assert len(set(e["candidates"])) == 2, e        # positives and negatives must not overlap
        assert e["target"] not in e["candidates"], e


def test_completions_only_cite_listed_candidates():
    for e in make_examples(TRAIN, 99):
        for line in e["completion"].split("\n"):
            if line == "none":
                continue
            child, parent = [s.strip() for s in line.split("<=")]
            assert e["target"] in (child, parent), line
            other = parent if child == e["target"] else child
            assert other in e["candidates"], (line, e["candidates"])


def test_deterministic_and_empty():
    assert make_examples(TRAIN, 99) == make_examples(TRAIN, 99)
    assert make_examples([]) == []


def _full_and_sub():
    """A FULL graph with SUB as a genuine subgraph, mirroring the benchmark layout."""
    G = nx.DiGraph([("food", "fruit"), ("fruit", "apple"), ("fruit", "pear"),
                    ("food", "vegetable"), ("vegetable", "carrot"), ("carrot", "baby carrot"),
                    ("food", "grain"), ("grain", "rice"), ("grain", "wheat")])
    G_sub = G.subgraph(["food", "fruit", "apple", "pear"]).copy()
    return G, G_sub


def test_graph_pool_node_disjoint_is_clean():
    G_full, G_sub = _full_and_sub()
    kept, stats = train_pairs_from_graph(G_full, G_sub, node_disjoint=True)
    pairs = {(tp["parent"], tp["child"]) for tp in kept}
    # every edge touching an eval node (all share ancestor 'food') is gone
    assert not any(n in pr for pr in pairs for n in G_sub.nodes()), pairs
    assert ("vegetable", "carrot") in pairs and ("grain", "rice") in pairs
    assert leakage_report(kept, G_sub)[0] == []            # must be leakage-clean
    assert stats["kept"] == len(kept) and stats["pool_edges"] == G_full.number_of_edges()


def test_graph_pool_edge_disjoint_still_filters_closure():
    G_full, G_sub = _full_and_sub()
    kept, _ = train_pairs_from_graph(G_full, G_sub, node_disjoint=False)
    # keeping eval nodes is allowed, but no kept pair may appear in the eval closure
    assert leakage_report(kept, G_sub)[0] == []
    pairs = {(tp["parent"], tp["child"]) for tp in kept}
    assert ("food", "fruit") not in pairs and ("fruit", "apple") not in pairs


def test_prompt_matches_live():
    """The built-in 'full' prompt copy must stay byte-identical to our_method.build_prompt.

    Skipped where the ontology stack isn't installed (the training-only env can't import
    our_method); it runs in the benchmark env, which is where a prompt edit would land and
    silently drift the training format away from inference.
    """
    try:
        from our_method import build_prompt
    except Exception:
        print("      (skipped test_prompt_matches_live -- our_method not importable here)")
        return
    for target, cands in [("fruit", ["food", "apple"]), ("cell", []),
                          ("x'y", ["a-b", "c", "d e"]), ("t", ["only"])]:
        assert build_full_prompt(target, cands) == build_prompt(target, cands, variant="full"), target


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll lora_data_prep checks passed.")
