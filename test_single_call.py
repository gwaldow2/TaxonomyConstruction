"""Self-checks for single_call_method. No GPU, no server -- the client is stubbed.

    python test_single_call.py
"""

import networkx as nx

from single_call_method import build_single_call_prompt, parse_pairs, method_single_call


class _Client:
    """Stub OpenAI client returning a canned (content, reasoning) pair."""

    def __init__(self, content, reasoning=""):
        outer = self

        class _Comp:
            def create(self, model, messages, temperature, max_tokens):
                outer.last_prompt = messages[0]["content"]
                msg = type("M", (), {"content": outer._content, "reasoning": outer._reasoning})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self._content, self._reasoning = content, reasoning
        self.chat = type("Chat", (), {"completions": _Comp()})()
        self.last_prompt = None


NODES = ["food", "fruit", "apple", "vehicle", "car"]


def test_prompt_contains_every_term_and_count():
    p = build_single_call_prompt(sorted(NODES))
    for t in NODES:
        assert t in p, t
    assert f"({len(NODES)} terms)" in p          # stated count == rendered list, by construction
    assert "EXAMPLE 1" in p and "EXAMPLE 2" in p
    assert "every instance of the child" in p    # the extensional is-a test survived


def test_parse_case_insensitive_and_vocab_bound():
    m = {"food": "food", "fruit": "fruit"}
    text = '[["Food", "Fruit"], ["fruit", "banana"], ["food", "food"]]'
    got = parse_pairs(text, m)
    assert got == [("food", "fruit")], got       # out-of-vocab and self-pairs dropped


def test_parse_survives_malformed_array():
    m = {"food": "food", "fruit": "fruit", "apple": "apple"}
    text = 'Sure! Here you go:\n[\n ["food", "fruit"],\n ["fruit", "apple"'   # truncated
    assert parse_pairs(text, m) == [("food", "fruit")]


def test_content_preferred_reasoning_fallback():
    c = _Client('[["food", "fruit"]]', '[["vehicle", "car"]]')
    G = method_single_call(NODES, c, "m", merge_synonyms=False)
    assert set(G.edges()) == {("food", "fruit")}          # content wins outright
    c = _Client("", '[["vehicle", "car"]]')
    G = method_single_call(NODES, c, "m", merge_synonyms=False)
    assert set(G.edges()) == {("vehicle", "car")}         # scratchpad rescues an empty answer


def test_merge_flag_controls_reciprocal_handling():
    both = '[["food", "fruit"], ["fruit", "food"], ["vehicle", "car"]]'
    G = method_single_call(NODES, _Client(both), "m", merge_synonyms=True)
    assert any("(" in str(n) or "|" in str(n) for n in G.nodes()) or "food" not in G.nodes(), \
        "reciprocal pair should have merged into a cluster node"
    assert nx.is_directed_acyclic_graph(G)
    G = method_single_call(NODES, _Client(both), "m", merge_synonyms=False)
    assert "food" in G.nodes() and "fruit" in G.nodes()   # no clustering
    assert nx.is_directed_acyclic_graph(G)                # but the 2-cycle was broken


def test_empty_response_yields_empty_dag():
    G = method_single_call(NODES, _Client("", ""), "m")
    assert G.number_of_edges() == 0
    assert nx.is_directed_acyclic_graph(G)


def test_synset_nodes_prompted_by_primary_term():
    nodes = ["food (food, nutrient)", "fruit"]
    c = _Client('[["food", "fruit"]]')
    G = method_single_call(nodes, c, "m", merge_synonyms=False)
    assert "nutrient)" not in c.last_prompt.split("NOW THE REAL TASK")[1], \
        "prompt must show the primary term, not the raw cluster string"
    assert set(G.edges()) == {("food (food, nutrient)", "fruit")}   # mapped back to full node


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll single_call checks passed.")
