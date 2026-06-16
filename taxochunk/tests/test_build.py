"""Tests for taxochunk -- run with `python -m pytest` or `python -m unittest`.

All LLM calls are stubbed with a fixed `respond` callable, so no endpoint is needed.
"""

import unittest
import networkx as nx

from taxochunk import (
    build_taxonomy, to_lemma_format, get_primary_term, parse_lemma_format, parse_response,
)


def const(text):
    """A respond() stub that returns the same text for every prompt."""
    return lambda prompt: text


class TextHelpers(unittest.TestCase):
    def test_primary_and_parse(self):
        self.assertEqual(get_primary_term("apple (apple, malus pumila)"), "apple")
        self.assertEqual(get_primary_term("apple"), "apple")
        self.assertEqual(parse_lemma_format("a|b|c"), ["a", "b", "c"])
        self.assertEqual(parse_lemma_format("apple (apple, malus pumila)"), ["apple", "malus pumila"])

    def test_to_lemma_format(self):
        self.assertEqual(to_lemma_format(["apple"]), "apple")
        self.assertEqual(to_lemma_format(["apple", "malus pumila"]), "apple (apple, malus pumila)")


class ParseResponse(unittest.TestCase):
    def test_standard(self):
        vocab = {"food", "fruit", "apple"}
        pairs = parse_response("fruit <= food\napple <= fruit\nxyz <= food", vocab)
        self.assertEqual(set(pairs), {("food", "fruit"), ("fruit", "apple")})  # xyz dropped

    def test_alt_json(self):
        vocab = {"food", "fruit", "apple"}
        pairs = parse_response('garbage [["food","fruit"],["fruit","apple"]] trailing', vocab, alt_prompt=True)
        self.assertEqual(set(pairs), {("food", "fruit"), ("fruit", "apple")})


class BuildTaxonomy(unittest.TestCase):
    def test_chain(self):
        G = build_taxonomy(
            ["food", "fruit", "apple", "granny smith"],
            respond=const("fruit <= food\napple <= fruit\ngranny smith <= apple"),
            keep_isolated=False,
        )
        self.assertTrue(nx.is_directed_acyclic_graph(G))
        self.assertEqual(set(G.edges()), {("food", "fruit"), ("fruit", "apple"), ("apple", "granny smith")})

    def test_alt_prompt(self):
        G = build_taxonomy(
            ["food", "fruit", "apple"],
            respond=const('[["food","fruit"],["fruit","apple"]]'),
            alt_prompt=True, keep_isolated=False,
        )
        self.assertEqual(set(G.edges()), {("food", "fruit"), ("fruit", "apple")})

    def test_synonym_clustering(self):
        # car <-> automobile are mutually is-a => merged into one cluster node.
        G = build_taxonomy(
            ["car", "automobile", "vehicle"],
            respond=const("car <= automobile\nautomobile <= car\ncar <= vehicle\nautomobile <= vehicle"),
            keep_isolated=False,
        )
        cluster = [n for n in G.nodes() if "automobile" in n and "car" in n]
        self.assertEqual(len(cluster), 1, G.nodes())
        self.assertEqual(set(G.edges()), {("vehicle", cluster[0])})

    def test_cycle_broken(self):
        # a -> b -> c -> a  (child <= parent encoding)
        G = build_taxonomy(
            ["a", "b", "c"],
            respond=const("b <= a\nc <= b\na <= c"),
            keep_isolated=False,
        )
        self.assertTrue(nx.is_directed_acyclic_graph(G))
        self.assertEqual(G.number_of_edges(), 2)  # one edge dropped to break the cycle

    def test_keep_isolated(self):
        respond = const("fruit <= food")  # 'lonely' gets no relations
        terms = ["food", "fruit", "lonely"]
        G_keep = build_taxonomy(terms, respond=respond, keep_isolated=True)
        G_drop = build_taxonomy(terms, respond=respond, keep_isolated=False)
        self.assertIn("lonely", G_keep.nodes())
        self.assertNotIn("lonely", G_drop.nodes())

    def test_requires_client_or_respond(self):
        with self.assertRaises(ValueError):
            build_taxonomy(["a", "b"])  # neither respond nor client/model


if __name__ == "__main__":
    unittest.main(verbosity=2)
