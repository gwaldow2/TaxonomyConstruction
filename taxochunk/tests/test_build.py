"""Tests for taxochunk -- run with `python -m pytest` or `python -m unittest`.

All LLM calls are stubbed with a fixed `respond` callable, so no endpoint is needed.
"""

import os
import csv
import json
import tempfile
import unittest
import networkx as nx

from taxochunk import (
    build_taxonomy, to_lemma_format, get_primary_term, parse_lemma_format, parse_response,
    TaxonomyBuilder, save_taxonomy, SUPPORTED_FORMATS,
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

    def test_prompts_differ_only_in_relationship_block(self):
        from taxochunk import build_prompt
        a = build_prompt("cell", ["x", "y"], alt_prompt=False).splitlines()
        b = build_prompt("cell", ["x", "y"], alt_prompt=True).splitlines()
        self.assertEqual(len(a), len(b))
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(diffs), 2)                         # exactly the two relationship bullets
        self.assertTrue(all("could logically" in x for x, _ in diffs))   # default: subsumption
        self.assertTrue(all("concept of" in y for _, y in diffs))        # alt: direct parent/child


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
        # alt_prompt changes only the prompt wording; output format/parsing are shared.
        G = build_taxonomy(
            ["food", "fruit", "apple"],
            respond=const("fruit <= food\napple <= fruit"),
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


class Builder(unittest.TestCase):
    def test_build_with_respond(self):
        b = TaxonomyBuilder(respond=const("fruit <= food\napple <= fruit"))
        G = b.build(["food", "fruit", "apple"], keep_isolated=False)
        self.assertEqual(set(G.edges()), {("food", "fruit"), ("fruit", "apple")})

    def test_override_options(self):
        b = TaxonomyBuilder(respond=const("fruit <= food"), keep_isolated=True)
        self.assertIn("x", b.build(["food", "fruit", "x"]).nodes())   # default keeps isolated
        self.assertNotIn("x", b.build(["food", "fruit", "x"], keep_isolated=False).nodes())  # per-call override

    def test_requires_client_or_respond(self):
        with self.assertRaises(ValueError):
            TaxonomyBuilder()                      # nothing supplied
        with self.assertRaises(ValueError):
            TaxonomyBuilder(client=object())       # client but no model

    def test_build_and_save(self):
        b = TaxonomyBuilder(respond=const("fruit <= food"))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.graphml")
            G = b.build_and_save(["food", "fruit"], path, keep_isolated=False)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(set(nx.read_graphml(path).edges()), set(G.edges()))


class Export(unittest.TestCase):
    def setUp(self):
        self.G = nx.DiGraph([("food", "fruit"), ("fruit", "apple")])

    def test_all_formats_write(self):
        with tempfile.TemporaryDirectory() as td:
            for fmt in SUPPORTED_FORMATS:
                p = os.path.join(td, f"t.{fmt}")
                save_taxonomy(self.G, p, fmt=fmt)
                self.assertTrue(os.path.getsize(p) > 0, fmt)

    def test_format_inferred_from_extension(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.graphml")
            save_taxonomy(self.G, p)  # no fmt -> infer graphml
            self.assertEqual(set(nx.read_graphml(p).edges()), set(self.G.edges()))

    def test_roundtrip_tsv_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            tsv = os.path.join(td, "t.tsv")
            save_taxonomy(self.G, tsv, fmt="tsv")
            with open(tsv, encoding="utf-8") as f:
                rows = list(csv.reader(f, delimiter="\t"))
            self.assertEqual(rows[0], ["parent", "child"])
            self.assertEqual({tuple(r) for r in rows[1:]}, set(self.G.edges()))

            js = os.path.join(td, "t.json")
            save_taxonomy(self.G, js, fmt="json")
            with open(js, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["nodes"]), 3)

    def test_bad_format_raises(self):
        with self.assertRaises(ValueError):
            save_taxonomy(self.G, "x.bogus", fmt="bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
