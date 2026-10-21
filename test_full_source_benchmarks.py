"""Offline tests for the full-source GT rebuild (no network, no LLM)."""

import json
import os

import networkx as nx
import pytest

import data_manager as dm
import full_source_benchmarks as fsb
from data_manager import load_benchmark_graph, save_benchmark_graph

JSONLD = {"@graph": [
    {"@id": "schema:Thing", "@type": "rdfs:Class"},
    {"@id": "schema:CreativeWork", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Thing"}},
    {"@id": "schema:Game", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:CreativeWork"}},
    {"@id": "schema:Review", "@type": ["rdfs:Class"],
     "rdfs:subClassOf": {"@id": "schema:CreativeWork"}},
    {"@id": "schema:Intangible", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Thing"}},
    {"@id": "schema:Offer", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Intangible"}},
    {"@id": "schema:Enumeration", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Intangible"}},
    {"@id": "schema:Place", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Thing"}},
    {"@id": "schema:Accommodation", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Place"}},
    {"@id": "schema:Apartment", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Accommodation"}},
    # a property (not a class) and a foreign-namespace node: both ignored
    {"@id": "schema:name", "@type": "rdf:Property"},
    {"@id": "http://example.org/Alien", "@type": "rdfs:Class",
     "rdfs:subClassOf": {"@id": "schema:Thing"}},
    # the punned DataType subtree: dual typing stands in for is_a
    {"@id": "schema:DataType", "@type": "rdfs:Class"},
    {"@id": "schema:Text", "@type": ["rdfs:Class", "schema:DataType"]},
]}

OBO = """format-version: 1.2

[Term]
id: X:1
name: Widget

[Term]
id: X:2
name: Blue Widget
is_a: X:1 ! Widget
is_a: UBERON:0001062

[Term]
id: X:3
name: Dark Blue Widget
is_a: X:2 ! Blue Widget

[Term]
id: X:4
name: OBSOLETE Old Widget
is_obsolete: true
"""


def test_parse_schemaorg_jsonld(tmp_path):
    p = tmp_path / "s.jsonld"
    p.write_text(json.dumps(JSONLD), encoding="utf-8")
    G = fsb.parse_schemaorg_jsonld(str(p))
    assert ("creativework", "game") in G.edges()
    assert ("intangible", "offer") in G.edges()
    assert ("thing", "place") in G.edges()
    assert "alien" not in G and "name" not in G          # non-schema / non-class ignored
    assert ("datatype", "text") in G.edges()             # punned DataType subtree kept
    assert G.number_of_nodes() == 12


def test_parse_obo(tmp_path):
    p = tmp_path / "t.obo"
    p.write_text(OBO, encoding="utf-8")
    G = fsb.parse_obo(str(p))
    # the dangling external ref (UBERON:0001062, no stanza in this file) and the
    # obsolete term are both excised; only defined, named terms survive
    assert set(G.edges()) == {("widget", "blue widget"), ("blue widget", "dark blue widget")}
    assert all("uberon" not in n and "obsolete" not in n for n in G.nodes())


def test_parse_ttl(tmp_path):
    pytest.importorskip("rdflib")
    ttl = """@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://matonto.org/ontology#> .
ex:Material rdfs:label "material" .
ex:Ceramic rdfs:label "ceramic" ; rdfs:subClassOf ex:Material .
ex:Oxide rdfs:subClassOf ex:Ceramic .
ex:Alloy rdfs:label "alloy" ; rdfs:subClassOf ex:Material ;
    owl:equivalentClass [ owl:unionOf ( ex:Bronze ex:Steel ) ] .
ex:Bronze rdfs:label "bronze" .
ex:Steel rdfs:label "steel" .
ex:Metal rdfs:label "metal" .
ex:Cermet rdfs:label "cermet" ;
    rdfs:subClassOf [ owl:intersectionOf ( ex:Ceramic ex:Metal ) ] .
ex:Halide rdfs:label "halide" ; rdfs:subClassOf ex:Material ;
    owl:equivalentClass ex:Salt .
ex:Salt rdfs:label "salt" .
"""
    p = tmp_path / "m.ttl"
    p.write_text(ttl, encoding="utf-8")
    G = fsb.parse_ttl(str(p))
    assert ("material", "ceramic") in G.edges()
    assert ("ceramic", "oxide") in G.edges()             # label missing -> local name
    # union equivalence entails membership: Alloy = Bronze or Steel
    assert ("alloy", "bronze") in G.edges()
    assert ("alloy", "steel") in G.edges()
    # subClassOf an intersection entails subsumption by each named component
    assert ("ceramic", "cermet") in G.edges()
    assert ("metal", "cermet") in G.edges()
    # named<->named equivalence merges into a synset node
    assert ("material", "halide (halide, salt)") in G.edges()
    assert "halide" not in G and "salt" not in G


def test_parse_ttl_erratum(tmp_path):
    pytest.importorskip("rdflib")
    ttl = """@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://matonto.org/ontology#> .
ex:G2 rdfs:label "group 2 element" ;
    owl:equivalentClass [ owl:unionOf ( ex:Act ex:Lan ex:Sc ex:Y ) ] ;
    owl:equivalentClass [ owl:unionOf ( ex:Ba ex:Be ) ] .
ex:Act rdfs:label "actinide" . ex:Lan rdfs:label "lanthanide" .
ex:Sc rdfs:label "scandium" .  ex:Y rdfs:label "yttrium" .
ex:Ba rdfs:label "barium" .    ex:Be rdfs:label "beryllium" .
"""
    p = tmp_path / "err.ttl"
    p.write_text(ttl, encoding="utf-8")
    G = fsb.parse_ttl(str(p))
    # the documented MatOnto erratum (group 3's membership on group 2) is skipped
    assert ("group 2 element", "actinide") not in G.edges()
    assert ("group 2 element", "barium") in G.edges()


def test_build_one_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(dm, "BENCHMARK_DIR", str(tmp_path / "bench"))
    os.makedirs(dm.BENCHMARK_DIR, exist_ok=True)
    # pre-place the pinned source so no network fetch happens
    os.makedirs(os.path.join(dm.DATA_DIR, "full_sources"), exist_ok=True)
    src = os.path.join(dm.DATA_DIR, "full_sources", "schemaorg-current-https.jsonld")
    open(src, "w", encoding="utf-8").write(json.dumps(JSONLD))
    # a legacy benchmark that must remain untouched
    save_benchmark_graph(nx.DiGraph([("x", "y")]), "LLMs4OL_SchemaOrg", scale="SUB")
    legacy = open(os.path.join(dm.BENCHMARK_DIR, "LLMs4OL_SchemaOrg_SUB.graphml"), "rb").read()

    assert fsb.build_one("SchemaOrgV2", fsb.SOURCES["SchemaOrgV2"], target_nodes=5) == "SchemaOrgV2"

    G_full, pairs = load_benchmark_graph("SchemaOrgV2", scale="FULL")
    assert G_full is not None and nx.is_directed_acyclic_graph(G_full)
    assert G_full.graph["source_url"].startswith("https://schema.org")
    assert len(G_full.graph["source_sha256"]) == 64
    assert G_full.graph["generator"] == "full_source_benchmarks.py"

    G_sub, _ = load_benchmark_graph("SchemaOrgV2", scale="SUB")
    assert G_sub is not None and G_sub.number_of_nodes() >= 5
    assert G_sub.graph["seed"] == "42"

    manifest = open(os.path.join(dm.DATA_DIR, "full_sources", "sources_manifest.csv"),
                    encoding="utf-8").read()
    assert "SchemaOrgV2" in manifest and "cached" in manifest

    # legacy untouched; second build skips without --force
    assert open(os.path.join(dm.BENCHMARK_DIR, "LLMs4OL_SchemaOrg_SUB.graphml"), "rb").read() == legacy
    assert fsb.build_one("SchemaOrgV2", fsb.SOURCES["SchemaOrgV2"], target_nodes=5) is None
