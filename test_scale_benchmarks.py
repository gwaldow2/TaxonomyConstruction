"""Tests for the scale-sweep benchmark generation (scale_benchmarks.py) and the
scale-token plumbing in data_manager. Offline: no LLM, no network, no GPU."""

import csv
import os

import networkx as nx
import pytest

import data_manager as dm
from data_manager import (scale_token, parse_scale_token, is_valid_scale,
                          get_closed_subgraph, save_benchmark_graph, load_benchmark_graph)
import scale_benchmarks as sb


# ---------- scale tokens ----------

def test_scale_token_roundtrip():
    assert scale_token(200) == "SUB200"
    assert scale_token(25, 7) == "SUB25S7"
    assert scale_token(25, 42) == "SUB25"          # default seed elided
    assert parse_scale_token("SUB200") == (200, 42)
    assert parse_scale_token("sub25s7") == (25, 7)  # case-insensitive
    for n in (25, 50, 100, 200, 400):
        for seed in (42, 1, 7):
            assert parse_scale_token(scale_token(n, seed)) == (n, seed)


def test_legacy_tokens_parse_as_none_but_stay_valid():
    assert parse_scale_token("SUB") is None
    assert parse_scale_token("FULL") is None
    assert is_valid_scale("sub") and is_valid_scale("FULL") and is_valid_scale("sub200")
    assert is_valid_scale("sub25s7")
    assert not is_valid_scale("sub200x") and not is_valid_scale("medium") and not is_valid_scale("")


# ---------- seeded subsampling ----------

def chains_graph(n_chains=40, depth=3):
    """A root with n_chains disjoint chains below it: sampling different leaves
    pulls in different chains, so seeds are distinguishable."""
    G = nx.DiGraph()
    for i in range(n_chains):
        prev = "root"
        for d in range(depth):
            node = f"c{i}_{d}"
            G.add_edge(prev, node)
            prev = node
    return G


def test_subgraph_seed_determinism_and_difference():
    G = chains_graph()
    a1 = set(get_closed_subgraph(G, target_nodes=10, seed=1).nodes())
    a2 = set(get_closed_subgraph(G, target_nodes=10, seed=1).nodes())
    b = set(get_closed_subgraph(G, target_nodes=10, seed=2).nodes())
    assert a1 == a2                       # deterministic per seed
    assert a1 != b                        # different seeds sample different nodes
    assert len(a1) >= 10                  # ancestor closure may overshoot, never undershoot


def test_same_seed_scales_are_nested():
    G = chains_graph()
    small = set(get_closed_subgraph(G, target_nodes=10, seed=42).nodes())
    large = set(get_closed_subgraph(G, target_nodes=25, seed=42).nodes())
    assert small <= large


def test_default_seed_matches_legacy_call():
    """The bare call (as taxonomy_metrics.py makes it) must reproduce seed=42
    exactly, so the frozen SUB benchmarks remain reproducible."""
    G = chains_graph()
    assert set(get_closed_subgraph(G, target_nodes=10).nodes()) == \
           set(get_closed_subgraph(G, target_nodes=10, seed=42).nodes())


# ---------- provenance stamping ----------

def test_saved_graph_carries_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "BENCHMARK_DIR", str(tmp_path))
    G = nx.DiGraph([("a", "b")])
    save_benchmark_graph(G, "Toy", scale="SUB5", meta={"scale": "SUB5", "seed": 7, "target_nodes": 5})
    G2, _ = load_benchmark_graph("Toy", scale="SUB5")
    assert G2.graph["scale"] == "SUB5"
    assert G2.graph["seed"] == "7"
    assert G2.graph["target_nodes"] == "5"
    assert set(G2.edges()) == {("a", "b")}


def test_strip_virtual_root():
    G = nx.DiGraph([("virtual_root", "a"), ("a", "b")])
    G.nodes["virtual_root"]["is_virtual"] = True
    H = sb.strip_virtual_root(G)
    assert "virtual_root" not in H
    assert set(H.edges()) == {("a", "b")}
    # graphml round-trips booleans as strings; the string form must strip too
    G2 = nx.DiGraph([("vr", "a")])
    G2.nodes["vr"]["is_virtual"] = "True"
    assert "vr" not in sb.strip_virtual_root(G2)


# ---------- end-to-end generation ----------

@pytest.fixture
def bench_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "BENCHMARK_DIR", str(tmp_path))
    G_full = sb.strip_virtual_root(chains_graph(n_chains=20, depth=3))  # 61 nodes
    G_full.add_node("virtual_root", is_virtual=True)
    G_full.add_edge("virtual_root", "root")
    pairs = [{"parent": f"p{i}", "child": f"k{i}"} for i in range(30)]
    save_benchmark_graph(G_full, "Toy", scale="FULL", train_pairs=pairs)
    # a pre-existing legacy SUB file that generation must never touch
    save_benchmark_graph(nx.DiGraph([("x", "y")]), "Toy", scale="SUB")
    return tmp_path


def test_generate_one_end_to_end(bench_dir):
    legacy_sub = os.path.join(str(bench_dir), "Toy_SUB.graphml")
    legacy_before = open(legacy_sub, "rb").read()

    row = sb.generate_one("Toy", target_nodes=10, seed=42)
    assert row is not None and row["scale"] == "SUB10"
    assert row["nodes"] >= 10

    # the generated file exists, is a DAG, and carries its provenance
    G, train = load_benchmark_graph("Toy", scale="SUB10")
    assert nx.is_directed_acyclic_graph(G)
    assert G.graph["scale"] == "SUB10" and G.graph["seed"] == "42"
    assert G.graph["source_file"] == "Toy_FULL.graphml"
    assert len(train) == 5                                  # target_nodes // 2 slice

    # manifest row written and upserted (regeneration must not duplicate it)
    with open(sb.manifest_path(), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["dataset"], r["scale"]) for r in rows] == [("Toy", "SUB10")]

    # legacy artifacts untouched
    assert open(legacy_sub, "rb").read() == legacy_before
    full_g, _ = load_benchmark_graph("Toy", scale="FULL")
    assert "virtual_root" in full_g


def test_generate_skips_existing_without_force(bench_dir):
    assert sb.generate_one("Toy", 10, 42) is not None
    assert sb.generate_one("Toy", 10, 42) is None            # skip: exists
    assert sb.generate_one("Toy", 10, 42, force=True) is not None
    with open(sb.manifest_path(), newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1             # still one manifest row


def test_generate_skips_saturated_target(bench_dir):
    assert sb.generate_one("Toy", 500, 42) is None           # universe has 61 real nodes
    assert not os.path.exists(os.path.join(str(bench_dir), "Toy_SUB500.graphml"))


def test_generate_seeds_get_distinct_tokens(bench_dir):
    r1 = sb.generate_one("Toy", 10, 1)
    r2 = sb.generate_one("Toy", 10, 2)
    assert r1["scale"] == "SUB10S1" and r2["scale"] == "SUB10S2"
    g1, _ = load_benchmark_graph("Toy", scale="SUB10S1")
    g2, _ = load_benchmark_graph("Toy", scale="SUB10S2")
    assert set(g1.nodes()) != set(g2.nodes())

    with open(sb.manifest_path(), newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_discover_datasets(bench_dir):
    assert sb.discover_datasets() == ["Toy"]
