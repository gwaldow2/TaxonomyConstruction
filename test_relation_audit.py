"""Self-checks for relation_type_audit. No GPU, no server, no LLM calls.

    python test_relation_audit.py
"""

import os
import csv
import shutil
import tempfile
from collections import Counter

import networkx as nx

from relation_type_audit import (CANONICAL_TYPES, parse_type, stratified_sample,
                                 write_isa_only_gt, rescore, load_pred_edges)


def test_canonical_types_unique():
    assert len(set(CANONICAL_TYPES)) == len(CANONICAL_TYPES)
    assert CANONICAL_TYPES[0] == "is_a"


def test_parse_type_last_label_wins():
    assert parse_type("child is a kind of parent\nis_a") == "is_a"
    assert parse_type("Not is_a; it is really part_of") == "part_of"
    assert parse_type("considered is_a and part_of, final: has_part") == "has_part"


def test_parse_type_substring_trap():
    """'is_a' is a substring of 'is_a_inverted'; a naive last-occurrence scan reports the wrong
    one for every inverted edge, which is exactly the error class the label exists to surface."""
    assert parse_type("is_a_inverted") == "is_a_inverted"
    assert parse_type("The direction is backwards.\nis_a_inverted") == "is_a_inverted"
    assert parse_type("final: Is_A_Inverted") == "is_a_inverted"


def test_parse_type_normalises_and_defaults():
    assert parse_type("answer: IS-A") == "is_a"          # hyphen + case
    assert parse_type("part-of") == "part_of"
    assert parse_type("no idea") == "unrelated"
    assert parse_type("") == "unrelated"
    assert parse_type(None) == "unrelated"


EDGES = [(f"p{i}", f"c{i}", 1 if i % 10 < 3 else 0) for i in range(300)]   # 30% FP


def test_stratified_sample_preserves_ratio():
    s = stratified_sample(EDGES, 60, 42)
    assert len(s) == 60
    got = Counter(e[2] for e in s)
    assert abs(got[1] / len(s) - 0.3) < 0.05, got


def test_stratified_sample_seeded():
    assert stratified_sample(EDGES, 60, 42) == stratified_sample(EDGES, 60, 42)
    assert stratified_sample(EDGES, 60, 7) != stratified_sample(EDGES, 60, 42)


def test_stratified_sample_no_op_and_pure():
    assert len(stratified_sample(EDGES, 0, 42)) == len(EDGES)
    assert len(stratified_sample(EDGES, 10 ** 6, 42)) == len(EDGES)
    assert len(EDGES) == 300      # input not mutated


def _fixture(tmp):
    """GT with 5 genuine is_a edges plus 2 that are really part_of, and matching predictions."""
    G = nx.DiGraph([("vehicle", "car"), ("vehicle", "truck"), ("car", "sedan"),
                    ("food", "fruit"), ("fruit", "apple"),
                    ("car", "wheel"), ("car", "engine")])
    nx.write_graphml(G, os.path.join(tmp, "GT_T_SUB_eval.graphml"))
    pred = [("vehicle", "car", 0), ("car", "sedan", 0), ("food", "fruit", 0),
            ("apple", "vehicle", 1), ("truck", "food", 1)]
    with open(os.path.join(tmp, "T_SUB_M_edge_diagnostics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "parent", "child", "leverage", "neighborhood_agreement",
                    "votes", "salience", "is_fp"])
        for p, c, fp in pred:
            w.writerow(["T_SUB", p, c, 0.1, 0.1, 2, 1, fp])
    rows = [{"dataset": "T_SUB", "source": "gt", "parent": p, "child": c,
             "type": "part_of" if (p, c) in {("car", "wheel"), ("car", "engine")} else "is_a"}
            for p, c in G.edges()]
    return G, pred, rows


def test_isa_only_gt_drops_exactly_the_non_isa():
    tmp = tempfile.mkdtemp()
    try:
        G, _, rows = _fixture(tmp)
        _, kept, dropped = write_isa_only_gt(tmp, "T_SUB", rows)
        out = nx.DiGraph(nx.read_graphml(os.path.join(tmp, "GT_T_SUB_isa_only.graphml")))
        assert (kept, dropped) == (5, 2)
        assert set(out.edges()) == set(G.edges()) - {("car", "wheel"), ("car", "engine")}
        assert set(out.nodes()) == set(G.nodes()), "nodes must survive edge removal"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_keeping_every_type_is_a_no_op():
    """The identity case: if nothing is filtered, the graph and the score must be unchanged."""
    tmp = tempfile.mkdtemp()
    try:
        G, pred, rows = _fixture(tmp)
        write_isa_only_gt(tmp, "T_SUB", rows, keep_types=tuple(CANONICAL_TYPES))
        out = nx.DiGraph(nx.read_graphml(os.path.join(tmp, "GT_T_SUB_isa_only.graphml")))
        assert set(out.edges()) == set(G.edges())
        d = rescore(tmp, "T_SUB", pred)
        assert abs(d["dF1"]) < 1e-12, d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unclassified_gt_edges_are_kept():
    """An edge nobody audited is not evidence of contamination, so it must survive."""
    tmp = tempfile.mkdtemp()
    try:
        G, _, _ = _fixture(tmp)
        _, kept, dropped = write_isa_only_gt(tmp, "T_SUB", [])      # no classifications at all
        assert (kept, dropped) == (G.number_of_edges(), 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_pred_edges_reads_is_fp():
    tmp = tempfile.mkdtemp()
    try:
        _fixture(tmp)
        rows, path = load_pred_edges(tmp, "T_SUB")
        assert path and len(rows) == 5
        assert Counter(r[2] for r in rows) == Counter({0: 3, 1: 2})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll relation_type_audit checks passed.")
