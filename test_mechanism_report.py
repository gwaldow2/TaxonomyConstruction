"""Self-checks for mechanism_report and the matched-semantics single call. No GPU, no LLM.

    python test_mechanism_report.py
"""

import os
import json
import shutil
import tempfile

from mechanism_report import load_results, pick_row, extract_k, collect_ksweep
from single_call_method import build_matched_prompt, build_single_call_prompt


def _write(tmp, name, blocks):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(blocks, f)
    return p


def _block(ds, methods):
    return {"dataset": ds, "use_synsets": False, "explode_nodes": False,
            "results": [{"method": m, "Cond_Clos_F1": v} for m, v in methods.items()]}


def test_pick_row_disambiguates_labels():
    rows = {
        "Our Method (k=1000, clawback=0)": {"Cond_Clos_F1": 0.7},
        "Our Method +restructure (k=1000, clawback=0)": {"Cond_Clos_F1": 0.1},
        "Our Method [oneway] (k=1000, clawback=0)": {"Cond_Clos_F1": 0.2},
        "Single-Call Baseline": {"Cond_Clos_F1": 0.6},
        "Single-Call Baseline [matched]": {"Cond_Clos_F1": 0.65},
        "Single-Call Baseline (no-merge)": {"Cond_Clos_F1": 0.59},
    }
    assert pick_row(rows, "structured")["Cond_Clos_F1"] == 0.7      # tags excluded
    assert pick_row(rows, "singlecall")["Cond_Clos_F1"] == 0.6      # shortest plain label wins
    assert pick_row(rows, "matched")["Cond_Clos_F1"] == 0.65
    assert pick_row({"Lexical": {}}, "structured") is None


def test_extract_k():
    assert extract_k("Our Method (k=1000, clawback=0)") == 1000
    assert extract_k("Our Method (k=25, clawback=0)") == 25
    assert extract_k("Single-Call Baseline") is None


def test_collect_ksweep_merges_files_and_skips_tagged_runs():
    tmp = tempfile.mkdtemp()
    try:
        a = _write(tmp, "a.json", [_block("D1_SUB", {
            "Our Method (k=25, clawback=0)": 0.60,
            "Our Method (k=50, clawback=0)": 0.63,
            "Our Method +restructure (k=25, clawback=0)": 0.10,   # tagged: excluded
        })])
        b = _write(tmp, "b.json", [_block("D1_SUB", {
            "Our Method (k=1000, clawback=0)": 0.66})])
        table = collect_ksweep([load_results(a), load_results(b)])
        assert table["D1_SUB"] == {25: 0.60, 50: 0.63, 1000: 0.66}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_results_shape():
    tmp = tempfile.mkdtemp()
    try:
        p = _write(tmp, "r.json", [_block("D1_SUB", {"Single-Call Baseline": 0.5}),
                                   _block("D2_SUB", {"Single-Call Baseline": 0.4})])
        m = load_results(p)
        assert set(m) == {"D1_SUB", "D2_SUB"}
        assert m["D1_SUB"]["Single-Call Baseline"]["Cond_Clos_F1"] == 0.5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_matched_prompt_isolates_call_structure():
    terms = ["apple", "food", "fruit"]
    matched = build_matched_prompt(terms)
    best = build_single_call_prompt(terms)
    # Same semantics as the structured 'full' variant: extensional rule, '<=', same example.
    assert "could logically also be labeled" in matched
    assert "'anucleate cell' <= 'cell'" in matched
    assert matched.rstrip().endswith("Relationships:")
    for t in terms:
        assert f"- {t}" in matched
    # And NONE of the best-practice bundle: no few-shot, no JSON, no direct-only rule.
    assert "EXAMPLE 1" not in matched and "JSON" not in matched and "DIRECT" not in matched
    # while the best-practice prompt keeps its own identity
    assert "EXAMPLE 1" in best and "JSON" in best


def test_matched_style_parses_le_lines():
    from single_call_method import method_single_call
    from test_single_call import _Client
    nodes = ["food", "fruit", "apple"]
    c = _Client("apple <= fruit\nfruit <= food")
    G = method_single_call(nodes, c, "m", merge_synonyms=False, style="matched")
    assert set(G.edges()) == {("fruit", "apple"), ("food", "fruit")}, set(G.edges())
    assert "Entities:" in c.last_prompt and "EXAMPLE 1" not in c.last_prompt


def test_bestpractice_style_unchanged():
    from single_call_method import method_single_call
    from test_single_call import _Client
    nodes = ["food", "fruit"]
    c = _Client('[["food", "fruit"]]')
    G = method_single_call(nodes, c, "m", merge_synonyms=False)
    assert set(G.edges()) == {("food", "fruit")}
    assert "NOW THE REAL TASK" in c.last_prompt


def test_density_prompt_holds_coverage_fixed():
    from density_sweep import build_density_prompt, make_groups
    terms = sorted(["apple", "food", "fruit", "grain", "bread"])
    p1 = build_density_prompt(terms[:1], terms)
    p3 = build_density_prompt(terms[:3], terms)
    # the ENTITY list is the full vocabulary at every m -- coverage never shrinks
    for t in terms:
        assert f"- {t}" in p1 and f"- {t}" in p3
    # structured-'full' semantics survive the generalization
    for p in (p1, p3):
        assert "could logically also be labeled" in p
        assert "'anucleate cell' <= 'cell'" in p
        assert p.rstrip().endswith("Relationships:")
    assert "EXAMPLE 1" not in p3 and "JSON" not in p3


def test_density_groups_cover_each_term_once():
    from density_sweep import make_groups
    terms = [f"t{i}" for i in range(10)]
    for m in (1, 3, 10, 25):
        groups = make_groups(terms, m)
        flat = [t for g in groups for t in g]
        assert sorted(flat) == sorted(terms)                  # exactly once each
        assert all(len(g) <= m for g in groups)
    assert make_groups(terms, 3) == make_groups(terms, 3)     # deterministic


def test_density_collection_and_knee():
    from mechanism_report import collect_density, find_knee
    tmp = tempfile.mkdtemp()
    try:
        blocks = [_block(ds, {f"Our Method [density m={m}]": v[m]
                              for m in v})
                  for ds, v in [("D1_SUB", {1: 0.70, 5: 0.69, 25: 0.60}),
                                ("D2_SUB", {1: 0.60, 5: 0.61, 25: 0.48})]]
        p = _write(tmp, "d.json", blocks)
        from mechanism_report import load_results
        table = collect_density([load_results(p)])
        assert table["D1_SUB"][25]["Cond_Clos_F1"] == 0.60
        mean_by_m = {m: sum(table[ds][m]["Cond_Clos_F1"] for ds in table) / 2
                     for m in (1, 5, 25)}
        knee, degraded = find_knee(mean_by_m, tol=0.02)
        assert knee == 5 and degraded == 25                   # holds at 5, breaks at 25
        knee, degraded = find_knee({1: 0.7, 5: 0.71, 25: 0.69}, tol=0.02)
        assert knee == 25 and degraded is None                # flat sweep: no knee found
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll mechanism_report checks passed.")
