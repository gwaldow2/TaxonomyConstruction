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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll mechanism_report checks passed.")
