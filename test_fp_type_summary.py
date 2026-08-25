"""Self-checks for fp_type_summary. No GPU, no server, no LLM calls.

    python test_fp_type_summary.py
"""

import os
import csv
import tempfile
import shutil

from fp_type_summary import POOLED, load_fp_rows, summarize, type_order, pick_justifications

COLS = ["dataset", "source", "parent", "child", "is_fp", "type", "model", "justification"]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp):
    """Two datasets. GT rows and predicted TRUE positives are present as decoys and must be
    excluded; only predicted rows with is_fp == 1 may be counted."""
    r = lambda ds, src, p, c, fp, t, j: dict(dataset=ds, source=src, parent=p, child=c,
                                             is_fp=fp, type=t, model="judge", justification=j)
    rows = [
        # decoys: GT rows (is_fp blank) and a correct prediction
        r("A_SUB", "gt",   "food", "fruit",  "",  "is_a",    "gt row"),
        r("A_SUB", "gt",   "car",  "wheel",  "",  "part_of", "gt row"),
        r("A_SUB", "pred", "food", "fruit",  "0", "is_a",    "true positive"),
        # A_SUB false positives: 3 part_of, 1 unrelated
        r("A_SUB", "pred", "car",  "wheel",  "1", "part_of", "wheel is a component of a car"),
        r("A_SUB", "pred", "car",  "engine", "1", "part_of", "engine is a component"),
        r("A_SUB", "pred", "car",  "door",   "1", "part_of", "door is a component"),
        r("A_SUB", "pred", "sky",  "tuna",   "1", "unrelated", "no relation at all"),
        # B_SUB false positives: 1 part_of, 1 is_a_inverted
        r("B_SUB", "pred", "cell", "nucleus", "1", "part_of", "nucleus is inside the cell"),
        r("B_SUB", "pred", "dog",  "animal",  "1", "is_a_inverted", "backwards"),
    ]
    p = os.path.join(tmp, "rt.csv")
    _write(p, rows)
    return p


def test_only_fp_predictions_are_counted():
    tmp = tempfile.mkdtemp()
    try:
        rows = load_fp_rows([_fixture(tmp)])
        assert len(rows) == 6, len(rows)
        assert all(r["source"] == "pred" and r["is_fp"] == "1" for r in rows)
        assert not any(r["justification"] in ("gt row", "true positive") for r in rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pooled_counts_equal_sum_of_datasets():
    tmp = tempfile.mkdtemp()
    try:
        rows = load_fp_rows([_fixture(tmp)])
        _, counters = summarize(rows, 3)
        pooled = counters[POOLED]
        assert sum(pooled.values()) == 6
        assert pooled["part_of"] == 4 and pooled["unrelated"] == 1 and pooled["is_a_inverted"] == 1
        per = sum(sum(c.values()) for s, c in counters.items() if s != POOLED)
        assert per == sum(pooled.values()), (per, sum(pooled.values()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_proportions_sum_to_one_per_scope():
    """Proportions are stored rounded to 4dp for readability, so the sum can drift by up to
    half a unit in the last place per type (e.g. 4/6 + 1/6 + 1/6 -> 1.0001). The tolerance
    allows exactly that much and no more, so a real normalisation bug would still fail."""
    tmp = tempfile.mkdtemp()
    try:
        lines, counters = summarize(load_fp_rows([_fixture(tmp)]), 3)
        for scope in counters:
            rows = [l for l in lines if l["scope"] == scope]
            tot = sum(l["proportion"] for l in rows)
            assert abs(tot - 1.0) <= 0.00005 * len(rows) + 1e-9, (scope, tot)
            # and the raw counts, which carry no rounding, must be exact
            assert sum(l["count"] for l in rows) == rows[0]["total_fps"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pooled_justifications_span_datasets():
    """part_of occurs in both datasets; the pooled examples must not all come from A_SUB."""
    tmp = tempfile.mkdtemp()
    try:
        lines, _ = summarize(load_fp_rows([_fixture(tmp)]), 3)
        row = next(l for l in lines if l["scope"] == POOLED and l["type"] == "part_of")
        js = [row[f"justification_{i}"] for i in range(1, 4) if row[f"justification_{i}"]]
        assert len(js) == 3, js
        assert any(j.startswith("[B_SUB]") for j in js), js
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fewer_than_n_justifications_is_not_padded():
    tmp = tempfile.mkdtemp()
    try:
        lines, _ = summarize(load_fp_rows([_fixture(tmp)]), 3)
        row = next(l for l in lines if l["scope"] == POOLED and l["type"] == "unrelated")
        assert row["justification_1"] and not row["justification_2"] and not row["justification_3"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_type_order_is_canonical_then_unknown():
    from collections import Counter
    order = type_order(Counter({"zzz_custom": 1, "part_of": 2, "is_a": 1}))
    assert order == ["is_a", "part_of", "zzz_custom"], order


def test_pick_justifications_skips_blank():
    rows = [{"dataset": "A", "justification": ""}, {"dataset": "A", "justification": "real"}]
    got = pick_justifications(rows, 2, spread_across_datasets=False)
    assert len(got) == 1 and got[0]["justification"] == "real"


def test_missing_file_is_refused():
    try:
        load_fp_rows(["definitely/not/here.csv"])
    except SystemExit as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("missing file should raise SystemExit")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll fp_type_summary checks passed.")
