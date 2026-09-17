"""Self-checks for judge_agreement. No GPU, no server, no LLM calls.

    python test_judge_agreement.py
"""

import os
import csv
import shutil
import tempfile

from judge_agreement import (wilson, cohen_kappa, load_judged, join_judged,
                             consensus_nonisa, consensus_table, load_votes, votes_join)


def test_wilson_reproduces_paper_interval():
    """78 non-is-a of 692 audited GT edges -> the [9.1%, 13.8%] CI in the contamination table."""
    lo, hi = wilson(78, 692)
    assert abs(lo - 0.091) < 0.002 and abs(hi - 0.138) < 0.002, (lo, hi)


def test_wilson_bounds():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson(20, 20)
    assert 0.8 < lo < 1.0 and hi == 1.0


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([("a", "a"), ("b", "b")]) == (1.0, 1.0)
    # 50% raw agreement with 50/50 marginals on both sides is exactly chance level
    po, k = cohen_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")])
    assert po == 0.5 and abs(k) < 1e-12
    assert cohen_kappa([]) == (0.0, 0.0)


def _csv(tmp, name, rows):
    p = os.path.join(tmp, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "source", "parent", "child",
                                          "is_fp", "type", "model", "justification"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(ds, source, p, c, t, model, just="looks right", is_fp=""):
    return {"dataset": ds, "source": source, "parent": p, "child": c,
            "is_fp": is_fp, "type": t, "model": model, "justification": just}


def test_load_judged_skips_failed_rows_and_refuses_dead_files():
    tmp = tempfile.mkdtemp()
    try:
        p = _csv(tmp, "j.csv", [_row("D", "gt", "a", "b", "is_a", "m1"),
                                _row("D", "gt", "c", "d", "unrelated", "m1", just="")])
        m, model = load_judged(p)
        assert set(m) == {("D", "gt", "a", "b")} and model == "m1"
        dead = _csv(tmp, "dead.csv", [_row("D", "gt", "a", "b", "unrelated", "m2", just="")])
        try:
            load_judged(dead)
        except SystemExit:
            pass
        else:
            raise AssertionError("expected SystemExit on an all-failed CSV")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_join_and_consensus_rule():
    tmp = tempfile.mkdtemp()
    try:
        a = _csv(tmp, "a.csv", [_row("D", "gt", "a", "b", "is_a", "m1"),
                                _row("D", "gt", "c", "d", "part_of", "m1"),
                                _row("D", "gt", "e", "f", "part_of", "m1"),
                                _row("D", "gt", "g", "h", "is_a_inverted", "m1"),
                                _row("D", "gt", "only1", "x", "part_of", "m1")])
        b = _csv(tmp, "b.csv", [_row("D", "gt", "a", "b", "is_a", "m2"),
                                _row("D", "gt", "c", "d", "made_of", "m2"),
                                _row("D", "gt", "e", "f", "is_a", "m2"),
                                _row("D", "gt", "g", "h", "part_of", "m2")])
        (ma, _), (mb, _) = load_judged(a), load_judged(b)
        joined = join_judged([ma, mb])
        assert len(joined) == 4, "edges judged by only one judge must not join"
        # unanimity: split verdicts keep the edge; agreeing non-is-a labels need not match
        assert not consensus_nonisa(joined[("D", "gt", "a", "b")])        # both is_a
        assert consensus_nonisa(joined[("D", "gt", "c", "d")])            # part_of/made_of
        assert not consensus_nonisa(joined[("D", "gt", "e", "f")])        # split verdict
        assert consensus_nonisa(joined[("D", "gt", "g", "h")])            # inverted != is_a
        table = dict(consensus_table(joined))
        assert table["D"]["n"] == 4 and table["D"]["consensus"] == 2
        assert table["D"]["by_judge"] == [3, 2]
        assert table["POOLED"]["consensus"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_votes_join_buckets_fps_only():
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "D_Our_Method_edge_diagnostics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "parent", "child", "votes", "is_fp"])
            for p, c, v in [("a", "b", 2), ("c", "d", 1), ("e", "f", 1), ("g", "h", 2)]:
                w.writerow(["D", p, c, v, 1])
        assert load_votes(tmp, "D") == {("a", "b"): 2, ("c", "d"): 1,
                                        ("e", "f"): 1, ("g", "h"): 2}
        a = _csv(tmp, "a.csv", [_row("D", "pred", "a", "b", "is_a", "m1", is_fp="1"),
                                _row("D", "pred", "c", "d", "is_a", "m1", is_fp="1"),
                                _row("D", "pred", "e", "f", "part_of", "m1", is_fp="1"),
                                _row("D", "pred", "g", "h", "is_a", "m1", is_fp="0")])
        b = _csv(tmp, "b.csv", [_row("D", "pred", "a", "b", "is_a", "m2", is_fp="1"),
                                _row("D", "pred", "c", "d", "part_of", "m2", is_fp="1"),
                                _row("D", "pred", "e", "f", "part_of", "m2", is_fp="1"),
                                _row("D", "pred", "g", "h", "is_a", "m2", is_fp="0")])
        (ma, _), (mb, _) = load_judged(a), load_judged(b)
        joined = join_judged([ma, mb])
        vj = votes_join(joined, [ma, mb], tmp)
        assert set(vj) == {1, 2}, "the is_fp=0 row must be excluded, not bucketed"
        assert vj[2]["n"] == 1 and vj[2]["by_judge"] == [1.0, 1.0] and vj[2]["all_judges"] == 1.0
        assert vj[1]["n"] == 2 and vj[1]["by_judge"] == [0.5, 0.0] and vj[1]["all_judges"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll judge_agreement checks passed.")
