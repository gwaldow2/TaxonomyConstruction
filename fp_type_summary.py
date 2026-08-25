"""What relation types are the FALSE POSITIVES, pooled and per dataset.

Aggregates the predicted-edge rows of the relation-type audit down to just the errors: rows with
source == "pred" and is_fp == 1. It calls no model -- the types and justifications were already
produced by relation_type_audit.py -- so this is free to re-run.

The question it answers is why the method's wrong edges are wrong: an FP labelled part_of is a
real relation misfiled as is-a, one labelled is_a_inverted is a direction error, and one labelled
unrelated is a plain mistake. Those three call for completely different fixes, and the headline
precision number cannot tell them apart.

    python fp_type_summary.py
    python fp_type_summary.py --types_csv results/relation_types_gemma.csv \
                              --types_csv results/relation_types_gemini.csv

--types_csv is repeatable so a second judge can be folded in later, matching isa_rescore.py; rows
are concatenated and the judge is kept in the `model` column rather than merged away.

Writes results/fp_relation_types.csv (detail) and results/fp_relation_types_summary.csv (counts,
proportions and example justifications, POOLED plus per dataset).
"""

import os
import csv
import argparse
from collections import Counter, defaultdict

from relation_type_audit import CANONICAL_TYPES, RESULTS_DIR

POOLED = "POOLED"


def load_fp_rows(paths):
    """-> predicted-edge rows that were scored as false positives, across all judge CSVs."""
    rows, seen_any = [], False
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"[!] {p} not found -- run relation_type_audit.py first.")
        with open(p, newline="", encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        seen_any = seen_any or bool(recs)
        keep = [r for r in recs if r.get("source") == "pred" and str(r.get("is_fp")) == "1"]
        n_pred = sum(1 for r in recs if r.get("source") == "pred")
        print(f"    {p}: {len(recs)} rows | {n_pred} predicted | {len(keep)} false positives")
        rows += keep
    if not seen_any:
        raise SystemExit("[!] the type CSV(s) contained no rows at all.")
    return rows


def type_order(counter):
    """Canonical order first (so tables line up with the audit), then anything unrecognised."""
    known = [t for t in CANONICAL_TYPES if counter.get(t)]
    extra = sorted(t for t in counter if t not in CANONICAL_TYPES)
    return known + extra


def pick_justifications(rows, n, spread_across_datasets):
    """Up to n example justifications for one relation type.

    For the pooled scope, prefer examples from DISTINCT datasets: otherwise all three tend to come
    from whichever dataset contributed the most FPs, which reads as a domain quirk rather than a
    general pattern.
    """
    out, used_ds = [], set()
    if spread_across_datasets:
        for r in rows:
            if len(out) >= n:
                break
            if r["dataset"] not in used_ds and r.get("justification", "").strip():
                out.append(r)
                used_ds.add(r["dataset"])
    for r in rows:                       # top up in file order if not enough distinct datasets
        if len(out) >= n:
            break
        if r not in out and r.get("justification", "").strip():
            out.append(r)
    return out


def summarize(rows, n_just):
    """-> (summary_lines, {scope: Counter}) for POOLED and each dataset."""
    scopes = [(POOLED, rows)]
    for ds in sorted({r["dataset"] for r in rows}):
        scopes.append((ds, [r for r in rows if r["dataset"] == ds]))

    lines, counters = [], {}
    for scope, subset in scopes:
        c = Counter(r["type"] for r in subset)
        counters[scope] = c
        total = sum(c.values())
        for t in type_order(c):
            examples = pick_justifications([r for r in subset if r["type"] == t],
                                           n_just, spread_across_datasets=(scope == POOLED))
            line = {"scope": scope, "type": t, "count": c[t], "total_fps": total,
                    "proportion": round(c[t] / total, 4) if total else 0.0}
            for i in range(n_just):
                r = examples[i] if i < len(examples) else None
                line[f"justification_{i+1}"] = (
                    f"[{r['dataset']}] {r['child']} under {r['parent']}: {r['justification']}"
                    if r else "")
            lines.append(line)
    return lines, counters


def print_report(lines, counters, n_just):
    for scope in [POOLED] + sorted(s for s in counters if s != POOLED):
        c = counters[scope]
        total = sum(c.values())
        head = "POOLED across datasets" if scope == POOLED else scope
        print("\n" + "=" * 78)
        print(f"### {head}   ({total} false positives)")
        print("=" * 78)
        for line in [l for l in lines if l["scope"] == scope]:
            print(f"\n  {line['type']:16s} {line['count']:5d}   {line['proportion']:.1%} of FPs")
            for i in range(1, n_just + 1):
                j = line.get(f"justification_{i}", "")
                if j:
                    print(f"      {i}. {j[:150]}")


def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Relation types of the false positives, pooled and per dataset.")
    ap.add_argument("--types_csv", action="append", default=None,
                    help="Relation-type CSV(s) from relation_type_audit.py. Repeatable. "
                         "Default: results/relation_types.csv")
    ap.add_argument("--n_justifications", type=int, default=3)
    ap.add_argument("--out_csv", default=os.path.join(RESULTS_DIR, "fp_relation_types.csv"))
    ap.add_argument("--summary_csv", default=os.path.join(RESULTS_DIR, "fp_relation_types_summary.csv"))
    args = ap.parse_args()

    print("[*] loading relation-type labels:")
    rows = load_fp_rows(args.types_csv or [os.path.join(RESULTS_DIR, "relation_types.csv")])
    if not rows:
        raise SystemExit("[!] no false-positive predicted edges found. Either the audit covered "
                         "only GT edges (--max_pred_per_dataset 0 with no diagnostics), or every "
                         "sampled prediction was correct.")

    judges = sorted({r.get("model", "") for r in rows if r.get("model")})
    print(f"[*] {len(rows)} false-positive edges | judge(s): {', '.join(judges) or 'unknown'}")

    lines, counters = summarize(rows, args.n_justifications)
    print_report(lines, counters, args.n_justifications)

    write_csv(args.out_csv, rows,
              ["dataset", "parent", "child", "type", "model", "justification"])
    write_csv(args.summary_csv, lines,
              ["scope", "type", "count", "total_fps", "proportion"]
              + [f"justification_{i+1}" for i in range(args.n_justifications)])
    print(f"\n[*] wrote {args.out_csv} and {args.summary_csv}")


if __name__ == "__main__":
    main()
