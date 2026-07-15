"""Why are the base-model false positives false? -- an LLM-classified breakdown per dataset.

For each false-positive edge in the raw Our Method run (the is_fp=1 rows of
<ds>_Our_Method_edge_diagnostics.csv), this shows the LLM (e.g. gemma via vLLM) the claimed
is-a plus where the two concepts ACTUALLY sit in the reference taxonomy, and asks it to
assign the single best reason the edge is a false positive:

  ontology_strictness | gt_missing | not_isa | wrong_parent | genuine_error | unclear

It then aggregates the distribution per dataset and overall, so you get a clear
"the majority of FPs are because of X". Per-edge labels + one-line justifications are saved
to a CSV for inspection, and a stacked-bar chart to vis/fp_reasons.png.

    python fp_reason_analysis.py --model google/gemma-4-31b-it --max_per_dataset 60

Needs the raw <ds>_Our_Method_edge_diagnostics.csv + GT_<ds>_eval.graphml in --results_dir,
and a vLLM/OpenAI server at --base_url serving --model.
"""

import os
import csv
import glob
import time
import random
import argparse
from collections import defaultdict

from prune_sweep_analysis import load_dataset, parse_lemma_format

BUCKETS = ["ontology_strictness", "gt_missing", "not_isa", "wrong_parent", "genuine_error", "unclear"]

_DESC = (
    "- ontology_strictness: reasonable in everyday/common-sense terms, but the reference taxonomy draws "
    "a stricter or more formal distinction that excludes it (e.g. element vs. atom, a property vs. the "
    "thing that has it, a unit vs. an object).\n"
    "- gt_missing: it IS a correct is-a relationship; the reference taxonomy is simply incomplete and "
    "should contain it (the builder is right, the reference is wrong by omission).\n"
    "- not_isa: the two concepts are related but NOT by is-a (part-of, made-of, has-property, "
    "instance-of, located-in); a different relation was mistaken for is-a.\n"
    "- wrong_parent: the child does have a genuine is-a parent, but THIS parent is the wrong one -- from "
    "a different branch, or far too broad/unrelated.\n"
    "- genuine_error: there is no sensible relationship between them; the prediction is just a mistake.\n"
    "- unclear: not enough information to decide.\n"
)


def _prim(node):
    return parse_lemma_format(node)[0]


def neighbor_maps(gt_pairs):
    """From the exploded GT-closure term-pairs: ancestors[t] and descendants[t] per term."""
    anc, desc = defaultdict(set), defaultdict(set)
    for a, b in gt_pairs:
        anc[b].add(a)
        desc[a].add(b)
    return anc, desc


def gt_context(parent, child, anc, desc, k=8):
    """What the child and parent are ACTUALLY related to in the reference taxonomy."""
    Pt, Ct = parse_lemma_format(parent), parse_lemma_format(child)

    def _union(m, terms):
        s = set()
        for t in terms:
            s |= m.get(t, set())
        return sorted(s)[:k]

    return {"child_is_a": _union(anc, Ct),
            "parent_is_a": _union(anc, Pt),
            "parent_contains": _union(desc, Pt)}


def build_prompt(parent, child, ctx):
    c, p = _prim(child), _prim(parent)

    def _fmt(lst):
        return ", ".join(f"'{x}'" for x in lst) if lst else "(none listed -- a top-level concept or absent from the reference)"

    return (
        "You are analyzing errors from an automatic taxonomy builder, scored against a reference "
        "(ground-truth) taxonomy.\n\n"
        f"The builder predicted this is-a relationship:  '{c}' is a kind of '{p}'.\n"
        f"The reference taxonomy does NOT contain this relationship (not even transitively), so it is "
        f"counted as a FALSE POSITIVE.\n\n"
        "Where they actually sit in the reference taxonomy:\n"
        f"- '{c}' is actually a kind of: {_fmt(ctx['child_is_a'])}.\n"
        f"- '{p}' is actually a kind of: {_fmt(ctx['parent_is_a'])}.\n"
        f"- '{p}' actually includes these kinds: {_fmt(ctx['parent_contains'])}.\n\n"
        "Choose the SINGLE best explanation for why this prediction is a false positive:\n"
        + _DESC +
        "\nGive a one-sentence justification, then on a FINAL line output exactly one label from the "
        "list above (one of: " + " ".join(BUCKETS) + ")."
    )


def parse_bucket(text):
    """The last bucket keyword mentioned (the FINAL-line label); 'unclear' if none."""
    low = text.lower()
    best, pos = "unclear", -1
    for b in BUCKETS:
        i = low.rfind(b)
        if i > pos:
            best, pos = b, i
    return best


def classify_fp(respond, parent, child, ctx):
    text = respond(build_prompt(parent, child, ctx))
    if not text:
        return "unclear", ""
    reason = text.strip().splitlines()[0][:300]
    return parse_bucket(text), reason


def openai_responder(base_url, api_key, model, max_tokens=400, temperature=0.0, max_retries=3):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def respond(prompt):
        for attempt in range(max_retries):
            try:
                r = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens)
                content = r.choices[0].message.content or ""
                if content.strip():
                    return content
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(2)
        return ""
    return respond


def summarize(counts, out_png):
    """Print the per-dataset + overall distribution and (if matplotlib) a stacked-bar chart."""
    datasets = sorted(counts)
    overall = defaultdict(int)
    for ds in datasets:
        for b, n in counts[ds].items():
            overall[b] += n
    total = sum(overall.values())

    print("\n" + "=" * 78)
    print("FALSE-POSITIVE CAUSE DISTRIBUTION")
    print("=" * 78)
    hdr = "dataset".ljust(24) + "".join(b[:12].rjust(13) for b in BUCKETS)
    print(hdr)
    for ds in datasets:
        n = sum(counts[ds].values()) or 1
        row = ds[:23].ljust(24) + "".join(f"{100*counts[ds].get(b,0)/n:11.0f}%" for b in BUCKETS)
        print(row)
    print("-" * len(hdr))
    row = "OVERALL".ljust(24) + "".join(f"{(100*overall.get(b,0)/total if total else 0):11.0f}%" for b in BUCKETS)
    print(row)
    if total:
        top = max(overall, key=overall.get)
        print(f"\n>>> MAJORITY: '{top}' -- {overall[top]}/{total} ({100*overall[top]/total:.0f}%) of false positives.")

    if not total:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        cats = datasets + ["OVERALL"]
        data = {b: [counts[ds].get(b, 0) / (sum(counts[ds].values()) or 1) for ds in datasets]
                   + [overall.get(b, 0) / total] for b in BUCKETS}
        x = np.arange(len(cats)); bottom = np.zeros(len(cats))
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(max(10, len(cats) * 1.1), 6))
        for i, b in enumerate(BUCKETS):
            ax.bar(x, data[b], bottom=bottom, label=b, color=cmap(i))
            bottom += np.array(data[b])
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("fraction of false positives", fontweight="bold")
        ax.set_title("Why are the base-model false positives false? (LLM-classified)", fontweight="bold")
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        ax.set_ylim(0, 1)
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[*] wrote {out_png}")
    except Exception as e:
        print(f"[!] chart skipped ({e})")


def main():
    ap = argparse.ArgumentParser(description="LLM breakdown of WHY the base-model false positives are false.")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="woohoo")
    ap.add_argument("--max_per_dataset", type=int, default=60,
                    help="Random sample of FPs per dataset to classify (0 = all). Sample is deterministic.")
    ap.add_argument("--out_csv", default=os.path.join("results", "fp_reasons.csv"))
    ap.add_argument("--out_png", default=os.path.join("vis", "fp_reasons.png"))
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.results_dir, "*_Our_Method_edge_diagnostics.csv")))
    if not csvs:
        print(f"[!] No *_Our_Method_edge_diagnostics.csv in {args.results_dir}.")
        return
    respond = openai_responder(args.base_url, args.api_key, args.model)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    counts = defaultdict(lambda: defaultdict(int))
    rows_out = []
    for c in csvs:
        loaded = load_dataset(c, args.results_dir)
        if loaded is None:
            continue
        ds, rows, G_pred, gt_pairs, gt_edges = loaded
        fps = [(r["parent"], r["child"]) for r in rows if r["is_fp"] == 1]
        if args.max_per_dataset and len(fps) > args.max_per_dataset:
            random.seed(42)
            fps = random.sample(fps, args.max_per_dataset)
        anc, desc = neighbor_maps(gt_pairs)
        for (P, C) in tqdm(fps, desc=f"  -> {ds}", leave=False):
            bucket, reason = classify_fp(respond, P, C, gt_context(P, C, anc, desc))
            counts[ds][bucket] += 1
            rows_out.append((ds, P, C, bucket, reason))
        n = sum(counts[ds].values()) or 1
        top = max(counts[ds], key=counts[ds].get) if counts[ds] else "n/a"
        print(f"    {ds:26s} classified {n:4d} FPs | top: {top} ({100*counts[ds].get(top,0)/n:.0f}%)")

    if rows_out:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "parent", "child", "bucket", "reason"])
            w.writerows(rows_out)
        print(f"[*] wrote per-edge labels -> {args.out_csv} ({len(rows_out)} rows)")
    summarize(counts, args.out_png)


if __name__ == "__main__":
    main()
