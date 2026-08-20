"""Audit what relation types the extracted benchmarks and the predicted taxonomies actually hold.

The pipeline is supposed to keep only is-a edges -- data_manager keeps `key == 'is_a'` from OBO,
WordNet uses hyponyms(), SemEval .taxo files are hypernym pairs -- but that has never been
checked against the graphs being evaluated, and the FP analysis (gt_missing 38%,
ontology_strictness 22%, plus a not_isa bucket) makes it look doubtful.

It cannot be checked by inspection: every loader calls add_edge(u, v) with no attributes, so the
.graphml files carry no relation-type information. An edge's type is only recoverable by
classifying the edge itself, which is what this does -- for the GROUND TRUTH as well as the
predictions, using one shared vocabulary so the two are directly comparable.

GT is a full census (~106 edges per SUB dataset), which makes the is-a-only filter exact rather
than extrapolated. Predictions are far larger, so they are sampled stratified across TP and FP.

    python relation_type_audit.py --dry_run                    # show the prompts, call nothing
    python relation_type_audit.py --datasets WordNetFood_SUB --max_pred_per_dataset 20
    python relation_type_audit.py                              # everything in results/

Writes results/relation_types.csv, results/relation_types_summary.csv, vis/relation_types.png,
and results/GT_<ds>_isa_only.graphml per dataset. Re-runs resume from the CSV, so an interrupted
audit never re-pays for edges it already classified.
"""

import os
import csv
import glob
import random
import argparse
from collections import Counter, defaultdict

import networkx as nx

RESULTS_DIR = "results"
VIS_DIR = "vis"

# One vocabulary for GT and predicted edges alike. It extends the relation names already used in
# fp_reason_analysis's not_isa description so the two analyses stay consistent.
CANONICAL_TYPES = [
    "is_a", "is_a_inverted", "part_of", "has_part", "instance_of", "attribute_of",
    "made_of", "located_in", "derives_from", "synonym", "other_related", "unrelated",
]

_TYPE_DESC = (
    "- is_a: proper subsumption -- every instance of the CHILD is also an instance of the PARENT.\n"
    "- is_a_inverted: subsumption holds but the direction is backwards (the stated parent is really "
    "the child).\n"
    "- part_of: the child is a component, region or portion of the parent, not a kind of it.\n"
    "- has_part: the parent is a component of the child (the reverse of part_of).\n"
    "- instance_of: the child is a specific named individual and the parent is its class.\n"
    "- attribute_of: one side is a property, quality, role or measurement and the other is the thing "
    "that bears it.\n"
    "- made_of: the child is constituted from the parent as a material or ingredient.\n"
    "- located_in: a spatial or containment relation rather than a kind-of relation.\n"
    "- derives_from: a developmental, origin or transformation relation (e.g. develops_from).\n"
    "- synonym: the two names denote the SAME concept.\n"
    "- other_related: genuinely related, but by none of the relations above.\n"
    "- unrelated: no sensible relation between the two concepts.\n"
)


def parse_type(text):
    """Last canonical label mentioned wins; ties at the same position go to the longer label.

    The tie rule matters: 'is_a' is a substring of 'is_a_inverted', so a plain last-occurrence
    scan would report 'is_a' for a response that actually said 'is_a_inverted'.
    """
    low = (text or "").lower().replace("-", "_")
    best, pos, ln = "unrelated", -1, -1
    for t in CANONICAL_TYPES:
        i = low.rfind(t)
        if i > pos or (i == pos and len(t) > ln):
            best, pos, ln = t, i, len(t)
    return best if pos >= 0 else "unrelated"


def build_prompt(parent, child, ctx, source):
    """Ask for the relation that actually holds, given where the terms sit in the reference."""
    from data_manager import get_primary_term
    p, c = get_primary_term(parent), get_primary_term(child)

    def _fmt(lst):
        return ", ".join("'%s'" % x for x in lst) if lst else "(none listed)"

    origin = ("This edge comes from a reference taxonomy that is supposed to contain only is-a "
              "relations." if source == "gt" else
              "This edge was produced by an automatic taxonomy builder that was asked for is-a "
              "relations only.")
    return (
        "You are auditing the relation types in a taxonomy.\n\n"
        f"The taxonomy states:  '{c}'  is placed under  '{p}'  (parent -> child).\n"
        f"{origin}\n\n"
        "For reference, in the taxonomy these terms also appear with:\n"
        f"- '{c}' placed under: {_fmt(ctx['child_is_a'])}.\n"
        f"- '{p}' placed under: {_fmt(ctx['parent_is_a'])}.\n"
        f"- '{p}' also contains: {_fmt(ctx['parent_contains'])}.\n\n"
        "Judge the relation that ACTUALLY holds between the two concepts, not the one the "
        "taxonomy claims. Choose exactly one:\n"
        + _TYPE_DESC +
        "\nGive a one-sentence justification, then on a FINAL line output exactly one label from: "
        + " ".join(CANONICAL_TYPES)
    )


def load_gt_graph(results_dir, dataset):
    p = os.path.join(results_dir, f"GT_{dataset}_eval.graphml")
    if not os.path.exists(p):
        return None
    G = nx.DiGraph(nx.read_graphml(p))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


def find_datasets(results_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "GT_*_eval.graphml"))):
        name = os.path.basename(p)
        out.append(name[len("GT_"):-len("_eval.graphml")])
    return out


def load_pred_edges(results_dir, dataset, label=None):
    """-> ([(parent, child, is_fp)], path) from the run's own edge diagnostics.

    The is_fp column is reused rather than recomputed, so the strata here match exactly how the
    run was scored.
    """
    hits = []
    for p in sorted(glob.glob(os.path.join(results_dir, f"{dataset}_*_edge_diagnostics.csv"))):
        if label and label not in os.path.basename(p):
            continue
        hits.append(p)
    if not hits:
        return [], None
    path = hits[-1]
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(r["parent"], r["child"], int(r["is_fp"])) for r in csv.DictReader(f)]
    return rows, path


def stratified_sample(edges, n, seed=42):
    """Sample n edges keeping the TP/FP proportion of the full set."""
    if not n or n >= len(edges):
        return list(edges)
    rng = random.Random(seed)
    strata = defaultdict(list)
    for e in edges:
        strata[e[2]].append(e)
    out = []
    for k, group in sorted(strata.items()):
        take = max(1, round(n * len(group) / len(edges)))
        g = group[:]
        rng.shuffle(g)
        out += g[:take]
    rng.shuffle(out)
    return out[:n]


def load_cache(path):
    """Previously classified edges, so an interrupted run resumes instead of re-paying."""
    if not os.path.exists(path):
        return {}, []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["dataset"], r["source"], r["parent"], r["child"]): r for r in rows}, rows


def classify_all(work, respond, cache, model, verbose=True):
    """work: [(dataset, source, parent, child, is_fp, ctx)] -> (rows, n_newly_classified)."""
    rows, n_new = [], 0
    for i, (ds, source, p, c, is_fp, ctx) in enumerate(work, 1):
        key = (ds, source, p, c)
        if key in cache:
            rows.append(cache[key])
            continue
        text = respond(build_prompt(p, c, ctx, source))
        t = parse_type(text)
        just = text.strip().splitlines()[0][:300] if text.strip() else ""
        row = {"dataset": ds, "source": source, "parent": p, "child": c,
               "is_fp": is_fp, "type": t, "model": model, "justification": just}
        rows.append(row)
        cache[key] = row
        n_new += 1
        if verbose and n_new % 25 == 0:
            print(f"    ... {n_new} newly classified ({i}/{len(work)} processed)")
    return rows, n_new


def write_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["dataset", "source", "parent", "child", "is_fp", "type", "model", "justification"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_summary(path, lines):
    if not lines:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "source", "type", "count", "total", "proportion"])
        w.writeheader()
        w.writerows(lines)


def summarize(rows, out_csv, out_png):
    """Per-dataset and pooled proportions, plus a stacked-bar figure."""
    per = defaultdict(Counter)
    for r in rows:
        per[(r["dataset"], r["source"])][r["type"]] += 1

    lines = []
    for (ds, src), c in sorted(per.items()):
        tot = sum(c.values())
        for t in CANONICAL_TYPES:
            if c[t]:
                lines.append({"dataset": ds, "source": src, "type": t,
                              "count": c[t], "total": tot, "proportion": round(c[t] / tot, 4)})
    write_summary(out_csv, lines)

    print("\n" + "=" * 78 + "\n### relation-type proportions\n" + "=" * 78)
    for (ds, src), c in sorted(per.items()):
        tot = sum(c.values())
        isa = c["is_a"] / tot if tot else 0.0
        print(f"\n  {ds} [{src}]  n={tot}   is_a = {isa:.1%}")
        for t, k in c.most_common():
            if t != "is_a":
                print(f"      {t:16s} {k:4d}  ({k / tot:.1%})")

    pooled = Counter()
    for (ds, src), c in per.items():
        if src == "gt":
            pooled += c
    if pooled:
        tot = sum(pooled.values())
        print(f"\n  POOLED GT across datasets: n={tot}   is_a = {pooled['is_a'] / tot:.1%}")
        print(f"  non-is_a GT edges: {tot - pooled['is_a']} ({1 - pooled['is_a'] / tot:.1%})")

    try:
        plot(per, out_png)
        print(f"\n[*] wrote {out_png}")
    except Exception as e:
        print(f"[!] figure skipped: {e}")


def plot(per, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = sorted({ds for ds, _ in per})
    sources = ["gt", "pred"]
    present = [t for t in CANONICAL_TYPES if any(c[t] for c in per.values())]
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(present), 2)))
    colors = {t: cmap[i] for i, t in enumerate(present)}

    fig, ax = plt.subplots(figsize=(max(10, 1.9 * len(datasets)), 7))
    width, xs, labels = 0.38, [], []
    for i, ds in enumerate(datasets):
        for j, src in enumerate(sources):
            c = per.get((ds, src))
            x = i + (j - 0.5) * width
            xs.append(x)
            labels.append(f"{ds.replace('_SUB', '')}\n{src}")
            if not c:
                continue
            tot = sum(c.values()) or 1
            bottom = 0.0
            for t in present:
                if not c[t]:
                    continue
                h = c[t] / tot
                ax.bar(x, h, width * 0.92, bottom=bottom, color=colors[t])
                bottom += h
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("proportion of edges", fontweight="bold")
    ax.set_ylim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in present]
    ax.legend(handles, present, fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Relation types actually present: ground truth vs predicted\n"
                 "(the pipeline assumes every edge is is_a)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_isa_only_gt(results_dir, dataset, rows, keep_types=("is_a",)):
    """Emit GT_<ds>_isa_only.graphml keeping only edges classified into keep_types.

    Edges with no classification are KEPT (an unaudited edge is not evidence of contamination).
    -> (path, n_kept, n_dropped), or None when the GT graph is missing.
    """
    G = load_gt_graph(results_dir, dataset)
    if G is None:
        return None
    keep = set(keep_types)
    types = {(r["parent"], r["child"]): r["type"] for r in rows
             if r["dataset"] == dataset and r["source"] == "gt"}
    drop = [e for e in G.edges() if e in types and types[e] not in keep]
    G_out = G.copy()
    G_out.remove_edges_from(drop)
    path = os.path.join(results_dir, f"GT_{dataset}_isa_only.graphml")
    nx.write_graphml(G_out, path)
    return path, G_out.number_of_edges(), len(drop)


def rescore(results_dir, dataset, pred_edges):
    """Score the predictions against the original GT and the is-a-only GT. -> dict of deltas."""
    from evaluator import evaluate_all_modes
    G_orig = load_gt_graph(results_dir, dataset)
    p_iso = os.path.join(results_dir, f"GT_{dataset}_isa_only.graphml")
    if G_orig is None or not os.path.exists(p_iso):
        return None
    G_iso = nx.DiGraph(nx.read_graphml(p_iso))
    G_pred = nx.DiGraph()
    G_pred.add_nodes_from(G_orig.nodes())
    G_pred.add_edges_from([(p, c) for p, c, _ in pred_edges])
    tmp = os.path.join(results_dir, f"_rescore_{dataset}")
    a = evaluate_all_modes(G_pred, G_orig, tmp + "_orig")["Cond_Clos"]
    b = evaluate_all_modes(G_pred, G_iso, tmp + "_isa")["Cond_Clos"]
    for f in glob.glob(tmp + "*"):
        try:
            os.remove(f)
        except OSError:
            pass
    return {"F1_orig": a["F1"], "F1_isa": b["F1"], "dF1": b["F1"] - a["F1"],
            "P_orig": a["Precision"], "P_isa": b["Precision"], "dP": b["Precision"] - a["Precision"],
            "R_orig": a["Recall"], "R_isa": b["Recall"], "dR": b["Recall"] - a["Recall"]}


def main():
    ap = argparse.ArgumentParser(description="Audit relation types in GT and predicted taxonomies.")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--datasets", nargs="+", default=None, help="Default: every GT_*_eval.graphml.")
    ap.add_argument("--pred_label", default=None, help="Substring filter for the diagnostics file.")
    ap.add_argument("--max_pred_per_dataset", type=int, default=200,
                    help="Stratified TP/FP sample size for predicted edges (0 = all).")
    ap.add_argument("--max_gt_per_dataset", type=int, default=0,
                    help="0 = full census (recommended: it makes the is-a filter exact).")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="woohoo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default=os.path.join(RESULTS_DIR, "relation_types.csv"))
    ap.add_argument("--summary_csv", default=os.path.join(RESULTS_DIR, "relation_types_summary.csv"))
    ap.add_argument("--out_png", default=os.path.join(VIS_DIR, "relation_types.png"))
    ap.add_argument("--no_resume", action="store_true", help="Ignore any existing --out_csv.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print one GT and one predicted prompt, then exit without calling the LLM.")
    ap.add_argument("--skip_rescore", action="store_true")
    args = ap.parse_args()

    from fp_reason_analysis import neighbor_maps, gt_context
    from evaluator import gt_closure_term_pairs

    datasets = args.datasets or find_datasets(args.results_dir)
    if not datasets:
        raise SystemExit(f"[!] no GT_*_eval.graphml in {args.results_dir}/")
    print(f"[*] auditing {len(datasets)} dataset(s): {', '.join(datasets)}")

    work, pred_by_ds = [], {}
    for ds in datasets:
        G = load_gt_graph(args.results_dir, ds)
        if G is None:
            print(f"    [!] {ds}: no GT graph -- skipping")
            continue
        anc, desc = neighbor_maps(gt_closure_term_pairs(G))

        gt_edges = [(p, c, 0) for p, c in G.edges()]
        if args.max_gt_per_dataset:
            gt_edges = stratified_sample(gt_edges, args.max_gt_per_dataset, args.seed)
        for p, c, _ in gt_edges:
            work.append((ds, "gt", p, c, "", gt_context(p, c, anc, desc)))

        pred, path = load_pred_edges(args.results_dir, ds, args.pred_label)
        pred_by_ds[ds] = pred
        sample = stratified_sample(pred, args.max_pred_per_dataset, args.seed) if pred else []
        for p, c, fp in sample:
            work.append((ds, "pred", p, c, fp, gt_context(p, c, anc, desc)))
        src = f"  <- {os.path.basename(path)}" if path else "  (no diagnostics)"
        print(f"    {ds:26s} gt={len(gt_edges):4d}  pred={len(sample):4d} of {len(pred)}{src}")

    if args.dry_run:
        for src in ("gt", "pred"):
            item = next((w for w in work if w[1] == src), None)
            if item:
                ds, s, p, c, fp, cx = item
                print("\n" + "=" * 78)
                print(f"### {src.upper()} prompt  ({ds}: {c} under {p})")
                print("=" * 78)
                print(build_prompt(p, c, cx, s))
        print(f"\n[*] dry run: {len(work)} edges would be classified. Nothing was called.")
        return

    cache, _ = ({}, []) if args.no_resume else load_cache(args.out_csv)
    if cache:
        print(f"[*] resuming: {len(cache)} edges already classified in {args.out_csv}")

    from fp_reason_analysis import openai_responder
    respond = openai_responder(args.base_url, args.api_key, args.model)
    print(f"[*] classifying {len(work)} edges with {args.model} ...")
    rows, n_new = classify_all(work, respond, cache, args.model)
    write_rows(args.out_csv, rows)
    print(f"[*] {n_new} newly classified, {len(rows)} total -> {args.out_csv}")

    summarize(rows, args.summary_csv, args.out_png)

    print("\n" + "=" * 78 + "\n### is-a-only ground truth\n" + "=" * 78)
    for ds in datasets:
        res = write_isa_only_gt(args.results_dir, ds, rows)
        if not res:
            continue
        path, kept, dropped = res
        print(f"\n  {ds}: kept {kept} edges, dropped {dropped} non-is_a -> {os.path.basename(path)}")
        if args.skip_rescore or not pred_by_ds.get(ds):
            continue
        d = rescore(args.results_dir, ds, pred_by_ds[ds])
        if d:
            print(f"      Cond_Clos F1 {d['F1_orig']:.4f} -> {d['F1_isa']:.4f}  ({d['dF1']:+.4f})")
            print(f"      precision    {d['P_orig']:.4f} -> {d['P_isa']:.4f}  ({d['dP']:+.4f})")
            print(f"      recall       {d['R_orig']:.4f} -> {d['R_isa']:.4f}  ({d['dR']:+.4f})")


if __name__ == "__main__":
    main()
