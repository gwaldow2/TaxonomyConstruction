"""Build LoRA SFT data for taxonomy construction, held out from the SUB eval graph.

Turns each domain's ontology into supervised examples in the EXACT prompt format the method
uses at inference (our_method.build_prompt), so train and test formats match. Completions list
the gold `child <= parent` lines implied by the training graph's transitive closure (ancestor
semantics, matching what the 'full' prompt asks for).

Training pool (`--train_source`):
  * graphml (default) -- the LARGE `<d>_FULL.graphml` (the benchmark's 80%-test graph, hundreds
    to thousands of edges) is the pool; the `<d>_SUB.graphml` eval graph is held out. This is
    the real training set. The starved `<d>_FULL_train_pairs.json` (the split's orphaned 20%,
    often 2-83 edges) is NOT used.
  * pairs -- the old path: read `<d>_<train_scale>_train_pairs.json` directly. Kept for
    back-compat; expect tiny volume.

Leakage safety (graphml path): the SUB eval nodes are dropped from the pool (node-disjoint),
then any remaining pair that still appears in the SUB transitive closure (synonym-aware, same
set-overlap the evaluator uses) is filtered out, and the kept set is ASSERTED clean before
anything is written. Teaching a->b leaks the answer whenever (a,b) is an ancestor pair in the
eval key, so the closure -- not just direct edges -- is what's checked.

    python lora_data_prep.py --domains all --train_source graphml --out_dir sft

Writes sft/<domain>.jsonl with {"prompt", "completion"} plus sft/_manifest.json.
"""

import os
import json
import random
import argparse

import networkx as nx

from prune_sweep_analysis import parse_lemma_format, _exploded_pairs, _matches

BENCH_DIR = "benchmark_sets"
ALL_DOMAINS = ["WordNetFood", "CellOntology", "SemEvalFood", "SemEvalScience", "SemEvalEnvironment",
               "LLMs4OL_OBI", "LLMs4OL_MatOnto", "LLMs4OL_SchemaOrg", "LLMs4OL_PO", "medium_components"]


def load_train_pairs(domain, scale, bench_dir=BENCH_DIR):
    p = os.path.join(bench_dir, f"{domain}_{scale}_train_pairs.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_graph(domain, scale, bench_dir=BENCH_DIR):
    p = os.path.join(bench_dir, f"{domain}_{scale}.graphml")
    if not os.path.exists(p):
        return None
    G = nx.DiGraph(nx.read_graphml(p))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


load_eval_graph = load_graph   # back-compat alias


def train_pairs_from_graph(G_pool, G_eval, node_disjoint=True):
    """Build leakage-safe training pairs from a pool graph, holding out the eval graph.

    -> (kept_pairs, stats). SUB is a subgraph of FULL, so node identities match exactly:
    dropping eval nodes (node_disjoint) removes the eval region wholesale, then the closure
    filter catches any surviving pair that still implies an eval ancestor pair (synonym-aware).
    The caller asserts the kept set is clean before writing.
    """
    eval_nodes = set(G_eval.nodes())
    pool = G_pool
    if node_disjoint:
        pool = G_pool.subgraph([n for n in G_pool.nodes() if n not in eval_nodes])
    raw = [{"parent": u, "child": v} for u, v in pool.edges()]

    eval_pairs = _exploded_pairs(list(nx.transitive_closure(G_eval).edges()))
    kept = [tp for tp in raw if not _matches(tp["parent"], tp["child"], eval_pairs)]
    stats = {"pool_edges": G_pool.number_of_edges(),
             "after_node_disjoint": len(raw),
             "dropped_node_overlap": G_pool.number_of_edges() - len(raw),
             "dropped_closure_leak": len(raw) - len(kept),
             "kept": len(kept)}
    return kept, stats


def leakage_report(train_pairs, G_eval):
    """-> (violations, node_overlap_fraction).

    A violation is a training pair that also appears in the EVAL graph's transitive closure
    (synonym-aware). Checking the closure -- not just direct edges -- is the point: teaching
    a->b leaks the answer whenever (a,b) is an ancestor pair in the eval key.
    """
    eval_pairs = _exploded_pairs(list(nx.transitive_closure(G_eval).edges()))
    violations = [(tp["parent"], tp["child"]) for tp in train_pairs
                  if _matches(tp["parent"], tp["child"], eval_pairs)]
    tr_terms, ev_terms = set(), set()
    for tp in train_pairs:
        tr_terms |= set(parse_lemma_format(tp["parent"])) | set(parse_lemma_format(tp["child"]))
    for n in G_eval.nodes():
        ev_terms |= set(parse_lemma_format(n))
    overlap = len(tr_terms & ev_terms) / len(ev_terms) if ev_terms else 0.0
    return violations, overlap


def make_examples(train_pairs, candidates_per_example=99, max_examples=0, seed=42):
    """-> [{target, candidates, completion}] mirroring inference: one example per train node,
    candidates = that node's true relations padded with random negatives."""
    G = nx.DiGraph()
    G.add_edges_from([(tp["parent"], tp["child"]) for tp in train_pairs])
    if G.number_of_nodes() == 0:
        return []
    prim = {n: parse_lemma_format(n)[0] for n in G.nodes()}
    C = nx.transitive_closure(G)
    nodes = list(G.nodes())
    rng = random.Random(seed)
    targets = nodes[:]
    rng.shuffle(targets)
    if max_examples:
        targets = targets[:max_examples]

    out = []
    for t in targets:
        related = [n for n in (set(C.predecessors(t)) | set(C.successors(t))) if n != t]
        rng.shuffle(related)
        keep = related[:candidates_per_example]
        negatives = [n for n in nodes if n != t and n not in set(keep)]
        rng.shuffle(negatives)
        cand = keep + negatives[:max(0, candidates_per_example - len(keep))]
        rng.shuffle(cand)
        lines = []
        for c in cand:
            if C.has_edge(c, t):        # c is an ancestor of t  -> "t is a kind of c"
                lines.append(f"{prim[t]} <= {prim[c]}")
            elif C.has_edge(t, c):      # t is an ancestor of c  -> "c is a kind of t"
                lines.append(f"{prim[c]} <= {prim[t]}")
        out.append({"target": prim[t],
                    "candidates": [prim[c] for c in cand],
                    "completion": "\n".join(lines) if lines else "none"})
    return out


def build_full_prompt(target, candidates):
    """Byte-for-byte copy of our_method.build_prompt(..., variant="full").

    Reproduced here so data prep runs WITHOUT the ontology stack: the live builder lives in
    our_method, which imports data_manager (requests/nltk/obonet) at module load, none of which
    this pure string formatting needs. test_lora_prep.test_prompt_matches_live keeps this in
    lock-step with the original whenever our_method is importable (e.g. in the benchmark env).
    """
    return (
        f"You are identifying hierarchical relationships for the target entity: '{target}'.\n"
        f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
        f"between the target and the candidates.\n"
        f"- If every entity labeled with '{target}' could logically also be labeled with a candidate 'C', output '{target} <= C'\n"
        f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{target}', output 'C <= {target}'\n"
        f"ONLY output relationships involving '{target}'. Do NOT output relationships between the candidates themselves. "
        f"Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
        f"Example: 'anucleate cell' <= 'cell'\n"
        f"Candidates:\n"
        + "\n".join([f"- {c}" for c in candidates]) + "\n\nRelationships:\n")


def render(examples, variant="full"):
    """Attach the inference prompt to each example.

    Prefers the LIVE our_method.build_prompt so any prompt change flows through automatically.
    Falls back to the built-in 'full' copy when our_method can't import (the training-only env
    has no data_manager deps) -- only 'full' is reproduced standalone, which is what the
    experiment uses; any other variant then requires the full env.
    """
    try:
        from our_method import build_prompt
        bp = lambda t, c: build_prompt(t, c, variant=variant)
    except Exception as e:
        if variant != "full":
            raise SystemExit(f"[!] variant '{variant}' needs the full env (our_method/data_manager "
                             f"failed to import: {e}). Only 'full' renders standalone.")
        print("    [i] our_method unavailable -- using the built-in 'full' prompt copy "
              "(kept in sync by test_lora_prep).")
        bp = build_full_prompt
    return [{"prompt": bp(e["target"], e["candidates"]),
             "completion": e["completion"]} for e in examples]


def prepare_domain(d, args):
    """-> (train_pairs, eval_graph, stats) for one domain, or (None, None, reason)."""
    G_eval = load_graph(d, args.eval_scale, args.bench_dir)
    if G_eval is None:
        return None, None, f"no {args.eval_scale} eval graph"

    if args.train_source == "graphml":
        if args.train_scale == args.eval_scale:
            return None, None, f"train_scale == eval_scale ({args.eval_scale}): would train on the eval graph"
        G_pool = load_graph(d, args.train_scale, args.bench_dir)
        if G_pool is None:
            return None, None, f"no {args.train_scale} pool graph"
        tp, stats = train_pairs_from_graph(G_pool, G_eval, node_disjoint=not args.keep_eval_nodes)
        # Belt-and-suspenders: the kept set must now be leakage-clean.
        violations, overlap = leakage_report(tp, G_eval)
        if violations and not args.allow_leakage:
            return None, None, f"{len(violations)} pair(s) still leak after filtering (unexpected): {violations[:3]}"
        stats["node_overlap_with_eval"] = round(overlap, 3)
        return tp, G_eval, stats

    # train_source == "pairs": the old, starved path.
    tp = load_train_pairs(d, args.train_scale, args.bench_dir)
    if not tp:
        return None, None, f"no {args.train_scale} train_pairs"
    violations, overlap = leakage_report(tp, G_eval)
    if violations and not args.allow_leakage:
        return None, None, f"{len(violations)} train pair(s) leak into {args.eval_scale} closure, e.g. {violations[:3]}"
    return tp, G_eval, {"pool_edges": len(tp), "kept": len(tp), "node_overlap_with_eval": round(overlap, 3)}


def main():
    ap = argparse.ArgumentParser(description="Build LoRA SFT data held out from the SUB eval graph.")
    ap.add_argument("--domains", nargs="+", default=["all"])
    ap.add_argument("--train_source", choices=["graphml", "pairs"], default="graphml",
                    help="graphml: FULL graph as pool, hold out SUB (real volume). "
                         "pairs: the starved <d>_<scale>_train_pairs.json (back-compat).")
    ap.add_argument("--train_scale", default="FULL", help="Pool scale (graphml) or train_pairs scale (pairs).")
    ap.add_argument("--eval_scale", default="SUB", help="Eval graph held out / leakage-checked (what you evaluate on).")
    ap.add_argument("--keep_eval_nodes", action="store_true",
                    help="graphml only: keep eval nodes in the pool (edge-disjoint only). "
                         "Default drops them for full node-disjointness.")
    ap.add_argument("--bench_dir", default=BENCH_DIR)
    ap.add_argument("--out_dir", default="sft")
    ap.add_argument("--candidates_per_example", type=int, default=99,
                    help="Candidate list size per example (match your inference scale).")
    ap.add_argument("--max_examples_per_domain", type=int, default=0, help="0 = all train nodes.")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--allow_leakage", action="store_true",
                    help="Do NOT use for real runs: downgrade the leakage assertion to a warning.")
    args = ap.parse_args()

    domains = ALL_DOMAINS if "all" in args.domains else args.domains
    os.makedirs(args.out_dir, exist_ok=True)
    manifest, failed = {}, []

    for d in domains:
        tp, G_eval, stats = prepare_domain(d, args)
        if tp is None:
            print(f"    [!] {d}: {stats} -- skipping")
            failed.append(d); continue
        if not tp:
            print(f"    [!] {d}: 0 training pairs after holdout -- skipping")
            failed.append(d); continue

        ex = make_examples(tp, args.candidates_per_example, args.max_examples_per_domain)
        recs = render(ex, args.variant)
        path = os.path.join(args.out_dir, f"{d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_pos = sum(1 for r in recs if r["completion"] != "none")
        manifest[d] = {"examples": len(recs), "with_relations": n_pos,
                       "train_pairs": len(tp), **stats}
        print(f"    {d:26s} {len(recs):5d} examples ({n_pos} w/ rel) | pool_edges={stats.get('pool_edges', '?'):>5} "
              f"kept={stats['kept']:>5} | eval-node overlap={stats['node_overlap_with_eval']:.2f} -> {path}")

    with open(os.path.join(args.out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[*] wrote {len(manifest)} domain file(s) to {args.out_dir}/")
    if failed:
        print(f"[!] skipped: {', '.join(failed)}")


if __name__ == "__main__":
    main()
