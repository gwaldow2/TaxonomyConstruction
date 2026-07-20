"""Build LoRA SFT data for taxonomy construction from the held-out TRAIN splits.

Turns each domain's `benchmark_sets/<d>_<scale>_train_pairs.json` into supervised examples in
the EXACT prompt format the method uses at inference (our_method.build_prompt), so train and
test formats match. Completions list the gold `child <= parent` lines implied by the TRAIN
graph's transitive closure (ancestor semantics, matching what the 'full' prompt asks for).

Before writing anything it runs a HARD LEAKAGE ASSERTION: if any training pair appears in the
evaluation graph's transitive closure (synonym-aware, same set-overlap the evaluator uses),
the domain is refused. Node overlap is reported but not fatal -- seeing a term is fine, seeing
its answer is not.

    python lora_data_prep.py --domains all --train_scale FULL --eval_scale SUB --out_dir sft

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


def load_eval_graph(domain, scale, bench_dir=BENCH_DIR):
    p = os.path.join(bench_dir, f"{domain}_{scale}.graphml")
    if not os.path.exists(p):
        return None
    G = nx.DiGraph(nx.read_graphml(p))
    if "virtual_root" in G:
        G.remove_node("virtual_root")
    return G


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


def render(examples, variant="full"):
    """Attach the real inference prompt (imported lazily: our_method pulls in data_manager)."""
    from our_method import build_prompt
    return [{"prompt": build_prompt(e["target"], e["candidates"], variant=variant),
             "completion": e["completion"]} for e in examples]


def main():
    ap = argparse.ArgumentParser(description="Build LoRA SFT data from the held-out train splits.")
    ap.add_argument("--domains", nargs="+", default=["all"])
    ap.add_argument("--train_scale", default="FULL", help="Scale whose train_pairs supply training data.")
    ap.add_argument("--eval_scale", default="SUB", help="Scale used for the leakage check (what you evaluate on).")
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
        tp = load_train_pairs(d, args.train_scale, args.bench_dir)
        G_eval = load_eval_graph(d, args.eval_scale, args.bench_dir)
        if not tp:
            print(f"    [!] {d}: no {args.train_scale} train_pairs -- skipping"); continue
        if G_eval is None:
            print(f"    [!] {d}: no {args.eval_scale} eval graph -- skipping"); continue

        violations, overlap = leakage_report(tp, G_eval)
        if violations:
            msg = (f"    [LEAKAGE] {d}: {len(violations)} train pair(s) appear in the {args.eval_scale} "
                   f"eval closure, e.g. {violations[:3]}")
            if not args.allow_leakage:
                print(msg + "  -> REFUSING to write this domain.")
                failed.append(d); continue
            print(msg + "  -> continuing anyway (--allow_leakage)")

        ex = make_examples(tp, args.candidates_per_example, args.max_examples_per_domain)
        recs = render(ex, args.variant)
        path = os.path.join(args.out_dir, f"{d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_pos = sum(1 for r in recs if r["completion"] != "none")
        manifest[d] = {"examples": len(recs), "with_relations": n_pos,
                       "train_pairs": len(tp), "node_overlap_with_eval": round(overlap, 3)}
        print(f"    {d:26s} {len(recs):5d} examples ({n_pos} with relations) | "
              f"train_pairs={len(tp):5d} | eval-node overlap={overlap:.2f} -> {path}")

    with open(os.path.join(args.out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[*] wrote {len(manifest)} domain file(s) to {args.out_dir}/")
    if failed:
        print(f"[!] REFUSED for leakage: {', '.join(failed)} -- fix the split before training on them.")


if __name__ == "__main__":
    main()
