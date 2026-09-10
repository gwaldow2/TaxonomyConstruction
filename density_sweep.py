"""Find the judgment-density knee: batch m targets per call and sweep m.

The structured method asks ~99 binary subsumption judgments per call; the single call asks
~4,950. The chunk-size sweep can only probe BELOW 99, so a flat result there cannot say where
capacity saturates -- the interesting region is between the two extremes. This sweep holds
everything else fixed (prompt semantics, output format, pair coverage) and moves only the
number of targets per call: judgments per call = m x (N-1), so m in {1,2,5,10,25,50,100}
spans ~99 to ~9,900 on a 100-term vocabulary.

Design invariants, so m is the ONLY moving part:
  * every call lists its m targets AND the full entity list, so each target is judged against
    all N-1 others at every m -- pair coverage never changes with m;
  * every term is a target exactly once per dataset, so every pair is judged exactly twice
    (once from each side) at every m, preserving the self-agreement semantics;
  * the rules are the structured 'full' variant's wording, generalized from one target to a
    target list; parsing is the structured method's own '<=' line parser;
  * m=1 is included as the anchor -- it is the structured method minus chunking, so the sweep
    connects directly to the existing control numbers.

    python density_sweep.py --dry_run
    python density_sweep.py --model google/gemma-4-31b-it --results_file density_gemma4.json

Rows are written in the standard benchmark-results format with method label
"Our Method [density m=<m>]"; analyze with:  python mechanism_report.py density ...
"""

import os
import time
import random
import argparse

import networkx as nx

CORE_DATASETS = ["WordNetFood", "CellOntology", "SemEvalFood", "LLMs4OL_OBI",
                 "LLMs4OL_PO", "LLMs4OL_SchemaOrg", "LLMs4OL_MatOnto"]


def build_density_prompt(targets, all_terms):
    """The structured 'full' variant generalized to m targets.

    The entity list always contains the ENTIRE vocabulary (targets included): a shared
    candidates list that excluded co-targets would silently drop within-group pairs, and the
    dropped fraction would grow with m -- confounding density with coverage.
    """
    t_list = "\n".join(f"- {t}" for t in targets)
    e_list = "\n".join(f"- {t}" for t in all_terms)
    return (
        "You are identifying hierarchical relationships for the following target entities:\n"
        f"{t_list}\n"
        "Below is the complete list of entities. For EACH target, identify any subclass or "
        "superclass relationships between that target and the other entities.\n"
        "- If every entity labeled with a target 'T' could logically also be labeled with "
        "another entity 'C', output 'T <= C'\n"
        "- If every entity labeled with an entity 'C' could logically also be labeled with a "
        "target 'T', output 'C <= T'\n"
        "ONLY output relationships involving at least one target. Do NOT output relationships "
        "between non-target entities. Output each relationship on a new line. If there are no "
        "relationships, output 'none'.\n\n"
        "Example: 'anucleate cell' <= 'cell'\n"
        "Entities:\n"
        f"{e_list}\n\nRelationships:\n")


def make_groups(terms, m, seed=42):
    """Deterministic groups of m targets covering every term exactly once."""
    order = sorted(terms)
    random.Random(seed).shuffle(order)
    return [order[i:i + m] for i in range(0, len(order), m)]


def extract_density_edges(nodes, client, model_name, m, max_retries=3):
    """Run the m-targets-per-call extraction -> raw DiGraph over the full node set."""
    from tqdm import tqdm
    from data_manager import get_primary_term
    from our_method import _llm_call, _parse_relations, EXTRACT_MAX_TOKENS
    pmap = {get_primary_term(n): n for n in nodes}
    all_terms = sorted(pmap)
    raw = nx.DiGraph()
    raw.add_nodes_from(nodes)
    groups = make_groups(all_terms, m)
    for targets in tqdm(groups, desc=f"  -> [Density m={m}]", leave=False):
        prompt = build_density_prompt(targets, all_terms)
        content, reasoning = _llm_call(client, model_name, prompt,
                                       max_tokens=EXTRACT_MAX_TOKENS, max_retries=max_retries)
        edges = _parse_relations(content, pmap) or _parse_relations(reasoning, pmap)
        for sup, sub in edges:
            raw.add_edge(sup, sub)
    return raw, len(groups)


def main():
    ap = argparse.ArgumentParser(description="Targets-per-call density sweep (knee finder).")
    ap.add_argument("--datasets", nargs="+", default=CORE_DATASETS)
    ap.add_argument("--targets_per_call", nargs="+", type=int,
                    default=[1, 2, 5, 10, 25, 50, 100])
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="woohoo")
    ap.add_argument("--scale", default="SUB")
    ap.add_argument("--results_file", default="density_results.json")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true",
                    help="Show group counts and one sample prompt per m; no LLM calls.")
    args = ap.parse_args()

    from data_manager import load_benchmark_graph, get_primary_term
    from our_method import cluster_synonyms_and_enforce_dag
    from evaluator import evaluate_all_modes, update_benchmark_results

    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for domain in args.datasets:
        G_gt, _ = load_benchmark_graph(domain, scale=args.scale)
        if G_gt is None:
            print(f"[!] {domain}: no {args.scale} benchmark graph -- skipping")
            continue
        ds = f"{domain}_{args.scale}"
        nodes = [n for n in G_gt.nodes() if n != "virtual_root"]
        if "virtual_root" in G_gt:
            G_gt = G_gt.copy()
            G_gt.remove_node("virtual_root")
        n_terms = len(nodes)
        print(f"\n### {ds} ({n_terms} terms)")

        for m in args.targets_per_call:
            m_eff = min(m, n_terms)
            n_groups = (n_terms + m_eff - 1) // m_eff
            jpc = m_eff * (n_terms - 1)
            if args.dry_run:
                print(f"    m={m_eff:4d}: {n_groups:3d} calls, ~{jpc} judgments/call")
                if m == args.targets_per_call[0]:
                    from data_manager import get_primary_term as gpt_
                    terms = sorted(gpt_(n) for n in nodes)
                    print("    sample prompt head: "
                          f"{build_density_prompt(terms[:m_eff], terms)[:150]!r}")
                continue
            t0 = time.time()
            raw, n_calls = extract_density_edges(nodes, client, args.model, m_eff,
                                                 args.max_retries)
            G_pred = cluster_synonyms_and_enforce_dag(raw)
            if "virtual_root" in G_pred:
                G_pred.remove_node("virtual_root")
            runtime = time.time() - t0
            label = f"Our Method [density m={m_eff}]"
            safe = f"Density_m{m_eff}"
            metrics = evaluate_all_modes(G_pred, G_gt, f"./results/{ds}_{safe}")
            flat = {"Cond_Red_F1": metrics["Cond_Red"]["F1"],
                    "Cond_Red_Precision": metrics["Cond_Red"]["Precision"],
                    "Cond_Red_Recall": metrics["Cond_Red"]["Recall"],
                    "Cond_Clos_F1": metrics["Cond_Clos"]["F1"],
                    "Cond_Clos_Precision": metrics["Cond_Clos"]["Precision"],
                    "Cond_Clos_Recall": metrics["Cond_Clos"]["Recall"],
                    "Exp_Raw_F1": metrics["Exp_Raw"]["F1"],
                    "Exp_Clos_F1": metrics["Exp_Clos"]["F1"],
                    "Runtime_sec": runtime,
                    "judgments_per_call": jpc, "n_calls": n_calls}
            update_benchmark_results(dataset_name=ds, method_name=label, metrics_dict=flat,
                                     use_synsets=False, explode_nodes=False,
                                     filepath=args.results_file, model=args.model)
            print(f"    m={m_eff:4d}: {n_calls:3d} calls | ~{jpc} judgments/call | "
                  f"CondClos F1={flat['Cond_Clos_F1']:.3f} "
                  f"P={flat['Cond_Clos_Precision']:.3f} R={flat['Cond_Clos_Recall']:.3f} | "
                  f"{runtime:.0f}s")

    if args.dry_run:
        print("\n[*] dry run complete -- nothing was called, nothing written.")
    else:
        print(f"\n[*] results -> {args.results_file}; analyze with "
              f"'python mechanism_report.py density --results {args.results_file} --tag <tag>'")


if __name__ == "__main__":
    main()
