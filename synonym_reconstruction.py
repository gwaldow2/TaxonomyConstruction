"""Direct measurement of synonym-cluster reconstruction for the O(N) structured method.

The condensed metrics match edges by synonym set-overlap, so end-to-end scores cannot certify
that a method actually reconstructed synonym clusters -- a method that never merges is barely
penalized. This measures reconstruction DIRECTLY: build a synset-formatted benchmark, explode
the clusters into individual terms, run the structured method over the exploded vocabulary,
form predicted clusters from mutual assertions (u<=v AND v<=u -- exactly the rule
condense_synonyms applies), and score the predicted partition against the reference one.

Reported per dataset and pooled:
  * pairwise same-cluster precision / recall / F1 (the partition comparison);
  * per reference cluster: exact / merged (kept together but absorbed outsiders) /
    split (spread over several components), so both failure directions are visible --
    under-merging (missed synonyms) and over-merging (the Qwen mega-cluster failure mode);
  * largest predicted component and reciprocal-assertion count.

The benchmark universe is built FRESH at runtime from the source loaders with synsets enabled
(seeded closed-subgraph sample, same algorithm as the benchmark) -- benchmark_sets/ on disk is
synset-free and is NOT touched, so nothing else in the suite is disturbed. Raw predicted edges
are saved per dataset, so this analysis is re-runnable offline forever.

    python synonym_reconstruction.py --dry_run          # build universes, no LLM calls
    python synonym_reconstruction.py                    # Gemma-4 via local vLLM

Writes results/synonym_reconstruction/: summary.csv, raw_edges_<ds>.csv,
gt_clusters_<ds>.csv, pred_clusters_<ds>.csv, manifest.json, reconstruction.png.
"""

import os
import json
import csv
import time
import argparse
from collections import defaultdict

import networkx as nx

OUT_DIR = os.path.join("results", "synonym_reconstruction")


# ----------------------------------------------------------------------------
# Benchmark universe (synset mode, built at runtime; benchmark_sets/ untouched)
# ----------------------------------------------------------------------------

def synset_graph(domain):
    """-> the domain's 80% test graph with use_synsets=True node formatting."""
    from data_manager import get_wordnet_food_graph, get_cell_ontology_graph, get_semeval_graph
    loaders = {
        "WordNetFood": lambda: get_wordnet_food_graph(use_synsets=True),
        "CellOntology": lambda: get_cell_ontology_graph(use_synsets=True),
        "SemEvalFood": lambda: get_semeval_graph("SemEvalFood", use_synsets=True),
        "SemEvalScience": lambda: get_semeval_graph("SemEvalScience", use_synsets=True),
        "SemEvalEnvironment": lambda: get_semeval_graph("SemEvalEnvironment", use_synsets=True),
    }
    if domain not in loaders:
        raise SystemExit(f"[!] no synset-capable loader registered for {domain} "
                         f"(available: {', '.join(sorted(loaders))})")
    G, _train_pairs = loaders[domain]()
    return G


def build_universe(G, n_nodes):
    """Seeded ancestor-closed subsample (same algorithm and seed as the benchmark),
    -> (gt_clusters, term_to_cluster, terms, n_ambiguous).

    gt_clusters: {cluster_node: frozenset(member terms)}. A term appearing in more than one
    cluster (e.g. a polysemous WordNet lemma) is AMBIGUOUS and dropped from the experiment
    entirely -- with duplicate surface forms in the vocabulary, neither the model nor the
    scorer could tell the copies apart.
    """
    from data_manager import get_closed_subgraph, parse_lemma_format
    sub = get_closed_subgraph(G, n_nodes).copy()
    if "virtual_root" in sub:
        sub.remove_node("virtual_root")

    owners = defaultdict(set)
    for node in sub.nodes():
        for t in parse_lemma_format(node):
            owners[t].add(node)
    ambiguous = {t for t, o in owners.items() if len(o) > 1}

    gt_clusters, term_to_cluster = {}, {}
    for node in sub.nodes():
        members = frozenset(t for t in parse_lemma_format(node) if t not in ambiguous)
        if not members:
            continue
        gt_clusters[node] = members
        for t in members:
            term_to_cluster[t] = node
    terms = sorted(term_to_cluster)
    return gt_clusters, term_to_cluster, terms, len(ambiguous)


# ----------------------------------------------------------------------------
# Extraction (same prompt, call and parse policy as the structured method)
# ----------------------------------------------------------------------------

def extract_raw_edges(terms, client, model_name, chunk_size=1000, max_retries=3):
    """Per-target extraction over the exploded vocabulary -> raw DiGraph, PRE-condensation.

    Mirrors _extract_condensed_with_votes' policy: committed answer first, reasoning
    scratchpad only as fallback -- but keeps the raw graph instead of condensing it.
    """
    from tqdm import tqdm
    from our_method import build_prompt, _llm_call, _parse_relations, EXTRACT_MAX_TOKENS
    pmap = {t: t for t in terms}
    raw = nx.DiGraph()
    raw.add_nodes_from(terms)
    for target in tqdm(terms, desc="  -> [SynRecon]", leave=False):
        cands = [t for t in terms if t != target]
        for i in range(0, len(cands), chunk_size):
            prompt = build_prompt(target, cands[i:i + chunk_size], variant="full")
            content, reasoning = _llm_call(client, model_name, prompt,
                                           max_tokens=EXTRACT_MAX_TOKENS, max_retries=max_retries)
            edges = _parse_relations(content, pmap) or _parse_relations(reasoning, pmap)
            for sup, sub in edges:
                raw.add_edge(sup, sub)
    return raw


# ----------------------------------------------------------------------------
# Clustering and scoring (pure functions -- covered by test_synonym_reconstruction)
# ----------------------------------------------------------------------------

def mutual_components(G):
    """-> ([frozenset components], n_reciprocal_pairs). Same merge rule as condense_synonyms."""
    mutual = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    S = nx.Graph()
    S.add_nodes_from(G.nodes())
    S.add_edges_from(mutual)
    return [frozenset(c) for c in nx.connected_components(S)], len(mutual) // 2


def pair_set(clusters):
    """Unordered same-cluster term pairs implied by a partition."""
    return {frozenset((a, b)) for c in clusters for a in c for b in c if a < b}


def score_partition(pred_clusters, gt_clusters):
    """Pairwise same-cluster precision/recall/F1 of the predicted partition.

    No predicted pairs means nothing was wrongly merged, so precision is vacuously 1.0
    (recall then carries the failure); no reference pairs makes recall vacuously 1.0.
    """
    pp, gp = pair_set(pred_clusters), pair_set(gt_clusters)
    tp = pp & gp
    precision = len(tp) / len(pp) if pp else 1.0
    recall = len(tp) / len(gp) if gp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pair_tp": len(tp), "pair_pred": len(pp), "pair_gt": len(gp),
            "precision": precision, "recall": recall, "f1": f1}


def gt_cluster_report(gt_clusters, pred_clusters):
    """Per multi-member reference cluster: exact / merged / split.

    exact:  one predicted component, identical member set.
    merged: members kept together but the component absorbed outside terms (over-merge).
    split:  members spread over several components (under-merge; a cluster fully
            shattered into singletons is the extreme case, n_pred_components == size).
    """
    pred_of = {t: c for c in pred_clusters for t in c}
    rows = []
    for node, g in sorted(gt_clusters.items()):
        if len(g) < 2:
            continue
        comps = {pred_of.get(t, frozenset((t,))) for t in g}
        if len(comps) == 1:
            status = "exact" if next(iter(comps)) == g else "merged"
        else:
            status = "split"
        rows.append({"cluster": node, "size": len(g), "members": "|".join(sorted(g)),
                     "status": status, "n_pred_components": len(comps)})
    return rows


def pred_cluster_report(pred_clusters, term_to_cluster):
    """Per multi-member predicted component: which reference clusters it spans."""
    rows = []
    for c in sorted((c for c in pred_clusters if len(c) > 1), key=len, reverse=True):
        spanned = sorted({term_to_cluster.get(t, "?") for t in c})
        rows.append({"size": len(c), "members": "|".join(sorted(c)),
                     "n_gt_clusters_spanned": len(spanned),
                     "gt_clusters": "|".join(spanned)})
    return rows


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot(summary_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in summary_rows if r["dataset"] != "POOLED"]
    pooled = next((r for r in summary_rows if r["dataset"] == "POOLED"), None)
    if pooled:
        rows = rows + [pooled]
    labels = [r["dataset"] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for off, key, color in [(-0.25, "precision", "#4C72B0"), (0.0, "recall", "#DD8452"),
                            (0.25, "f1", "#55A868")]:
        ax1.bar(x + off, [r[key] for r in rows], 0.23, label=key, color=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("pairwise same-cluster score", fontweight="bold")
    ax1.legend()
    ax1.set_title("Synonym reconstruction: partition agreement", fontweight="bold")

    per = [r for r in rows if r["dataset"] != "POOLED"]
    xb = np.arange(len(per))
    bottom = np.zeros(len(per))
    for key, color in [("clusters_exact", "#55A868"), ("clusters_merged", "#DD8452"),
                       ("clusters_split", "#C44E52")]:
        vals = np.array([r[key] for r in per], dtype=float)
        ax2.bar(xb, vals, 0.6, bottom=bottom, label=key.replace("clusters_", ""), color=color)
        bottom += vals
    ax2.set_xticks(xb)
    ax2.set_xticklabels([r["dataset"] for r in per], rotation=20, ha="right")
    ax2.set_ylabel("reference multi-member clusters", fontweight="bold")
    ax2.legend()
    ax2.set_title("Per-cluster outcome (largest predicted component annotated)", fontweight="bold")
    for i, r in enumerate(per):
        ax2.annotate(f"max comp: {r['largest_pred_component']}", (i, bottom[i]),
                     ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Direct synonym-reconstruction measurement.")
    ap.add_argument("--datasets", nargs="+",
                    default=["WordNetFood", "CellOntology", "SemEvalFood"])
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="woohoo")
    ap.add_argument("--n_nodes", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--dry_run", action="store_true",
                    help="Build the universes and report cluster stats; no LLM calls.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    summary, manifest = [], {"model": args.model, "n_nodes": args.n_nodes,
                             "chunk_size": args.chunk_size, "datasets": {}}
    pooled_pred, pooled_gt = [], []

    for ds in args.datasets:
        print(f"\n### {ds}: building synset universe ({args.n_nodes} cluster nodes)")
        G = synset_graph(ds)
        gt_clusters, term_to_cluster, terms, n_amb = build_universe(G, args.n_nodes)
        multi = [c for c in gt_clusters.values() if len(c) > 1]
        print(f"    {len(gt_clusters)} clusters | {len(multi)} multi-member "
              f"| {len(terms)} exploded terms | {n_amb} ambiguous terms dropped")
        if not multi:
            print(f"    [!] {ds}: no multi-member synsets -- nothing to reconstruct, skipping")
            continue
        if args.dry_run:
            from our_method import build_prompt
            p = build_prompt(terms[0], terms[1:4], variant="full")
            print(f"    sample prompt head: {p[:120]!r}")
            continue

        t0 = time.time()
        raw = extract_raw_edges(terms, client, args.model, args.chunk_size, args.max_retries)
        runtime = time.time() - t0
        pred_clusters, n_recip = mutual_components(raw)

        write_csv(os.path.join(args.out_dir, f"raw_edges_{ds}.csv"),
                  [{"parent": u, "child": v, "mutual": int(raw.has_edge(v, u))}
                   for u, v in sorted(raw.edges())], ["parent", "child", "mutual"])

        gt_rows = gt_cluster_report(gt_clusters, pred_clusters)
        write_csv(os.path.join(args.out_dir, f"gt_clusters_{ds}.csv"), gt_rows,
                  ["cluster", "size", "members", "status", "n_pred_components"])
        pr_rows = pred_cluster_report(pred_clusters, term_to_cluster)
        write_csv(os.path.join(args.out_dir, f"pred_clusters_{ds}.csv"), pr_rows,
                  ["size", "members", "n_gt_clusters_spanned", "gt_clusters"])

        s = score_partition(pred_clusters, list(gt_clusters.values()))
        by_status = defaultdict(int)
        for r in gt_rows:
            by_status[r["status"]] += 1
        row = {"dataset": ds, **s,
               "clusters_multi": len(multi),
               "clusters_exact": by_status["exact"], "clusters_merged": by_status["merged"],
               "clusters_split": by_status["split"],
               "largest_pred_component": max(len(c) for c in pred_clusters),
               "reciprocal_pairs": n_recip, "n_terms": len(terms),
               "runtime_sec": round(runtime, 1)}
        summary.append(row)
        pooled_pred += pred_clusters
        pooled_gt += list(gt_clusters.values())
        manifest["datasets"][ds] = {"terms": len(terms), "clusters": len(gt_clusters),
                                    "multi_member": len(multi), "ambiguous_dropped": n_amb,
                                    "runtime_sec": round(runtime, 1)}
        print(f"    pairwise P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
              f"| exact {by_status['exact']}/{len(multi)} | merged {by_status['merged']} "
              f"| split {by_status['split']} | largest component {row['largest_pred_component']} "
              f"| {n_recip} reciprocal pairs | {runtime:.0f}s")

    if args.dry_run:
        print("\n[*] dry run complete -- nothing was called, nothing written.")
        return
    if not summary:
        raise SystemExit("[!] no dataset produced results.")

    # Term names are unique within each dataset run, so pooling the partitions
    # micro-averages the pairwise counts without cross-dataset collisions.
    pooled = score_partition(pooled_pred, pooled_gt)
    summary.append({"dataset": "POOLED", **pooled,
                    "clusters_multi": sum(r["clusters_multi"] for r in summary),
                    "clusters_exact": sum(r["clusters_exact"] for r in summary),
                    "clusters_merged": sum(r["clusters_merged"] for r in summary),
                    "clusters_split": sum(r["clusters_split"] for r in summary),
                    "largest_pred_component": max(r["largest_pred_component"] for r in summary),
                    "reciprocal_pairs": sum(r["reciprocal_pairs"] for r in summary),
                    "n_terms": sum(r["n_terms"] for r in summary), "runtime_sec": ""})

    cols = ["dataset", "precision", "recall", "f1", "pair_tp", "pair_pred", "pair_gt",
            "clusters_multi", "clusters_exact", "clusters_merged", "clusters_split",
            "largest_pred_component", "reciprocal_pairs", "n_terms", "runtime_sec"]
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary, cols)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    plot(summary, os.path.join(args.out_dir, "reconstruction.png"))
    print(f"\n[*] wrote summary.csv, per-dataset CSVs, manifest.json and reconstruction.png "
          f"to {args.out_dir}/")


if __name__ == "__main__":
    main()
