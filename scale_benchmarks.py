"""Generate additional benchmark scales (SUB25 / SUB50 / SUB200 / ...) for the
taxonomy-size scaling experiments, from the FROZEN full-scale benchmark graphs.

Provenance design
-----------------
- Samples from benchmark_sets/{name}_FULL.graphml -- the exact frozen evaluation
  universe every existing result was drawn from -- NEVER from live sources (the
  WordNet / OBO / LLMs4OL loaders can drift between fetches). taxonomy_metrics.py
  remains the only script that touches sources, and this script never rewrites
  {name}_SUB.graphml or {name}_FULL.graphml: it refuses those tokens by
  construction and skips any existing file unless --force is given.
- Every generated graphml carries its generation metadata as graph attributes
  (scale token, target_nodes, seed, source file and size, timestamp), and every
  generation is recorded in benchmark_sets/scale_manifest.csv, upserted by
  (dataset, scale).
- Naming: SUB{N} for the default seed 42, SUB{N}S{seed} otherwise (SUB200,
  SUB25S7). main.py accepts these via --scale sub200, and the dataset name seen
  by results rows, diagnostics, and saved prediction graphs becomes e.g.
  WordNetFood_SUB200 -- downstream provenance rides the same mechanism as every
  existing campaign, so runs at different scales can never collide.
- The legacy SUB (100-node, seed 42) benchmarks stay untouched and remain the
  N=100 point of the scaling curve. Generating --target_nodes 100 would create a
  distinct SUB100 file that differs from SUB only through node iteration order;
  reuse SUB instead of spending compute on it.
- Same-seed scales from this script are nested in one another (the sampler
  shuffles the node list once per seed and a smaller target stops earlier along
  the same order). Use --seeds for independent variance estimates at small N;
  use the default seed for the scaling curve. The legacy SUB was sampled from
  the same universe but with a different node ordering, so it is NOT exactly
  nested within SUB200.

One deliberate difference from the original SUB generation: FULL was saved after
enforce_dag, so the universe here has cycles already broken and may carry a
virtual_root. The virtual root (and its incident edges) is stripped before
sampling and re-added by enforce_dag only if the sample is multi-rooted, matching
how SUB was built from the pre-DAG test graph.
"""

import os
import csv
import glob
import argparse
from datetime import datetime

import networkx as nx

import data_manager as dm
from data_manager import (load_benchmark_graph, save_benchmark_graph,
                          get_closed_subgraph, enforce_dag, scale_token)

MANIFEST_FIELDS = ["dataset", "scale", "target_nodes", "seed", "nodes", "edges",
                   "roots", "leaves", "max_depth", "edge_node_ratio",
                   "source_nodes", "source_edges", "train_pairs", "generated"]


def manifest_path():
    return os.path.join(dm.BENCHMARK_DIR, "scale_manifest.csv")


def strip_virtual_root(G):
    H = G.copy()
    for n, d in list(H.nodes(data=True)):
        if n == "virtual_root" or d.get("is_virtual") in (True, "True", "true"):
            H.remove_node(n)
    return H


def light_metrics(G):
    """Topology stats for the manifest -- no LLM client, unlike taxonomy_metrics."""
    nodes, edges = G.number_of_nodes(), G.number_of_edges()
    return {
        "nodes": nodes,
        "edges": edges,
        "roots": len([n for n, d in G.in_degree() if d == 0]),
        "leaves": len([n for n, d in G.out_degree() if d == 0]),
        "max_depth": nx.dag_longest_path_length(G) if nodes else 0,
        "edge_node_ratio": round(edges / nodes, 3) if nodes else 0.0,
    }


def upsert_manifest(row, path=None):
    path = path or manifest_path()
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["dataset"], r["scale"])] = r
    rows[(row["dataset"], row["scale"])] = {k: str(row.get(k, "")) for k in MANIFEST_FIELDS}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def discover_datasets():
    """Every dataset with a frozen FULL benchmark graph."""
    paths = glob.glob(os.path.join(dm.BENCHMARK_DIR, "*_FULL.graphml"))
    return sorted(os.path.basename(p)[:-len("_FULL.graphml")] for p in paths)


def generate_one(name, target_nodes, seed, force=False):
    token = scale_token(target_nodes, seed)
    out_path = os.path.join(dm.BENCHMARK_DIR, f"{name}_{token}.graphml")
    if os.path.exists(out_path) and not force:
        print(f"  [skip] {name}_{token} exists (use --force to regenerate)")
        return None

    G_full, train_pairs = load_benchmark_graph(name, scale="FULL")
    if G_full is None:
        print(f"  [skip] {name}: no FULL benchmark graph in {dm.BENCHMARK_DIR} "
              f"(run taxonomy_metrics.py once to freeze it)")
        return None

    universe = strip_virtual_root(G_full)
    if universe.number_of_nodes() <= target_nodes:
        print(f"  [skip] {name}_{token}: universe has only {universe.number_of_nodes()} nodes "
              f"-- a {target_nodes}-node 'subsample' would be the whole graph")
        return None
    if target_nodes > 0.75 * universe.number_of_nodes():
        print(f"  [warn] {name}_{token}: sampling {target_nodes} of "
              f"{universe.number_of_nodes()} nodes ({100 * target_nodes / universe.number_of_nodes():.0f}% "
              f"of the universe) -- interpret as near-saturated, not an independent subsample")

    G_sub = enforce_dag(get_closed_subgraph(universe, target_nodes=target_nodes, seed=seed))
    # Same convention as the original SUB build: a train-pair slice proportional to
    # target size (SUB used 50 pairs for 100 nodes), to keep few-shot prompts bounded.
    sub_pairs = (train_pairs or [])[: target_nodes // 2]

    meta = {
        "scale": token, "target_nodes": target_nodes, "seed": seed,
        "source_file": f"{name}_FULL.graphml",
        "source_nodes": G_full.number_of_nodes(), "source_edges": G_full.number_of_edges(),
        "generator": "scale_benchmarks.py",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    save_benchmark_graph(G_sub, name, scale=token, train_pairs=sub_pairs, meta=meta)

    row = {"dataset": name, "scale": token, "target_nodes": target_nodes, "seed": seed,
           **light_metrics(G_sub),
           "source_nodes": G_full.number_of_nodes(), "source_edges": G_full.number_of_edges(),
           "train_pairs": len(sub_pairs), "generated": meta["generated"]}
    upsert_manifest(row)
    print(f"  [ok]   {name}_{token}: {row['nodes']} nodes, {row['edges']} edges, "
          f"depth {row['max_depth']} (universe {row['source_nodes']})")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target_nodes", nargs="+", type=int, required=True,
                    help="Scale sizes to generate, e.g. --target_nodes 25 50 200")
    ap.add_argument("--seeds", nargs="+", type=int, default=[dm.DEFAULT_SCALE_SEED],
                    help="Subsampling seeds (default 42 -> plain SUB{N} tokens; other seeds "
                         "-> SUB{N}S{seed}). Use several for variance estimates at small N.")
    ap.add_argument("--datasets", nargs="+", default=["all"],
                    help="Dataset names (default: every dataset with a frozen FULL benchmark)")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate scales whose graphml already exists")
    args = ap.parse_args()

    datasets = discover_datasets() if "all" in args.datasets else args.datasets
    if not datasets:
        raise SystemExit(f"[!] No FULL benchmark graphs found in {dm.BENCHMARK_DIR}")

    rows = []
    for name in datasets:
        print(f"\n--- {name} ---")
        for n in args.target_nodes:
            for seed in args.seeds:
                row = generate_one(name, n, seed, force=args.force)
                if row:
                    rows.append(row)

    print(f"\n[*] Generated {len(rows)} benchmark(s); manifest: {manifest_path()}")
    if rows:
        header = f"{'dataset':28s} {'scale':10s} {'nodes':>6s} {'edges':>6s} {'depth':>6s} {'e/n':>6s}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['dataset']:28s} {r['scale']:10s} {r['nodes']:>6d} {r['edges']:>6d} "
                  f"{r['max_depth']:>6d} {r['edge_node_ratio']:>6.3f}")


if __name__ == "__main__":
    main()
