"""Graph post-processing: synonym clustering and DAG enforcement.

The raw LLM output is a directed graph that may contain cycles and bidirectional
("X is-a Y" *and* "Y is-a X") edges. Bidirectional edges are treated as synonymy:
their endpoints are merged into a single cluster node. Remaining cycles are broken
to yield a directed acyclic graph (a valid taxonomy).
"""

import networkx as nx

from .text import to_lemma_format

__all__ = ["enforce_dag", "cluster_synonyms_and_enforce_dag"]


def enforce_dag(G):
    """Return a copy of ``G`` with cycles removed (one edge per cycle, until acyclic)."""
    G_dag = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = next(nx.simple_cycles(G_dag))
            G_dag.remove_edge(cycle[-1], cycle[0])
    except StopIteration:
        pass
    return G_dag


def cluster_synonyms_and_enforce_dag(G):
    """Merge mutually-pointing nodes into synonym clusters, then enforce a DAG.

    1. Find bidirectional edges (u->v and v->u) and treat them as synonymy links.
    2. Contract each connected component of those links into one lemma-format node.
    3. Re-add the remaining edges between distinct clusters and break any cycles.
    """
    bidirectional_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    G_sym = nx.Graph()
    G_sym.add_nodes_from(G.nodes())
    G_sym.add_edges_from(bidirectional_edges)
    clusters = list(nx.connected_components(G_sym))

    condensed_dag = nx.DiGraph()
    node_mapping = {}
    for cluster in clusters:
        new_node_name = to_lemma_format(sorted(list(cluster)))
        condensed_dag.add_node(new_node_name)
        for node in cluster:
            node_mapping[node] = new_node_name

    for u, v in G.edges():
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        if new_u != new_v:
            condensed_dag.add_edge(new_u, new_v)

    return enforce_dag(condensed_dag)
