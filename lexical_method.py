import networkx as nx
from sentence_transformers import util
from data_manager import get_primary_term, to_lemma_format

def enforce_dag(G):
    G_dag = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = next(nx.simple_cycles(G_dag))
            G_dag.remove_edge(cycle[-1], cycle[0])
    except StopIteration:
        pass
    return G_dag

def cluster_synonyms_and_enforce_dag(G):
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

def method_lexical(nodes):
    G = nx.DiGraph()
    for u in nodes:
        for v in nodes:
            u_primary = get_primary_term(u)
            v_primary = get_primary_term(v)
            if u_primary != v_primary and len(u_primary) > 3:
                if u_primary in v_primary:
                    G.add_edge(u, v)
                    
    print(f"    [Lexical] SUCCESS | Built {G.number_of_edges()} raw edges via string containment.")
    return cluster_synonyms_and_enforce_dag(G)

def method_vector(nodes, encoder_model):
    G = nx.DiGraph()
    embeddings = encoder_model.encode(nodes, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j and cosine_scores[i][j] > 0.65:
                u, v = nodes[i], nodes[j]
                if len(u) < len(v):
                    G.add_edge(u, v)
                    
    print(f"    [Vector] SUCCESS | Built {G.number_of_edges()} raw edges via cosine similarity > 0.65.")
    return cluster_synonyms_and_enforce_dag(G)
