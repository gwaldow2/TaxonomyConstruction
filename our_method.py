import networkx as nx
from tqdm import tqdm
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

def method_our_approach(nodes, client, model_name):
    dag = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    
    for target_raw in tqdm(primary_nodes, desc="  -> [Our Method] O(N) Pairwise", leave=False):
        candidates = [t for t in primary_nodes if t != target_raw]
        
        instructions = (
            f"You are identifying hierarchical relationships for the target entity: '{target_raw}'.\n"
            f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
            f"between the target and the candidates.\n"
            f"- If every entity labeled with '{target_raw}' could logically also be labeled with a candidate 'C', output '{target_raw} <= C'\n"
            f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{target_raw}', output 'C <= {target_raw}'\n"
            f"ONLY output relationships involving '{target_raw}'. Do NOT output relationships between the candidates themselves. "
            f"Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
            f"Example: 'anucleate cell' <= 'cell', but the inverse is not necessarily true.\n"
            f"Candidates:\n"
        )
        prompt = instructions + "\n".join([f"- {c}" for c in candidates])
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16328
            )
            response_text = response.choices[0].message.content.strip().lower()

            for line in response_text.split('\n'):
                if '<=' in line:
                    parts = line.split('<=')
                    if len(parts) == 2:
                        sub_raw = parts[0].strip().strip("-'\" ")
                        sup_raw = parts[1].strip().strip("-'\" ")
                        
                        if sub_raw in primary_to_full_map and sup_raw in primary_to_full_map:
                            actual_sub = primary_to_full_map[sub_raw]
                            actual_sup = primary_to_full_map[sup_raw]
                            dag.add_edge(actual_sup, actual_sub)
        except Exception as e:
            pass 
            
    return cluster_synonyms_and_enforce_dag(dag)
