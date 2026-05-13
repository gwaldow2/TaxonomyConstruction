import networkx as nx
from tqdm import tqdm
import time
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

def method_our_approach(nodes, client, model_name, chunk_size=1000, max_retries=3):
    dag = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    
    for target_raw in tqdm(primary_nodes, desc="  -> [Our Method] O(N) Pairwise", leave=False):
        all_candidates = [t for t in primary_nodes if t != target_raw]
        
        # Split the candidates into chunks to prevent context window overflow
        for i in range(0, len(all_candidates), chunk_size):
            candidates_chunk = all_candidates[i:i + chunk_size]
            
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
            prompt = instructions + "\n".join([f"- {c}" for c in candidates_chunk])
            
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=4096, # Safely lowered from 16k to prevent HTTP 400 rejections
                        timeout=180.0
                    )
                    
                    # Intercept generation truncations before attempting to parse
                    if response.choices[0].finish_reason == 'length':
                        raise Exception("Response truncated due to max_tokens limit being reached.")
                        
                    response_text = response.choices[0].message.content.strip().lower()
                    safe_snippet = response_text[:80].replace('\n', ' ')
                    edges_added = 0

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
                                    edges_added += 1
                                    
                    # Only spam the console if it actually found edges, otherwise just break and move on.
                    # If you want to see 'none' responses too, remove the `if edges_added > 0:`
                    if edges_added > 0:
                        tqdm.write(f"    [Our Method] SUCCESS | Target '{target_raw}' | Parsed {edges_added} edges | Snippet: {safe_snippet}...")
                    
                    # Break the retry loop if parsing succeeds without errors
                    break 
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        tqdm.write(f"\n  [Our Method] API ERROR | Target '{target_raw}' (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in 2s...")
                        time.sleep(2)
                    else:
                        tqdm.write(f"\n  [Our Method] FAILED | Target '{target_raw}' against candidate chunk {i//chunk_size + 1} after {max_retries} attempts. Skipping chunk.")
            
    return cluster_synonyms_and_enforce_dag(dag)
