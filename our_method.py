import networkx as nx
from tqdm import tqdm
import time
import json
import re
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

def method_our_approach(nodes, client, model_name, chunk_size=1000, max_retries=3, alt_prompt=False):
    dag = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    
    desc_label = "Our Method (alt. Prompt)" if alt_prompt else "Our Method"
    
    for target_raw in tqdm(primary_nodes, desc=f"  -> [{desc_label}] ChunkSize={chunk_size}", leave=False):
        all_candidates = [t for t in primary_nodes if t != target_raw]
        
        for i in range(0, len(all_candidates), chunk_size):
            candidates_chunk = all_candidates[i:i + chunk_size]
            
            if alt_prompt:
                vocab = [target_raw] + candidates_chunk
                vocab_string = ", ".join([f'"{v}"' for v in vocab])
                prompt = f"""You are an expert ontologist building a hierarchical taxonomy.
You are given a vocabulary of {len(vocab)} terms.
Identify ALL direct parent-child relationships between these terms.
A parent is a broader concept, a child is a more specific concept.
ONLY use terms EXACTLY as they appear in the vocabulary list.

Vocabulary: [{vocab_string}]

Format Example:
[
  ["parent_term_1", "child_term_1"],
  ["parent_term_2", "child_term_2"]
]

Output your answer strictly as a list of arrays. Do not add conversational text.
"""
            else:
                instructions = (
                    f"You are identifying hierarchical relationships for the target entity: '{target_raw}'.\n"
                    f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
                    f"between the target and the candidates.\n"
                    f"- If every entity labeled with '{target_raw}' could logically also be labeled with a candidate 'C', output '{target_raw} <= C'\n"
                    f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{target_raw}', output 'C <= {target_raw}'\n"
                    f"ONLY output relationships involving '{target_raw}'. Do NOT output relationships between the candidates themselves. "
                    f"Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
                    f"Example: 'anucleate cell' <= 'cell'\n"
                    f"Candidates:\n"
                )
                prompt = instructions + "\n".join([f"- {c}" for c in candidates_chunk]) + "\n\nRelationships:\n"
            
            for attempt in range(max_retries):
                try:
                    kwargs = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 16328
                    }

                    response = client.chat.completions.create(**kwargs)
                    
                    message = response.choices[0].message
                    content = getattr(message, 'content', '') or ""
                    reasoning = getattr(message, 'reasoning', '') or getattr(message, 'reasoning_content', '') or ""
                    
                    full_text = (str(reasoning) + "\n" + str(content)).strip()
                    
                    if not full_text:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        break
                        
                    edges_added = 0
                    
                    if alt_prompt:
                        # Extract the list of arrays using regex to bypass any leaked conversational text
                        match = re.search(r'\[\s*\[.*\]\s*\]', full_text, re.DOTALL)
                        json_str = match.group(0) if match else full_text
                        
                        try:
                            relationships = json.loads(json_str)
                            for pair in relationships:
                                if len(pair) == 2:
                                    sup_raw = pair[0].strip()
                                    sub_raw = pair[1].strip()
                                    
                                    if sub_raw in primary_to_full_map and sup_raw in primary_to_full_map:
                                        actual_sub = primary_to_full_map[sub_raw]
                                        actual_sup = primary_to_full_map[sup_raw]
                                        dag.add_edge(actual_sup, actual_sub)
                                        edges_added += 1
                        except json.JSONDecodeError:
                            if attempt < max_retries - 1:
                                time.sleep(2)
                                continue
                    else:
                        full_text_lower = full_text.lower()
                        for line in full_text_lower.split('\n'):
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
                                
                    break 
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        tqdm.write(f"  [{desc_label}] FAILED | Target '{target_raw}' after {max_retries} attempts. {e}")
            
    return cluster_synonyms_and_enforce_dag(dag)
