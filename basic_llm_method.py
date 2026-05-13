import re
import json
import networkx as nx
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

def method_llm_single_shot(nodes, client, model_name):
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
   
    vocab_string = ", ".join(primary_nodes)
    
    prompt = f"""You are an expert ontologist building a hierarchical taxonomy.
You are given a vocabulary of {len(primary_nodes)} terms.
Your task is to identify ALL direct parent-child relationships between these terms.

Vocabulary: [{vocab_string}]

Rules:
1. ONLY use terms EXACTLY as they appear in the vocabulary list.
2. A parent is a broader concept, a child is a more specific concept (e.g., mammal -> dog).
3. Output your answer strictly as a JSON list of arrays.
4. Do not add any conversational text, explanations, or markdown outside the JSON.

Format Example:
[
  ["parent_term_1", "child_term_1"],
  ["parent_term_2", "child_term_2"]
]"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly outputs valid JSON data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4096,
            timeout=180.0
        )
       
        if not response.choices[0].message.content:
            print("    [LLM Zero-Shot] API returned empty content.")
            return cluster_synonyms_and_enforce_dag(G)
           
        ans = response.choices[0].message.content.strip()
        safe_snippet = ans[:80].replace('\n', ' ')
        
        start_idx = ans.find('[')
        end_idx = ans.rfind(']')
        if start_idx != -1 and end_idx != -1:
            json_str = ans[start_idx:end_idx+1]
            try:
                pairs = json.loads(json_str)
                edges_added = 0
                for pair in pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        parent_raw = str(pair[0]).strip().lower()
                        child_raw = str(pair[1]).strip().lower()
                       
                        if parent_raw in primary_to_full_map and child_raw in primary_to_full_map and parent_raw != child_raw:
                            actual_parent_node = primary_to_full_map[parent_raw]
                            actual_child_node = primary_to_full_map[child_raw]
                            G.add_edge(actual_parent_node, actual_child_node)
                            edges_added += 1
                            
                print(f"    [LLM Zero-Shot] SUCCESS | Parsed {edges_added} valid edges | Snippet: {safe_snippet}...")
            except json.JSONDecodeError as e:
                print(f"    [LLM Zero-Shot] PARSE ERROR | {e} | Snippet: {safe_snippet}...")
        else:
            print(f"    [LLM Zero-Shot] FORMAT ERROR (No Array) | Snippet: {safe_snippet}...")
                
    except Exception as e:
        print(f"    [LLM Zero-Shot] API ERROR | {e}")
       
    return cluster_synonyms_and_enforce_dag(G)
