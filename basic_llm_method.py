import re
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
Identify ALL direct parent-child relationships between these terms.
A parent is a broader concept, a child is a more specific concept.
ONLY use terms EXACTLY as they appear in the vocabulary list.

Vocabulary: [{vocab_string}]

Format Example:
[
  ["parent_term_1", "child_term_1"],
  ["parent_term_2", "child_term_2"]
]

Output your answer strictly as a JSON list of arrays. Do not add any conversational text.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=16328
        )
       
        message = response.choices[0].message
        # Safely grab content even if the API returns None or puts it in a weird field
        content = getattr(message, 'content', '') or ""
        reasoning = getattr(message, 'reasoning', '') or getattr(message, 'reasoning_content', '') or ""
        
        full_text = (str(reasoning) + "\n" + str(content)).strip()
        
        if not full_text:
            print(f"    [LLM Zero-Shot] FATAL: API returned completely empty content. Finish Reason: {response.choices[0].finish_reason}")
            return cluster_synonyms_and_enforce_dag(G)
           
        # Brute-force Regex Extraction
        pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        
        edges_added = 0
        for p, c in matches:
            parent_raw = p.strip().lower()
            child_raw = c.strip().lower()
            
            if parent_raw in primary_to_full_map and child_raw in primary_to_full_map and parent_raw != child_raw:
                actual_parent_node = primary_to_full_map[parent_raw]
                actual_child_node = primary_to_full_map[child_raw]
                G.add_edge(actual_parent_node, actual_child_node)
                edges_added += 1
                
        safe_snippet = full_text[:100].replace('\n', ' ')
        if edges_added > 0:
            print(f"    [LLM Zero-Shot] SUCCESS | Parsed {edges_added} valid edges | Snippet: {safe_snippet}...")
        else:
            print(f"    [LLM Zero-Shot] ZERO EDGES FOUND | Snippet: {safe_snippet}...")
                
    except Exception as e:
        print(f"    [LLM Zero-Shot] EXCEPTION | {e}")
       
    return cluster_synonyms_and_enforce_dag(G)
