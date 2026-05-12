import re
import json
import networkx as nx
from tqdm import tqdm
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

def f1_token_overlap(text1, text2):
    tokens1 = re.findall(r'\w+', str(text1).lower())
    tokens2 = re.findall(r'\w+', str(text2).lower())
    common = set(tokens1) & set(tokens2)
    if not tokens1 or not tokens2:
        return 0.0
    precision = len(common) / len(tokens2)
    recall = len(common) / len(tokens1)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def method_sbu_ensemble(nodes, encoder_model):
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    
    overlap_parents = {}
    for child in primary_nodes:
        best_match = None
        best_score = 0.0
        for parent in primary_nodes:
            if child == parent: continue
            score = f1_token_overlap(child, parent)
            if score > best_score:
                best_score = score
                best_match = parent
        if best_match and best_score > 0:
            overlap_parents[child] = best_match

    embeddings = encoder_model.encode(primary_nodes, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    for i, child in enumerate(primary_nodes):
        actual_child = primary_to_full_map[child]
        
        if child in overlap_parents:
            actual_parent = primary_to_full_map[overlap_parents[child]]
            if actual_child != actual_parent:
                G.add_edge(actual_parent, actual_child)
        else:
            best_match_idx = -1
            best_score = -1.0
            for j, parent in enumerate(primary_nodes):
                if i == j: continue
                score = cosine_scores[i][j].item()
                if score > best_score:
                    best_score = score
                    best_match_idx = j
            if best_match_idx != -1:
                actual_parent = primary_to_full_map[primary_nodes[best_match_idx]]
                if actual_child != actual_parent:
                    G.add_edge(actual_parent, actual_child)
                    
    return cluster_synonyms_and_enforce_dag(G)

def method_sbu_batch(nodes, client, model_name, num_chunks=2):
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    
    total_items = len(primary_nodes)
    if total_items == 0: return G
    
    num_chunks = min(num_chunks, total_items)
    base_chunk_size = total_items // num_chunks
    remainder = total_items % num_chunks

    chunks = []
    start_idx = 0
    for i in range(num_chunks):
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        chunks.append(primary_nodes[start_idx:end_idx])
        start_idx = end_idx

    sys_prompt = """From this file, extract all parent and child relations for all pairs like examples in JSON file.
    The relationship is defined as follows:
    If every entity labeled with 'child1' could logically also be labeled with 'parent1', you would output { "parent": "parent1", "child": "child1" }"
Output file must be in this format:
[
{ "parent": "parent1", "child": "child1" },
{ "parent": "parent2", "child": "child2" }
]
You must find all parent-child pairs from the input file.
Each pair should be extracted and formatted as shown above.
You should find pairs in [PAIR] tag terms.
Example: { "parent": "cell", "child": "anucleate" }, but the inverse is not necessarily true"""

    for chunk in tqdm(chunks, desc="  -> [SBU Batch] Chunking Execution", leave=False):
        user_content = "[PAIR]\n" + "\n".join(chunk) + "\n[/PAIR]"
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,
                max_tokens=16328
            )
            response_text = response.choices[0].message.content.strip()

            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                pairs = json.loads(json_str)
                for item in pairs:
                    p = str(item.get("parent", "")).strip().lower()
                    c = str(item.get("child", "")).strip().lower()
                    
                    if p and c and p != c:
                        if p in primary_to_full_map and c in primary_to_full_map:
                            G.add_edge(primary_to_full_map[p], primary_to_full_map[c])
                            
        except Exception as e:
            print(f"SBU Batch Error: {e}")
            
    return cluster_synonyms_and_enforce_dag(G)
