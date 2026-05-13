import json
import networkx as nx
from tqdm import tqdm
from sentence_transformers import util

def method_sbu_embedding(test_nodes, encoder_model, train_nodes=None):
    G = nx.DiGraph()
    if not test_nodes: return G
    
    candidate_parents = train_nodes if train_nodes else test_nodes
    child_embeddings = encoder_model.encode(test_nodes, convert_to_tensor=True)
    parent_embeddings = encoder_model.encode(candidate_parents, convert_to_tensor=True)
    cosine_scores = util.cos_sim(child_embeddings, parent_embeddings)
    
    for i, child in enumerate(test_nodes):
        best_parent_idx = cosine_scores[i].argmax().item()
        best_parent = candidate_parents[best_parent_idx]
        if child != best_parent:
            G.add_edge(best_parent, child)
            
    print(f"    [SBU Embedding] SUCCESS | Embedded {len(test_nodes)} children against {len(candidate_parents)} parents.")
    return G

def method_sbu_batch(test_nodes, client, model_name, train_pairs=None, chunk_size=100):
    G = nx.DiGraph()
    if not test_nodes: return G
    
    test_chunks = [test_nodes[i:i + chunk_size] for i in range(0, len(test_nodes), chunk_size)]
    train_chunks = []
    if train_pairs:
        train_chunks = [train_pairs[i:i + chunk_size] for i in range(0, len(train_pairs), chunk_size)]
    else:
        train_chunks = [[]]
        
    sys_prompt = """From this file, extract all parent and child relations for all
pairs like examples in JSON file.
Output file must be in this format:
[
{ "parent": "parent1", "child": "child1" },
{ "parent": "parent2", "child": "child2" },
{ "parent": "parent3", "child": "child3" },
{ "parent": "parent4", "child": "child4" }
]
You must find all parent-child pairs from the input file.
Each pair should be extracted and formatted as shown above."""

    total_prompts = len(train_chunks) * len(test_chunks)
    pbar = tqdm(total=total_prompts, desc="  -> [SBU Batch] N x M Chunking", leave=False)

    for t_chunk in train_chunks:
        train_json_str = json.dumps(t_chunk) if t_chunk else "[]"
        
        for s_chunk in test_chunks:
            user_content = f"JSON file examples:\n{train_json_str}\n\nInput file terms:\n" + "\n".join(s_chunk)
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    timeout=180.0
                )
                
                if not response.choices[0].message.content:
                    print(f"\n    [SBU Batch] API returned empty content.")
                    pbar.update(1)
                    continue
                    
                response_text = response.choices[0].message.content.strip()
                safe_snippet = response_text[:80].replace('\n', ' ')

                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx+1]
                    try:
                        pairs = json.loads(json_str)
                        edges_added = 0
                        for item in pairs:
                            if isinstance(item, dict):
                                p = str(item.get("parent", "")).strip().lower()
                                c = str(item.get("child", "")).strip().lower()
                                if p and c and p != c:
                                    G.add_edge(p, c)
                                    edges_added += 1
                                    
                        tqdm.write(f"    [SBU Batch] SUCCESS | Parsed {edges_added} edges | Snippet: {safe_snippet}...")
                    except json.JSONDecodeError as e:
                        tqdm.write(f"    [SBU Batch] PARSE ERROR | {e} | Snippet: {safe_snippet}...")
                else:
                    tqdm.write(f"    [SBU Batch] FORMAT ERROR (No Array) | Snippet: {safe_snippet}...")
                            
            except Exception as e:
                tqdm.write(f"    [SBU Batch] API ERROR | {e}")
                
            pbar.update(1)
            
    pbar.close()
    return G
