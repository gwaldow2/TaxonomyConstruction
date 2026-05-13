import json
import networkx as nx
from tqdm import tqdm
from sentence_transformers import util

def method_sbu_embedding(test_nodes, encoder_model, train_nodes=None):
    """
    Matches Section 3.3.3: Sentence Embedding Strategy
    Evaluates argmax cosine similarity without DAG enforcement.
    """
    G = nx.DiGraph()
    if not test_nodes: return G
    
    # If no training terms are provided to act as the parent candidate pool, 
    # the test nodes themselves form the candidate pool.
    candidate_parents = train_nodes if train_nodes else test_nodes
        
    child_embeddings = encoder_model.encode(test_nodes, convert_to_tensor=True)
    parent_embeddings = encoder_model.encode(candidate_parents, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(child_embeddings, parent_embeddings)
    
    for i, child in enumerate(test_nodes):
        # Section 3.3.3: "The parent term with the highest cosine similarity score was selected"
        best_parent_idx = cosine_scores[i].argmax().item()
        best_parent = candidate_parents[best_parent_idx]
        
        if child != best_parent:
            G.add_edge(best_parent, child)
            
    # Equation 15: Returns union of edges (handled inherently by DiGraph)
    return G

def method_sbu_batch(test_nodes, client, model_name, train_pairs=None, chunk_size=100):
    """
    Matches Section 3.3.2 and Figure 7: N x M Batch Prompting.
    """
    G = nx.DiGraph()
    if not test_nodes: return G
    
    # Partition test data into M disjoint chunks
    test_chunks = [test_nodes[i:i + chunk_size] for i in range(0, len(test_nodes), chunk_size)]
    
    # Partition training data into N disjoint chunks
    train_chunks = []
    if train_pairs:
        train_chunks = [train_pairs[i:i + chunk_size] for i in range(0, len(train_pairs), chunk_size)]
    else:
        train_chunks = [[]] # Fallback
        
    # Exact Prompt from Figure 7 of the SBU-NLP paper
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

    # Section 3.3.2: Pair each training chunk with each test chunk
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
                            G.add_edge(p, c)
                            
            except Exception:
                pass
            pbar.update(1)
            
    pbar.close()
    return G
