import json
import networkx as nx
from tqdm import tqdm
import time
from pydantic import BaseModel, Field
from typing import List, Dict
from data_manager import get_primary_term, to_lemma_format

class NodeRelationships(BaseModel):
    relationships: List[Dict[str, str]] = Field(
        description="List of discovered edges. Keys must be 'superclass' and 'subclass'."
    )

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

def method_our_approach(nodes, client, model_name, chunk_size=1000, max_retries=3, reasoning_level='medium'):
    dag = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    
    rtpt_map = {'none': 0, 'low': 10, 'medium': 20, 'high': 40}
    rtpt = rtpt_map.get(reasoning_level.lower(), 20)
    
    for target_raw in tqdm(primary_nodes, desc="  -> [Our Method] O(N) Pairwise", leave=False):
        all_candidates = [t for t in primary_nodes if t != target_raw]
        
        for i in range(0, len(all_candidates), chunk_size):
            candidates_chunk = all_candidates[i:i + chunk_size]
            current_chunk_size = len(candidates_chunk)
            
            # Dynamic token calculation specific to this chunk
            reasoning_budget = current_chunk_size * rtpt
            json_budget = (current_chunk_size * 25) + 500
            
            base_prompt = (
                f"Identify hierarchical relationships involving the Target Entity: '{target_raw}'.\n"
                f"Determine if the Target Entity is a subclass or superclass of any Candidate Entities.\n\n"
                f"Candidate Entities:\n" + "\n".join([f"- {c}" for c in candidates_chunk])
            )
            
            for attempt in range(max_retries):
                try:
                    reasoning_context = ""
                    
                    # STAGE 1: Reasoning
                    if reasoning_budget > 0:
                        res_reasoning = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a structural ontologist. Think step-by-step."},
                                {"role": "user", "content": base_prompt}
                            ],
                            temperature=0.6,
                            max_tokens=reasoning_budget
                        )
                        raw_thought = getattr(res_reasoning.choices[0].message, 'content', '') or ""
                        reasoning_context = f"\n\nPreliminary Analysis:\n{raw_thought}\n"

                    # STAGE 2: Pydantic Extraction
                    sys_extract = f"Extract relationships involving '{target_raw}'. Output MUST strictly match this JSON schema:\n{NodeRelationships.schema_json()}"
                    
                    res_extract = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": sys_extract},
                            {"role": "user", "content": base_prompt + reasoning_context}
                        ],
                        temperature=0.0,
                        max_tokens=json_budget,
                        response_format={"type": "json_object"}
                    )
                    
                    content = getattr(res_extract.choices[0].message, 'content', '') or ""
                    if not content.strip():
                        raise Exception("Empty JSON content returned.")

                    parsed_data = json.loads(content.strip())
                    relationships = parsed_data.get("relationships", [])
                    
                    edges_added = 0
                    for item in relationships:
                        sup = item.get("superclass", "").strip().lower()
                        sub = item.get("subclass", "").strip().lower()
                        
                        if (sup == target_raw or sub == target_raw) and sup in primary_to_full_map and sub in primary_to_full_map and sup != sub:
                            dag.add_edge(primary_to_full_map[sup], primary_to_full_map[sub])
                            edges_added += 1
                            
                    if edges_added > 0:
                        tqdm.write(f"    [Our Method] SUCCESS | Target '{target_raw}' | Reasoned: {reasoning_budget}t | Extracted: {edges_added} edges.")
                    
                    break # Success, break retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        tqdm.write(f"  [Our Method] FAILED | Target '{target_raw}' after {max_retries} attempts. {e}")
            
    return cluster_synonyms_and_enforce_dag(dag)
