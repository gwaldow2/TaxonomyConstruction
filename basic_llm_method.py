import json
import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Tuple
from data_manager import get_primary_term, to_lemma_format

class TaxonomyOutput(BaseModel):
    edges: List[Tuple[str, str]] = Field(
        description="A list of parent-child relationship pairs. Example: [['mammal', 'dog'], ['vehicle', 'car']]"
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

def method_llm_single_shot(nodes, client, model_name, reasoning_level='medium'):
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    vocab_string = ", ".join(primary_nodes)
    num_terms = len(primary_nodes)
    
    # 1. Dynamic Token Scaling
    rtpt_map = {'none': 0, 'low': 10, 'medium': 20, 'high': 40}
    rtpt = rtpt_map.get(reasoning_level.lower(), 20)
    
    reasoning_budget = num_terms * rtpt
    json_budget = (num_terms * 30) + 500  # Assume roughly O(N) edges + 500 token structural buffer
    
    base_instructions = f"""You are an expert ontologist building a hierarchical taxonomy.
You are given a vocabulary of {num_terms} terms.
Identify ALL direct parent-child relationships between these terms.
A parent is a broader concept, a child is a more specific concept.
ONLY use terms EXACTLY as they appear in the vocabulary list.

Vocabulary: [{vocab_string}]
"""

    try:
        reasoning_context = ""
        
        # STAGE 1: Free-form Reasoning
        if reasoning_budget > 0:
            reasoning_prompt = base_instructions + "\nThink step-by-step about the hierarchical groupings of these terms."
            
            res_reasoning = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=0.6,
                max_tokens=reasoning_budget
            )
            raw_thought = getattr(res_reasoning.choices[0].message, 'content', '') or ""
            reasoning_context = f"\nPreliminary Analysis:\n{raw_thought}\n"

        # STAGE 2: Constrained Extraction
        extraction_prompt = base_instructions + reasoning_context + f"\nBased on the analysis, output the final relationships matching this JSON schema:\n{TaxonomyOutput.schema_json()}"
        
        res_extract = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict JSON data extraction pipeline. Output raw JSON without markdown."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.0,
            max_tokens=json_budget,
            response_format={"type": "json_object"}
        )
       
        content = getattr(res_extract.choices[0].message, 'content', '') or ""
        
        if not content.strip():
            print(f"    [LLM Zero-Shot] FATAL: Empty JSON content returned.")
            return cluster_synonyms_and_enforce_dag(G)
           
        parsed_data = json.loads(content.strip())
        edges_list = parsed_data.get("edges", [])
        
        edges_added = 0
        for pair in edges_list:
            if len(pair) == 2:
                parent_raw = str(pair[0]).strip().lower()
                child_raw = str(pair[1]).strip().lower()
                
                if parent_raw in primary_to_full_map and child_raw in primary_to_full_map and parent_raw != child_raw:
                    G.add_edge(primary_to_full_map[parent_raw], primary_to_full_map[child_raw])
                    edges_added += 1
                    
        print(f"    [LLM Zero-Shot] SUCCESS | Scaled for {num_terms} terms | Reasoned: {reasoning_budget}t | Extracted: {edges_added} edges.")
            
    except Exception as e:
        print(f"    [LLM Zero-Shot] EXCEPTION | {e}")
       
    return cluster_synonyms_and_enforce_dag(G)
