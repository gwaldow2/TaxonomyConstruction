import os
import math
import random
import argparse
import networkx as nx
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from data_manager import (
    enforce_dag, get_wordnet_food_graph, get_google_products_graph,
    get_geonames_graph, get_gene_ontology_graph, get_cell_ontology_graph,
    get_semeval_graph, get_csv_graph, get_llms4ol_task_c_data, 
    get_closed_subgraph, save_benchmark_graph, DATA_DIR
)

# ==========================================
# 1. METRICS COMPUTATION
# ==========================================

def compute_hearst_ppl(G, client, model_name, sample_size=100):
    if G.number_of_edges() == 0 or client is None:
        return 0.0
        
    edges = [e for e in G.edges() if e[0] != 'virtual_root' and e[1] != 'virtual_root']
    if len(edges) > sample_size:
        random.seed(42)
        edges = random.sample(edges, sample_size)
        
    total_ppl = 0.0
    valid_edges = 0
    
    for parent, child in tqdm(edges, desc="  -> Computing Hearst PPL", leave=False):
        prompt = f"A {child} is a type of {parent}."
        try:
            response = client.completions.create(
                model=model_name, prompt=prompt, max_tokens=1, echo=True, logprobs=1
            )
            logprobs = response.choices[0].logprobs.token_logprobs[:-1]
            valid_logprobs = [lp for lp in logprobs if lp is not None]
            
            if valid_logprobs:
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                total_ppl += math.exp(-avg_logprob)
                valid_edges += 1
        except Exception:
            pass
            
    if valid_edges == 0:
        return 0.0
    return total_ppl / valid_edges

def compute_metrics(G, name, client, model_name):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    
    if nodes == 0:
        return {"Dataset": name, "Nodes": 0, "Edges": 0, "Roots": 0, "Leaves": 0, "Components": 0, 
                "Max Depth": 0, "Avg Branching": 0.0, "Tree-likeness": 0.0, "Edge/Node Ratio": 0.0, 
                "Lexical Overlap": 0.0, "Redundancy Ratio": 0.0, "Avg Hearst PPL": 0.0}
        
    depth = nx.dag_longest_path_length(G)
    roots = len([n for n, d in G.in_degree() if d == 0])
    leaves = len([n for n, d in G.out_degree() if d == 0])
    components = nx.number_weakly_connected_components(G)
    edge_node_ratio = edges / nodes if nodes > 0 else 0
        
    out_degrees = [d for n, d in G.out_degree() if d > 0]
    branching_factor = sum(out_degrees) / len(out_degrees) if out_degrees else 0
    
    in_degrees = [d for n, d in G.in_degree() if d > 0]
    single_parents = [d for d in in_degrees if d == 1]
    tree_likeness = len(single_parents) / len(in_degrees) if in_degrees else 0
    
    lexical_matches = 0
    for u, v in G.edges():
        if u == 'virtual_root' or v == 'virtual_root': continue
        u_str, v_str = str(u), str(v)
        if (u_str in v_str) or (v_str in u_str):
            lexical_matches += 1
    lexical_overlap = lexical_matches / edges if edges > 0 else 0
    
    closure_edges = sum(len(nx.descendants(G, n)) for n in G.nodes())
    redundancy_ratio = edges / closure_edges if closure_edges > 0 else 0

    hearst_ppl = compute_hearst_ppl(G, client, model_name)

    return {
        "Dataset": name, "Nodes": nodes, "Edges": edges, "Roots": roots,
        "Leaves": leaves, "Components": components, "Edge/Node Ratio": round(edge_node_ratio, 3),
        "Max Depth": depth, "Avg Branching": round(branching_factor, 2),
        "Tree-likeness": round(tree_likeness, 3), "Lexical Overlap": round(lexical_overlap, 3),
        "Redundancy Ratio": round(redundancy_ratio, 4), "Avg Hearst PPL": round(hearst_ppl, 2)
    }

# ==========================================
# 2. MAIN PIPELINE (Stage 0 Graph Generation)
# ==========================================

def main(args):
    print("Initializing LLM Client...")
    MODEL_NAME = "openai/gpt-oss-120b"
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="woohoo")

    datasets = {
        "WordNetFood": {"loader": get_wordnet_food_graph, "is_llms4ol": False},
        "GoogleProducts": {"loader": get_google_products_graph, "is_llms4ol": False},
        "GeoNames": {"loader": get_geonames_graph, "is_llms4ol": False},
        "GeneOntology": {"loader": get_gene_ontology_graph, "is_llms4ol": False},
        "CellOntology": {"loader": get_cell_ontology_graph, "is_llms4ol": False},
        "SemEvalFood": {"loader": lambda: get_semeval_graph("SemEvalFood"), "is_llms4ol": False},
        "SemEvalScience": {"loader": lambda: get_semeval_graph("SemEvalScience"), "is_llms4ol": False},
        "SemEvalEnvironment": {"loader": lambda: get_semeval_graph("SemEvalEnvironment"), "is_llms4ol": False}
    }
    
    llms4ol_base_path = os.path.join(DATA_DIR, "TaskC-TaxonomyDiscovery")
    for ont in ["OBI", "MatOnto", "SWEET", "SchemaOrg", "PO", "DOID", "FoodOn", "PROCO"]:
        ont_path = os.path.join(llms4ol_base_path, ont)
        if os.path.exists(ont_path):
            datasets[f"LLMs4OL_{ont}"] = {"loader": lambda p=ont_path: get_llms4ol_task_c_data(p)[0], "is_llms4ol": True}
    
    if args.csv_dataset:
        base = os.path.basename(args.csv_dataset)
        actual_name = os.path.splitext(base)[0]
        datasets[actual_name] = {"loader": lambda: get_csv_graph(args.csv_dataset), "is_llms4ol": False}
    
    results = []
    
    for name, config in datasets.items():
        is_llms4ol = config.get("is_llms4ol", False)
        loader_func = config["loader"]
        
        G_raw = loader_func()
        if isinstance(G_raw, tuple): G_raw = G_raw[0]
        
        if G_raw.number_of_nodes() == 0:
            continue
            
        if is_llms4ol:
            # LLMs4OL evaluated on its full train hierarchy
            G_eval = G_raw.copy()
        else:
            target_size = G_raw.number_of_nodes() if name == getattr(args, "csv_dataset", "") else 100
            G_eval = get_closed_subgraph(G_raw, target_nodes=target_size)
            
        # Unify & Break Cycles
        G_eval_dag = enforce_dag(G_eval)
        
        # Save as Ground Truth Artifact for main.py
        save_benchmark_graph(G_eval_dag, name)
        
        metrics = compute_metrics(G_eval_dag, name, client, MODEL_NAME)
        results.append(metrics)
        print(f" -> Finished {name}: Nodes={metrics['Nodes']}, Edges={metrics['Edges']}, PPL={metrics['Avg Hearst PPL']}")
        
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*130)
        print("TAXONOMY DOMAIN CHARACTERISTICS")
        print("="*130)
        print(df.to_string(index=False))
        print("="*130)
        
        output_file = "dataset_metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[*] Metrics successfully saved to {output_file}")
    else:
        print("\n[!] No datasets were successfully loaded.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy Metrics Evaluation")
    parser.add_argument("--csv_dataset", type=str, default=None, help="Path to your custom edge list CSV file.")
    args = parser.parse_args()
    
    main(args)
