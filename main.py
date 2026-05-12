import os
import argparse
import gc
import torch
import pandas as pd
import networkx as nx
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from data_manager import (
    get_semeval_graph, get_wordnet_food_graph, get_google_products_graph,
    get_geonames_graph, get_gene_ontology_graph, get_cell_ontology_graph,
    get_csv_graph, get_closed_subgraph, enforce_dag
)
from evaluator import evaluate_all_modes, update_benchmark_results
from lexical_method import method_lexical, method_vector
from basic_llm_method import method_llm_single_shot
from our_method import method_our_approach
from taxollama_method import precompute_taxollama_ppl, build_taxollama_graph
from SBU_NLP_method import method_sbu_batch, method_sbu_ensemble

def display_summary_table(domain, eval_results):
    data = []
    for method, metrics in eval_results.items():
        row = {
            "Method": method,
            "Cond_Red (F1)": f"{metrics['Cond_Red']['F1']:.4f}",
            "Cond_Clos (F1)": f"{metrics['Cond_Clos']['F1']:.4f}",
            "Exp_Raw (F1)": f"{metrics['Exp_Raw']['F1']:.4f}",
            "Exp_Clos (F1)": f"{metrics['Exp_Clos']['F1']:.4f}"
        }
        data.append(row)
        
    df = pd.DataFrame(data).set_index("Method")
    print(f"\n{'='*70}")
    print(f"=== SUMMARY: {domain} ===")
    print(f"{'='*70}")
    print(df.to_string())
    print(f"{'='*70}\n")

def main(args):
    os.makedirs("./results", exist_ok=True)
    
    client = None
    vector_encoder = None
    taxo_model = None
    taxo_tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_methods = args.method
    if "all" in selected_methods:
        selected_methods = ["lexical", "vector", "llm_zero", "our_method", "taxollama", "sbu_batch", "sbu_ensemble"]

    if any(m in selected_methods for m in ["llm_zero", "our_method", "sbu_batch"]):
        print("Initializing LLM Client...")
        MODEL_NAME = "openai/gpt-oss-120b"
        client = OpenAI(
            base_url="http://localhost:8000/v1", 
            api_key="woohoo"
        )
        
    if any(m in selected_methods for m in ["vector", "sbu_embedding", "sbu_ensemble"]):
        print("Initializing Vector Encoder...")
        vector_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    if "taxollama" in selected_methods:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print("Initializing TaxoLLaMA (Base Llama-2 + Adapter)...")
        
        base_model_id = "meta-llama/Llama-2-7b-hf"
        adapter_id = "VityaVitalich/TaxoLLaMA"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        taxo_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=True)
        if taxo_tokenizer.pad_token is None:
            taxo_tokenizer.pad_token = taxo_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            token=True
        )

        print(f"Applying adapter weights from {adapter_id}...")
        taxo_model = PeftModel.from_pretrained(base_model, adapter_id)
        taxo_model.eval()

    selected_ds = args.datasets
    if "all" in selected_ds:
        selected_ds = ["WordNetFood", "GoogleProducts", "GeoNames", "GeneOntology", "CellOntology", "csv", "SemEvalFood", "SemEvalScience", "SemEvalEnvironment"]

    datasets = {}
    if "WordNetFood" in selected_ds: 
        datasets["WordNetFood"] = lambda: get_wordnet_food_graph(use_synsets=args.use_synsets)
    if "GoogleProducts" in selected_ds: 
        datasets["GoogleProducts"] = lambda: get_google_products_graph(use_synsets=args.use_synsets)
    if "GeoNames" in selected_ds: 
        datasets["GeoNames"] = lambda: get_geonames_graph(use_synsets=args.use_synsets)
    if "GeneOntology" in selected_ds: 
        datasets["GeneOntology"] = lambda: get_gene_ontology_graph(use_synsets=args.use_synsets)
    if "CellOntology" in selected_ds: 
        datasets["CellOntology"] = lambda: get_cell_ontology_graph(use_synsets=args.use_synsets)
    if "csv" in selected_ds and args.csv_dataset:
        datasets["CustomCSV"] = lambda: get_csv_graph(args.csv_dataset)
        
    if "SemEvalFood" in selected_ds:
        datasets["SemEvalFood"] = lambda: get_semeval_graph("SemEvalFood", use_synsets=args.use_synsets)
    if "SemEvalScience" in selected_ds:
        datasets["SemEvalScience"] = lambda: get_semeval_graph("SemEvalScience", use_synsets=args.use_synsets)
    if "SemEvalEnvironment" in selected_ds:
        datasets["SemEvalEnvironment"] = lambda: get_semeval_graph("SemEvalEnvironment", use_synsets=args.use_synsets)
        
    for domain, loader_func in datasets.items():
        G_full = loader_func()
        if G_full.number_of_nodes() == 0:
            continue
            
        if domain == "CustomCSV" and args.csv_dataset:
            base = os.path.basename(args.csv_dataset)
            actual_domain_name = os.path.splitext(base)[0]
        else:
            actual_domain_name = domain

        print(f"\nEvaluating Domain: {actual_domain_name} (Full Size: {G_full.number_of_nodes()})")
        
        target_size = G_full.number_of_nodes() if domain == "CustomCSV" else 100
        G_gt = get_closed_subgraph(G_full, target_nodes=target_size)
        G_gt = enforce_dag(G_gt)
        
        if args.explode_nodes:
            print("  -> Exploding cluster nodes for individual term analysis...")
            from evaluator import explode_graph
            G_eval_input = explode_graph(G_gt)
            input_nodes = list(G_eval_input.nodes())
            print(f"Extracted Subgraph: {G_eval_input.number_of_nodes()} exploded nodes, {G_eval_input.number_of_edges()} eval edges.")
            print(f"  (Ground Truth Condensed Graph has {G_gt.number_of_nodes()} nodes, {G_gt.number_of_edges()} edges)")
        else:
            G_eval_input = G_gt
            input_nodes = list(G_eval_input.nodes())
            print(f"Extracted Closed Subgraph: {G_eval_input.number_of_nodes()} nodes, {G_gt.number_of_edges()} GT edges.")
            
        nx.write_graphml(G_gt, f"./results/GT_{actual_domain_name}.graphml")
        eval_results = {}
        
        if "lexical" in selected_methods:
            print("  -> Running Lexical...")
            G_lex = method_lexical(input_nodes)
            eval_results["Lexical"] = evaluate_all_modes(G_lex, G_gt, f"./results/{actual_domain_name}_Lexical")
            
        if "vector" in selected_methods:
            print("  -> Running Vector...")
            G_vec = method_vector(input_nodes, vector_encoder)
            eval_results["Vector"] = evaluate_all_modes(G_vec, G_gt, f"./results/{actual_domain_name}_Vector")
            
        if "llm_zero" in selected_methods:
            print("  -> Running Zero-Shot LLM...")
            G_llm_zero = method_llm_single_shot(input_nodes, client, MODEL_NAME)
            eval_results["LLM Zero-Shot"] = evaluate_all_modes(G_llm_zero, G_gt, f"./results/{actual_domain_name}_LLMZero")

        if "our_method" in selected_methods:
            print("  -> Running Our Method...")
            G_our = method_our_approach(input_nodes, client, MODEL_NAME)
            eval_results["Our Method O(N)"] = evaluate_all_modes(G_our, G_gt, f"./results/{actual_domain_name}_OurMethod")

        if "taxollama" in selected_methods:
            print("  -> Precomputing TaxoLLaMA Scores...")
            ppl_cache = precompute_taxollama_ppl(input_nodes, taxo_model, taxo_tokenizer, device)
            
            thresh = 15.0
            G_taxo = build_taxollama_graph(input_nodes, ppl_cache, threshold=thresh, max_parents=1)
            method_name = f"TaxoLLaMA (PPL<{thresh})"
            safe_file_name = f"./results/{actual_domain_name}_TaxoLLaMA_PPL{thresh}"
            eval_results[method_name] = evaluate_all_modes(G_taxo, G_gt, safe_file_name)

        if "sbu_batch" in selected_methods:
            print("  -> Running SBU LLM Batch Prompting...")
            G_sbu_batch = method_sbu_batch(input_nodes, client, MODEL_NAME)
            eval_results["SBU LLM Batch"] = evaluate_all_modes(G_sbu_batch, G_gt, f"./results/{actual_domain_name}_SBU_Batch")

        if "sbu_ensemble" in selected_methods:
            print("  -> Running SBU Ensemble (Overlap + Embedding)...")
            G_sbu_ens = method_sbu_ensemble(input_nodes, vector_encoder)
            eval_results["SBU Ensemble"] = evaluate_all_modes(G_sbu_ens, G_gt, f"./results/{actual_domain_name}_SBU_Ensemble")

        if eval_results:
            display_summary_table(actual_domain_name, eval_results)
            
            print(f"Saving benchmark results to JSON...")
            for method_name, metrics in eval_results.items():
                flat_metrics = {
                    "Cond_Red_F1": metrics["Cond_Red"]["F1"],
                    "Cond_Clos_F1": metrics["Cond_Clos"]["F1"],
                    "Exp_Raw_F1": metrics["Exp_Raw"]["F1"],
                    "Exp_Clos_F1": metrics["Exp_Clos"]["F1"]
                }
                update_benchmark_results(
                    dataset_name=actual_domain_name,
                    method_name=method_name, 
                    metrics_dict=flat_metrics, 
                    use_synsets=args.use_synsets, 
                    explode_nodes=args.explode_nodes
                )
            print("Save complete.")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy Extraction Benchmark")
    parser.add_argument("--method", nargs="+", default=["all"], choices=["all", "lexical", "vector", "llm_zero", "our_method", "taxollama", "sbu_batch", "sbu_ensemble"], help="List of methods to run (space separated)")
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", "csv", "WordNetFood", "GoogleProducts", "GeoNames", "GeneOntology", "CellOntology", "SemEvalFood", "SemEvalScience", "SemEvalEnvironment"], help="List of datasets to run (space separated)")
    parser.add_argument("--use_synsets", action="store_true", help="Include synonyms for nodes when available")
    parser.add_argument("--csv_dataset", type=str, default=None, help="Path to Gephi Adjacency CSV to use as dataset")
    parser.add_argument("--explode_nodes", action="store_true", help="Explode clustered nodes into individual evaluation nodes")
    args = parser.parse_args()
    
    main(args)
