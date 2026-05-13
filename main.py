import os
import argparse
import gc
import time
import torch
import pandas as pd
import networkx as nx
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from data_manager import load_benchmark_graph
from evaluator import evaluate_all_modes, update_benchmark_results
from lexical_method import method_lexical, method_vector
from basic_llm_method import method_llm_single_shot
from our_method import method_our_approach
from taxollama_method import precompute_taxollama_ppl, build_taxollama_graph

# Import the corrected SBU methods
from SBU_NLP_method import method_sbu_batch, method_sbu_embedding

def display_summary_table(domain, eval_results):
    data = []
    for method, info in eval_results.items():
        metrics = info["metrics"]
        runtime = info["runtime"]
        row = {
            "Method": method,
            "Runtime (s)": f"{runtime:.1f}",
            "Cond_Red (F1)": f"{metrics['Cond_Red']['F1']:.4f}",
            "Cond_Clos (F1)": f"{metrics['Cond_Clos']['F1']:.4f}",
            "Exp_Raw (F1)": f"{metrics['Exp_Raw']['F1']:.4f}",
            "Exp_Clos (F1)": f"{metrics['Exp_Clos']['F1']:.4f}"
        }
        data.append(row)
        
    df = pd.DataFrame(data).set_index("Method")
    print(f"\n{'='*80}")
    print(f"=== SUMMARY: {domain} ===")
    print(f"{'='*80}")
    print(df.to_string())
    print(f"{'='*80}\n")

def main(args):
    os.makedirs("./results", exist_ok=True)
    
    client = None
    vector_encoder = None
    taxo_model = None
    taxo_tokenizer = None
    MODEL_NAME = "openai/gpt-oss-120b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_methods = args.method
    if "all" in selected_methods:
        selected_methods = ["lexical", "vector", "llm_zero", "our_method", "taxollama", "sbu_batch", "sbu_embedding"]

    if any(m in selected_methods for m in ["llm_zero", "our_method", "sbu_batch"]):
        print("Initializing LLM Client...")
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="woohoo")
        
    if any(m in selected_methods for m in ["vector", "sbu_embedding"]):
        print("Initializing Vector Encoder...")
        vector_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    if "taxollama" in selected_methods:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print("Initializing TaxoLLaMA (Base Llama-2 + Adapter)...")
        
        base_model_id = "meta-llama/Llama-2-7b-hf"
        adapter_id = "VityaVitalich/TaxoLLaMA"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        taxo_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=True)
        if taxo_tokenizer.pad_token is None:
            taxo_tokenizer.pad_token = taxo_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=quantization_config, device_map="auto", token=True
        )
        taxo_model = PeftModel.from_pretrained(base_model, adapter_id)
        taxo_model.eval()

    selected_ds = args.datasets
    if "all" in selected_ds:
        selected_ds = ["WordNetFood", "CellOntology", "SemEvalFood", "SemEvalScience", "SemEvalEnvironment", 
                       "LLMs4OL_OBI", "LLMs4OL_MatOnto", "LLMs4OL_SchemaOrg", "LLMs4OL_PO"]

    target_scale = args.scale.upper()
    print(f"\n--- Running Benchmark across {target_scale} datasets ---")
    print(f"--- Reasoning Level: {args.reasoning_level.upper()} ---")

    for domain in selected_ds:
        # Load the graph and the isolated 20% training pairs
        G_gt, train_pairs = load_benchmark_graph(domain, scale=target_scale)
        if G_gt is None:
            print(f"\n [!] Error: No GT graph found for {domain} at scale {target_scale}. Run taxonomy_metrics.py first.")
            continue
            
        dataset_name_eval = f"{domain}_{target_scale}"
        print(f"\nEvaluating Domain: {dataset_name_eval} (Benchmark Nodes: {G_gt.number_of_nodes()})")
        
        input_nodes = [n for n in G_gt.nodes() if n != "virtual_root"]
        
        # Calculate strict reasoning budget for SBU Batch Method
        rtpt_map = {'none': 0, 'low': 10, 'medium': 20, 'high': 40}
        sbu_budget = len(input_nodes) * rtpt_map.get(args.reasoning_level.lower(), 20)
        
        if args.explode_nodes:
            print("  -> Exploding cluster nodes for individual term analysis...")
            from evaluator import explode_graph
            G_gt = explode_graph(G_gt)
            input_nodes = [n for n in G_gt.nodes() if n != "virtual_root"]

        # Strip virtual root from GT permanently for strict edge evaluation
        if "virtual_root" in G_gt:
            G_gt.remove_node("virtual_root")
            
        nx.write_graphml(G_gt, f"./results/GT_{dataset_name_eval}_eval.graphml")
        eval_results = {}
        
        if "lexical" in selected_methods:
            print("  -> Running Lexical...")
            t0 = time.time()
            G_lex = method_lexical(input_nodes)
            if "virtual_root" in G_lex: G_lex.remove_node("virtual_root")
            eval_results["Lexical"] = {
                "metrics": evaluate_all_modes(G_lex, G_gt, f"./results/{dataset_name_eval}_Lexical"), 
                "runtime": time.time() - t0
            }
            
        if "vector" in selected_methods:
            print("  -> Running Vector...")
            t0 = time.time()
            G_vec = method_vector(input_nodes, vector_encoder)
            if "virtual_root" in G_vec: G_vec.remove_node("virtual_root")
            eval_results["Vector"] = {
                "metrics": evaluate_all_modes(G_vec, G_gt, f"./results/{dataset_name_eval}_Vector"), 
                "runtime": time.time() - t0
            }
            
        if "llm_zero" in selected_methods:
            print("  -> Running Zero-Shot LLM...")
            t0 = time.time()
            G_llm_zero = method_llm_single_shot(input_nodes, client, MODEL_NAME, reasoning_level=args.reasoning_level)
            if "virtual_root" in G_llm_zero: G_llm_zero.remove_node("virtual_root")
            eval_results["LLM Zero-Shot"] = {
                "metrics": evaluate_all_modes(G_llm_zero, G_gt, f"./results/{dataset_name_eval}_LLMZero"), 
                "runtime": time.time() - t0
            }

        if "our_method" in selected_methods:
            print("  -> Running Our Method...")
            t0 = time.time()
            G_our = method_our_approach(input_nodes, client, MODEL_NAME, reasoning_level=args.reasoning_level)
            if "virtual_root" in G_our: G_our.remove_node("virtual_root")
            eval_results["Our Method O(N)"] = {
                "metrics": evaluate_all_modes(G_our, G_gt, f"./results/{dataset_name_eval}_OurMethod"), 
                "runtime": time.time() - t0
            }

        if "taxollama" in selected_methods:
            print("  -> Precomputing TaxoLLaMA Scores...")
            t0 = time.time()
            ppl_cache = precompute_taxollama_ppl(input_nodes, taxo_model, taxo_tokenizer, device)
            thresh = 15.0
            G_taxo = build_taxollama_graph(input_nodes, ppl_cache, threshold=thresh, max_parents=1)
            if "virtual_root" in G_taxo: G_taxo.remove_node("virtual_root")
            method_name = f"TaxoLLaMA (PPL<{thresh})"
            eval_results[method_name] = {
                "metrics": evaluate_all_modes(G_taxo, G_gt, f"./results/{dataset_name_eval}_TaxoLLaMA_PPL{thresh}"), 
                "runtime": time.time() - t0
            }

        if "sbu_batch" in selected_methods:
            print("  -> Running SBU LLM Batch Prompting...")
            t0 = time.time()
            G_sbu_batch = method_sbu_batch(input_nodes, client, MODEL_NAME, train_pairs=train_pairs, reasoning_budget=sbu_budget)
            if "virtual_root" in G_sbu_batch: G_sbu_batch.remove_node("virtual_root")
            eval_results["SBU LLM Batch"] = {
                "metrics": evaluate_all_modes(G_sbu_batch, G_gt, f"./results/{dataset_name_eval}_SBU_Batch"), 
                "runtime": time.time() - t0
            }

        if "sbu_embedding" in selected_methods:
            print("  -> Running SBU Embedding Strategy...")
            t0 = time.time()
            
            train_nodes = []
            if train_pairs:
                for pair in train_pairs:
                    train_nodes.extend([pair["parent"], pair["child"]])
                train_nodes = list(set(train_nodes))
                
            G_sbu_emb = method_sbu_embedding(input_nodes, vector_encoder, train_nodes=train_nodes)
            if "virtual_root" in G_sbu_emb: G_sbu_emb.remove_node("virtual_root")
            
            eval_results["SBU Embedding"] = {
                "metrics": evaluate_all_modes(G_sbu_emb, G_gt, f"./results/{dataset_name_eval}_SBU_Embedding"), 
                "runtime": time.time() - t0
            }

        if eval_results:
            display_summary_table(dataset_name_eval, eval_results)
            print(f"Saving benchmark results to JSON...")
            for method_name, info in eval_results.items():
                metrics = info["metrics"]
                flat_metrics = {
                    "Cond_Red_F1": metrics["Cond_Red"]["F1"],
                    "Cond_Clos_F1": metrics["Cond_Clos"]["F1"],
                    "Exp_Raw_F1": metrics["Exp_Raw"]["F1"],
                    "Exp_Clos_F1": metrics["Exp_Clos"]["F1"],
                    "Runtime_sec": info["runtime"]
                }
                update_benchmark_results(
                    dataset_name=dataset_name_eval, method_name=method_name, 
                    metrics_dict=flat_metrics, use_synsets=args.use_synsets, explode_nodes=args.explode_nodes
                )
            print("Save complete.")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy Extraction Benchmark")
    parser.add_argument("--method", nargs="+", default=["all"], choices=["all", "lexical", "vector", "llm_zero", "our_method", "taxollama", "sbu_batch", "sbu_embedding"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--scale", type=str, default="sub", choices=["sub", "full"], help="Benchmark on 'sub' (100-node) or 'full' ontology")
    parser.add_argument("--reasoning_level", type=str, default="medium", choices=["none", "low", "medium", "high"], help="Reasoning tokens per term budget")
    parser.add_argument("--use_synsets", action="store_true")
    parser.add_argument("--explode_nodes", action="store_true")
    args = parser.parse_args()
    main(args)
