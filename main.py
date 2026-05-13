import os
import argparse
import gc
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
    MODEL_NAME = "openai/gpt-oss-120b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_methods = args.method
    if "all" in selected_methods:
        selected_methods = ["lexical", "vector", "llm_zero", "our_method", "taxollama", "sbu_batch", "sbu_ensemble"]

    if any(m in selected_methods for m in ["llm_zero", "our_method", "sbu_batch"]):
        print("Initializing LLM Client...")
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="woohoo")
        
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
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        taxo_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=True)
        if taxo_tokenizer.pad_token is None:
            taxo_tokenizer.pad_token = taxo_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=quantization_config, device_map="auto", token=True
        )
        print(f"Applying adapter weights from {adapter_id}...")
        taxo_model = PeftModel.from_pretrained(base_model, adapter_id)
        taxo_model.eval()

    selected_ds = args.datasets
    if "all" in selected_ds:
        # Reduced Dataset Load
        selected_ds = ["WordNetFood", "CellOntology", "SemEvalFood", "SemEvalScience", "SemEvalEnvironment", 
                       "LLMs4OL_OBI", "LLMs4OL_MatOnto", "LLMs4OL_SchemaOrg", "LLMs4OL_PO"]

    for domain in selected_ds:
        G_gt = load_benchmark_graph(domain)
        if G_gt is None:
            print(f"\n [!] Error: No GT graph found for {domain}. Run taxonomy_metrics.py first.")
            continue
            
        print(f"\nEvaluating Domain: {domain} (Benchmark Nodes: {G_gt.number_of_nodes()})")
        
        # 1. Provide Domain Nodes without Virtual Scaffold
        input_nodes = [n for n in G_gt.nodes() if n != "virtual_root"]
        
        # 2. Reconstruct Train Pairs for Few-Shot Baselines (Excluding Virtual Root)
        train_pairs = [{"parent": u, "child": v} for u, v in G_gt.edges() if u != "virtual_root" and v != "virtual_root"]
        
        if args.explode_nodes:
            print("  -> Exploding cluster nodes for individual term analysis...")
            from evaluator import explode_graph
            G_gt = explode_graph(G_gt)
            input_nodes = [n for n in G_gt.nodes() if n != "virtual_root"]

        # Strip virtual root from GT permanently for strict edge evaluation
        if "virtual_root" in G_gt:
            G_gt.remove_node("virtual_root")
            
        nx.write_graphml(G_gt, f"./results/GT_{domain}_eval.graphml")
        eval_results = {}
        
        if "lexical" in selected_methods:
            print("  -> Running Lexical...")
            G_lex = method_lexical(input_nodes)
            if "virtual_root" in G_lex: G_lex.remove_node("virtual_root")
            eval_results["Lexical"] = evaluate_all_modes(G_lex, G_gt, f"./results/{domain}_Lexical")
            
        if "vector" in selected_methods:
            print("  -> Running Vector...")
            G_vec = method_vector(input_nodes, vector_encoder)
            if "virtual_root" in G_vec: G_vec.remove_node("virtual_root")
            eval_results["Vector"] = evaluate_all_modes(G_vec, G_gt, f"./results/{domain}_Vector")
            
        if "llm_zero" in selected_methods:
            print("  -> Running Zero-Shot LLM...")
            G_llm_zero = method_llm_single_shot(input_nodes, client, MODEL_NAME)
            if "virtual_root" in G_llm_zero: G_llm_zero.remove_node("virtual_root")
            eval_results["LLM Zero-Shot"] = evaluate_all_modes(G_llm_zero, G_gt, f"./results/{domain}_LLMZero")

        if "our_method" in selected_methods:
            print("  -> Running Our Method...")
            G_our = method_our_approach(input_nodes, client, MODEL_NAME)
            if "virtual_root" in G_our: G_our.remove_node("virtual_root")
            eval_results["Our Method O(N)"] = evaluate_all_modes(G_our, G_gt, f"./results/{domain}_OurMethod")

        if "taxollama" in selected_methods:
            print("  -> Precomputing TaxoLLaMA Scores...")
            ppl_cache = precompute_taxollama_ppl(input_nodes, taxo_model, taxo_tokenizer, device)
            thresh = 15.0
            G_taxo = build_taxollama_graph(input_nodes, ppl_cache, threshold=thresh, max_parents=1)
            if "virtual_root" in G_taxo: G_taxo.remove_node("virtual_root")
            method_name = f"TaxoLLaMA (PPL<{thresh})"
            eval_results[method_name] = evaluate_all_modes(G_taxo, G_gt, f"./results/{domain}_TaxoLLaMA_PPL{thresh}")

        if "sbu_batch" in selected_methods:
            print("  -> Running SBU LLM Batch Prompting...")
            G_sbu_batch = method_sbu_batch(input_nodes, client, MODEL_NAME, train_pairs=train_pairs)
            if "virtual_root" in G_sbu_batch: G_sbu_batch.remove_node("virtual_root")
            eval_results["SBU LLM Batch"] = evaluate_all_modes(G_sbu_batch, G_gt, f"./results/{domain}_SBU_Batch")

        if "sbu_ensemble" in selected_methods:
            print("  -> Running SBU Ensemble (Overlap + Embedding)...")
            G_sbu_ens = method_sbu_ensemble(input_nodes, vector_encoder)
            if "virtual_root" in G_sbu_ens: G_sbu_ens.remove_node("virtual_root")
            eval_results["SBU Ensemble"] = evaluate_all_modes(G_sbu_ens, G_gt, f"./results/{domain}_SBU_Ensemble")

        if eval_results:
            display_summary_table(domain, eval_results)
            print(f"Saving benchmark results to JSON...")
            for method_name, metrics in eval_results.items():
                flat_metrics = {
                    "Cond_Red_F1": metrics["Cond_Red"]["F1"],
                    "Cond_Clos_F1": metrics["Cond_Clos"]["F1"],
                    "Exp_Raw_F1": metrics["Exp_Raw"]["F1"],
                    "Exp_Clos_F1": metrics["Exp_Clos"]["F1"]
                }
                update_benchmark_results(
                    dataset_name=domain, method_name=method_name, 
                    metrics_dict=flat_metrics, use_synsets=args.use_synsets, explode_nodes=args.explode_nodes
                )
            print("Save complete.")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy Extraction Benchmark")
    parser.add_argument("--method", nargs="+", default=["all"], choices=["all", "lexical", "vector", "llm_zero", "our_method", "taxollama", "sbu_batch", "sbu_ensemble"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--use_synsets", action="store_true")
    parser.add_argument("--explode_nodes", action="store_true")
    args = parser.parse_args()
    main(args)
