import os
import json
import networkx as nx
from data_manager import parse_lemma_format

def update_benchmark_results(dataset_name, method_name, metrics_dict, use_synsets, explode_nodes, filepath="benchmark_results.json"):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = [] 
                
        if isinstance(data, dict):
            data = []
    else:
        data = []

    target_block = None
    for block in data:
        if (block.get("dataset") == dataset_name and 
            block.get("use_synsets", False) == use_synsets and 
            block.get("explode_nodes", False) == explode_nodes):
            target_block = block
            break

    if target_block is None:
        target_block = {
            "dataset": dataset_name,
            "use_synsets": use_synsets,
            "explode_nodes": explode_nodes,
            "results": []
        }
        data.append(target_block)

    found_method = False
    for i, res in enumerate(target_block["results"]):
        if res.get("method") == method_name:
            res.update(metrics_dict)
            found_method = True
            break

    if not found_method:
        new_res = {"method": method_name}
        new_res.update(metrics_dict)
        target_block["results"].append(new_res)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def compute_and_save_metrics(G_pred, G_gt, output_file, mode_name, match_type="exact"):
    edges_pred = set(G_pred.edges())
    edges_gt = set(G_gt.edges())

    tp_pred = set()
    tp_gt = set()
    tp_log = []

    if match_type == "exact":
        tp_edges = edges_pred.intersection(edges_gt)
        tp_pred = tp_edges
        tp_gt = tp_edges
        for u, v in sorted(tp_edges):
            tp_log.append(f"  [TP] {u} -> {v}")
            
    elif match_type == "set_overlap":
        def get_terms(node_str):
            return set(parse_lemma_format(node_str))

        for p_u, p_v in edges_pred:
            pu_set = get_terms(p_u)
            pv_set = get_terms(p_v)
            for g_u, g_v in edges_gt:
                gu_set = get_terms(g_u)
                gv_set = get_terms(g_v)
                
                if (pu_set & gu_set) and (pv_set & gv_set):
                    tp_pred.add((p_u, p_v))
                    tp_gt.add((g_u, g_v))
                    tp_log.append(f"  [TP] PRED: ({p_u} -> {p_v})\n       matches GT: ({g_u} -> {g_v})")

    fp_edges = edges_pred - tp_pred
    fn_edges = edges_gt - tp_gt

    precision = len(tp_pred) / len(edges_pred) if edges_pred else 0.0
    recall = len(tp_gt) / len(edges_gt) if edges_gt else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(f"=== Detailed Node Pair Analysis ({mode_name}) ===\n")
            f.write(f"Match Type Used: {match_type.upper()}\n")
            f.write("="*50 + "\n")
            f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")
            
            f.write(f"\nCORRECT PREDICTIONS (True Positives: {len(tp_pred)} Pred edges covering {len(tp_gt)} GT edges):\n")
            if tp_log:
                f.write("\n".join(sorted(tp_log)) + "\n")
            else:
                f.write("  None.\n")

            f.write(f"\nINCORRECT PREDICTIONS (False Positives: {len(fp_edges)}):\n")
            if fp_edges:
                for u, v in sorted(fp_edges):
                    f.write(f"  [FP] {u} -> {v}\n")
            else:
                f.write("  None. (Perfect Precision!)\n")

            f.write(f"\nMISSED RELATIONS (False Negatives: {len(fn_edges)}):\n")
            if fn_edges:
                for u, v in sorted(fn_edges):
                    f.write(f"  [FN] {u} -> {v}\n")
            else:
                f.write("  None. (Perfect Recall!)\n")
                
    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")

    return {"Precision": precision, "Recall": recall, "F1": f1}

def explode_graph(G):
    G_exp = nx.DiGraph()
    for node in G.nodes():
        sub_terms = parse_lemma_format(node)
        G_exp.add_nodes_from(sub_terms)
        for i in range(len(sub_terms)):
            for j in range(len(sub_terms)):
                if i != j:
                    G_exp.add_edge(sub_terms[i], sub_terms[j])
                    
    for u, v in G.edges():
        u_terms = parse_lemma_format(u)
        v_terms = parse_lemma_format(v)
        for ut in u_terms:
            for vt in v_terms:
                if ut != vt:
                    G_exp.add_edge(ut, vt)
    return G_exp

def evaluate_all_modes(G_pred_condensed, G_gt_condensed, out_prefix):
    G_pred_exploded = explode_graph(G_pred_condensed)
    G_gt_exploded = explode_graph(G_gt_condensed)

    results = {}

    G_pred_cond_red = nx.transitive_reduction(G_pred_condensed)
    G_gt_cond_red = nx.transitive_reduction(G_gt_condensed)
    results["Cond_Red"] = compute_and_save_metrics(G_pred_cond_red, G_gt_cond_red, f"{out_prefix}_condensed_reduction.txt", "Condensed Transitive Reduction", match_type="set_overlap")

    G_pred_cond_clos = nx.transitive_closure(G_pred_condensed)
    G_gt_cond_clos = nx.transitive_closure(G_gt_condensed)
    results["Cond_Clos"] = compute_and_save_metrics(G_pred_cond_clos, G_gt_cond_clos, f"{out_prefix}_condensed_closure.txt", "Condensed Transitive Closure", match_type="set_overlap")

    results["Exp_Raw"] = compute_and_save_metrics(G_pred_exploded, G_gt_exploded, f"{out_prefix}_exploded_raw.txt", "Exploded Raw Edges", match_type="exact")

    G_pred_exp_clos = nx.transitive_closure(G_pred_exploded)
    G_gt_exp_clos = nx.transitive_closure(G_gt_exploded)
    results["Exp_Clos"] = compute_and_save_metrics(G_pred_exp_clos, G_gt_exp_clos, f"{out_prefix}_exploded_closure.txt", "Exploded Transitive Closure", match_type="exact")

    return results
