import os
import re
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

# Create visualization directory
VIS_DIR = "vis"
os.makedirs(VIS_DIR, exist_ok=True)

# Set visual style for dense, academic-style figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

def load_and_merge_data(json_path="benchmark_results.json", csv_path="dataset_metrics.csv"):
    """Loads the results and metrics, normalizes dataset names, and merges them."""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    records = []
    for ds_block in json_data:
        dataset_name = ds_block.get("dataset")
        use_synsets = ds_block.get("use_synsets", False)
        explode_nodes = ds_block.get("explode_nodes", False)
        
        for res in ds_block.get("results", []):
            primary_f1 = res.get("Cond_Clos_F1", 0)
            
            records.append({
                "Dataset_JSON": dataset_name,
                "Use_Synsets": use_synsets,
                "Explode_Nodes": explode_nodes,
                "Method": res["method"],
                "Cond_Red_F1": res.get("Cond_Red_F1", 0),
                "Cond_Clos_F1": res.get("Cond_Clos_F1", 0),
                "Exp_Raw_F1": res.get("Exp_Raw_F1", 0),
                "Exp_Clos_F1": res.get("Exp_Clos_F1", 0),
                "Primary_F1": primary_f1,
                "Runtime_sec": res.get("Runtime_sec", 0.0) 
            })
    df_perf = pd.DataFrame(records)
    
    df_metrics = pd.read_csv(csv_path)
    
    df_perf["Dataset_JSON"] = df_perf["Dataset_JSON"].str.replace('.csv', '', regex=False)
    df_metrics["Dataset"] = df_metrics["Dataset"].str.replace('.csv', '', regex=False)

    df_merged = pd.merge(df_perf, df_metrics, left_on="Dataset_JSON", right_on="Dataset", how="inner")
    
    json_datasets = set(df_perf["Dataset_JSON"])
    csv_datasets = set(df_metrics["Dataset"])
    missing_in_csv = json_datasets - csv_datasets
    if missing_in_csv:
        print(f" [!] WARNING: These datasets are in your JSON but missing from dataset_metrics.csv: {missing_in_csv}")
    
    def make_scenario_name(row):
        flags = []
        if row['Use_Synsets']: flags.append('Syn')
        if row['Explode_Nodes']: flags.append('Exp')
        suffix = f" ({'+'.join(flags)})" if flags else ""
        return str(row['Dataset']) + suffix

    df_merged['Dataset_Scenario'] = df_merged.apply(make_scenario_name, axis=1)
    
    print(f"[*] Successfully merged data for {df_merged['Dataset_Scenario'].nunique()} unique dataset scenarios.")
    return df_merged

def plot_method_vs_dataset_heatmap(df):
    """Creates a heatmap of F1 scores for each Method across all Datasets & Scenarios."""
    
    df_main = df[~((df['Use_Synsets'] == True) & (df['Explode_Nodes'] == True))].copy()
    
    metrics_to_plot = {
        "Exp_Raw_F1": "Raw Exact Match F1",
        "Cond_Clos_F1": "Condensed Closure F1"
    }
    
    for metric_key, metric_title in metrics_to_plot.items():
        print(f" -> Generating Main Method vs Dataset Heatmap for {metric_title}...")
        pivot_df = df_main.pivot(index="Method", columns="Dataset_Scenario", values=metric_key)
        
        pivot_df["Row_Mean"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values(by="Row_Mean", ascending=False)
        pivot_df = pivot_df.drop(columns=["Row_Mean"])
        
        num_scenarios = len(pivot_df.columns)
        fig_width = max(14, num_scenarios * 1.2)
        
        plt.figure(figsize=(fig_width, 8))
        ax = sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, vmin=0, vmax=1)
        plt.title(f"Method Performance ({metric_title}) Across Base Datasets", pad=20, fontsize=14, fontweight='bold')
        plt.ylabel("Extraction Method", fontweight='bold')
        plt.xlabel("Dataset Scenario", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for label in ax.get_yticklabels():
            if "Our Method" in label.get_text():
                label.set_color("darkred")
                label.set_fontweight("bold")
                
        plt.tight_layout()
        filename = f"1a_method_vs_dataset_heatmap_{metric_key}.png"
        plt.savefig(os.path.join(VIS_DIR, filename), dpi=300)
        plt.close()

def plot_syn_exp_comparison_heatmap(df):
    """Creates a side-by-side comparison heatmap specifically for Our Method."""
    
    syn_exp_datasets = df[(df['Use_Synsets'] == True) & (df['Explode_Nodes'] == True)]['Dataset_JSON'].unique()
    
    if len(syn_exp_datasets) == 0:
        print(" -> [!] No (Syn+Exp) runs found. Skipping synonym comparison heatmap.")
        return

    mask = (df['Dataset_JSON'].isin(syn_exp_datasets)) & (
        ((df['Use_Synsets'] == False) & (df['Explode_Nodes'] == False)) |
        ((df['Use_Synsets'] == True) & (df['Explode_Nodes'] == True))
    )
    comp_df = df[mask].copy()
    
    comp_df = comp_df[comp_df["Method"].str.contains("Our Method", case=False, na=False)]
    if comp_df.empty:
        print(" -> [!] No 'Our Method' data found for Synonym comparison. Skipping.")
        return
    
    metrics_to_plot = {
        "Cond_Red_F1": "Condensed Reduction F1 (Synonym Recovery Focus)",
        "Cond_Clos_F1": "Condensed Closure F1",
        "Exp_Raw_F1": "Raw Exact Match F1"
    }

    for metric_key, metric_title in metrics_to_plot.items():
        print(f" -> Generating Synonym Recovery Comparison Heatmap for {metric_title}...")
        pivot_df = comp_df.pivot(index="Method", columns="Dataset_Scenario", values=metric_key)
        
        sorted_cols = sorted(pivot_df.columns, key=lambda x: (x.replace(' (Syn+Exp)', ''), x))
        pivot_df = pivot_df[sorted_cols]
        
        num_scenarios = len(pivot_df.columns)
        fig_width = max(10, num_scenarios * 1.5)
        
        plt.figure(figsize=(fig_width, 4))
        ax = sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, vmin=0, vmax=1)
        plt.title(f"Synonym Discovery and Recovery: {metric_title}", pad=20, fontsize=14, fontweight='bold')
        plt.ylabel("Extraction Method", fontweight='bold')
        plt.xlabel("Paired Dataset Scenarios (Base vs Syn+Exp)", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for label in ax.get_yticklabels():
            label.set_color("darkred")
            label.set_fontweight("bold")
                
        plt.tight_layout()
        filename = f"1b_syn_exp_comparison_heatmap_{metric_key}.png"
        plt.savefig(os.path.join(VIS_DIR, filename), dpi=300)
        plt.close()

def plot_method_variance(df):
    print(" -> Generating Method Variance Plot...")
    plt.figure(figsize=(12, 6))
    
    order = df.groupby("Method")["Primary_F1"].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x="Method", y="Primary_F1", order=order, palette="Set2", showmeans=True, 
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=df, x="Method", y="Primary_F1", order=order, color=".25", size=5, alpha=0.6, jitter=True)
    
    plt.title("Condensed Closure F1 Variance per Method Across All Scenarios", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("Primary F1 Score (Cond Closure)", fontweight='bold')
    plt.xlabel("Extraction Method", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "2_method_variance_boxplot.png"), dpi=300)
    plt.close()

def plot_metric_vs_f1_scatter_grid(df):
    print(" -> Generating Metrics vs F1 Scatter Grid (with Trendlines)...")
    
    taxo_mask = df['Method'].str.startswith('TaxoLLaMA')
    df_non_taxo = df[~taxo_mask].copy()
    df_taxo = df[taxo_mask].copy()
    
    if not df_taxo.empty:
        idx_best_taxo = df_taxo.groupby('Dataset_Scenario')['Primary_F1'].idxmax()
        df_best_taxo = df_taxo.loc[idx_best_taxo].copy()
        df_best_taxo['Method'] = 'TaxoLLaMA (Best PPL)'
        df_plot = pd.concat([df_non_taxo, df_best_taxo])
    else:
        df_plot = df_non_taxo

    metrics_to_plot = ["Avg Hearst PPL", "Lexical Overlap", "Tree-likeness", "Max Depth", "Avg Branching", "Edge/Node Ratio", "Nodes"]
    methods = df_plot['Method'].unique()
    palette = sns.color_palette("husl", n_colors=len(methods))
    color_dict = dict(zip(methods, palette))

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric not in df_plot.columns: continue
            
        sns.scatterplot(
            data=df_plot, x=metric, y="Primary_F1", hue="Method", 
            palette=color_dict, s=100, alpha=1.0, ax=axes[i], legend=(i == 0)
        )
        
        for method in methods:
            subset = df_plot[df_plot['Method'] == method]
            if len(subset) > 1:
                sns.regplot(
                    data=subset, x=metric, y="Primary_F1", ax=axes[i],
                    color=color_dict[method], scatter=False, 
                    line_kws={'linewidth': 2}, ci=None
                )
        
        axes[i].set_title(f"Cond Closure F1 vs {metric}", fontweight='bold')
        axes[i].set_ylim(-0.05, 1.05)
        if metric == "Nodes": axes[i].set_xscale("log") 
            
        if i == 0:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        else:
            if axes[i].get_legend() is not None: axes[i].get_legend().remove()

    if len(metrics_to_plot) < len(axes):
        for j in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "3_metric_vs_f1_scatters.png"), dpi=300)
    plt.close()

def plot_f1_metric_correlation_heatmap(df):
    print(" -> Generating F1-Metric Correlation Heatmap (Spearman)...")
    
    metrics = ["Avg Hearst PPL", "Nodes", "Edges", "Roots", "Leaves", "Components", "Edge/Node Ratio", 
               "Max Depth", "Avg Branching", "Tree-likeness", "Lexical Overlap", "Redundancy Ratio"]
    metrics = [m for m in metrics if m in df.columns]
    
    corr_data = {}
    methods = df['Method'].unique()
    
    for method in methods:
        method_df = df[df["Method"] == method]
        if len(method_df) > 2: 
            corr_data[method] = {}
            for metric in metrics:
                corr = method_df["Primary_F1"].corr(method_df[metric], method='spearman')
                corr_data[method][metric] = corr

    corr_df = pd.DataFrame(corr_data)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_df, annot=True, cmap="vlag", fmt=".2f", center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Spearman Rank Correlation (Metric vs Cond Closure F1)'})
    plt.title("Correlation: How Dataset Topology Affects Method Cond Closure F1 Scores", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Extraction Method", fontweight='bold')
    plt.ylabel("Dataset Topography Metric", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "4_topology_f1_correlation_heatmap.png"), dpi=300)
    plt.close()

def plot_runtime_complexity(df):
    print(" -> Generating Runtime Scaling Chart...")
    
    df_runtime = df[df["Runtime_sec"] > 0].copy()
    if df_runtime.empty:
        print("    [!] No runtime data found to plot. Skipping chart.")
        return
        
    plt.figure(figsize=(10, 6))
    
    palette = sns.color_palette("husl", n_colors=len(df_runtime['Method'].unique()))
    color_dict = dict(zip(df_runtime['Method'].unique(), palette))

    sns.scatterplot(
        data=df_runtime, x="Nodes", y="Runtime_sec", hue="Method", 
        palette=color_dict, s=120, alpha=0.8, edgecolor='black'
    )
    
    for method in df_runtime['Method'].unique():
        subset = df_runtime[df_runtime['Method'] == method].sort_values(by="Nodes")
        if len(subset) > 1:
            x_vals = subset["Nodes"].values
            y_vals = subset["Runtime_sec"].values
            
            coeffs = np.polyfit(np.log10(x_vals), np.log10(y_vals), 1)
            m, c = coeffs
            
            x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 100)
            y_fit = (10 ** c) * (x_fit ** m)
            
            plt.plot(x_fit, y_fit, color=color_dict[method], alpha=0.5, linewidth=2.5, linestyle='--')
            
    plt.title("Computational Scaling: Runtime vs Graph Size", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Total Nodes in Dataset", fontweight='bold')
    plt.ylabel("Execution Time in Seconds (Log Scale)", fontweight='bold')
    plt.xscale("log")
    plt.yscale("log")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 9})
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "9_runtime_scaling.png"), dpi=300)
    plt.close()

def plot_f1_metric_comparison(df):
    print(" -> Generating F1 Metric Version Comparison...")
    
    f1_cols = ["Cond_Red_F1", "Cond_Clos_F1", "Exp_Raw_F1", "Exp_Clos_F1"]
    melted_df = df.melt(id_vars=["Dataset_Scenario", "Method"], value_vars=f1_cols, 
                        var_name="Evaluation_Type", value_name="F1_Score")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_df, x="Evaluation_Type", y="F1_Score", palette="pastel", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=melted_df, x="Evaluation_Type", y="F1_Score", color=".25", size=3, alpha=0.4, jitter=True)
    
    plt.title("Comparison of F1 Score Variations Across All Scenarios", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("F1 Score", fontweight='bold')
    plt.xlabel("Evaluation Metric Type", fontweight='bold')
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Condensed Reduction", "Condensed Closure", "Exploded Raw", "Exploded Closure"])
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "6_f1_version_comparison_overall.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.barplot(data=melted_df, x="Method", y="F1_Score", hue="Evaluation_Type", palette="Set2", errorbar=None)
    
    plt.title("Average F1 Score by Method and Evaluation Type", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("Average F1 Score", fontweight='bold')
    plt.xlabel("Extraction Method", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    nice_labels = {"Cond_Red_F1": "Cond. Reduction", "Cond_Clos_F1": "Cond. Closure", 
                   "Exp_Raw_F1": "Exploded Raw", "Exp_Clos_F1": "Exploded Closure"}
    plt.legend(handles, [nice_labels.get(l, l) for l in labels], title="Evaluation Type")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "7_f1_version_comparison_by_method.png"), dpi=300)
    plt.close()

def report_method_variance(df):
    print(" -> Generating Method Variance Report...")
    
    var_df = df.groupby("Method")["Primary_F1"].agg(
        Mean="mean", Variance="var", Std_Dev="std", Min="min", Max="max"
    ).reset_index()
    
    var_df = var_df.sort_values(by="Mean", ascending=False).reset_index(drop=True)
    output_csv = os.path.join(VIS_DIR, "8_method_variance_report.csv")
    var_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*85)
    print("   METHOD VARIANCE REPORT (Condensed Closure F1)   ")
    print("="*85)
    print(var_df.to_string(formatters={
        'Mean': '{:,.4f}'.format, 'Variance': '{:,.4f}'.format,
        'Std_Dev': '{:,.4f}'.format, 'Min': '{:,.4f}'.format, 'Max': '{:,.4f}'.format
    }, index=False))
    print("="*85)

def generate_summary_table(df):
    print(" -> Generating Summary Table...")
    
    idx = df.groupby('Dataset_Scenario')['Primary_F1'].idxmax()
    best_df = df.loc[idx, ['Dataset_Scenario', 'Method', 'Primary_F1', 'Runtime_sec', 'Nodes', 'Lexical Overlap', 'Avg Hearst PPL']]
    
    best_df = best_df.sort_values(by="Dataset_Scenario").reset_index(drop=True)
    best_df.columns = ["Dataset Scenario", "Best Method", "Best Cond. Closure F1", "Runtime (s)", "Total Nodes", "Lexical Overlap", "Avg Hearst PPL"]
    
    best_df.to_csv(os.path.join(VIS_DIR, "5_best_methods_summary.csv"), index=False)
    print("\n" + "="*110)
    print("   BEST PERFORMING METHODS PER DATASET SCENARIO (By Cond Closure F1)   ")
    print("="*110)
    print(best_df.to_string(index=False))
    print("="*110)

def plot_graph_overlays(dataset_name):
    """Draws a strict hierarchical visualization mapping nodes to exact Y-levels based on depth."""
    gt_path = f"./results/GT_{dataset_name}_eval.graphml"
    if not os.path.exists(gt_path):
        print(f" [!] GT graph not found at {gt_path}. Cannot generate graph overlay.")
        return
        
    G_gt = nx.read_graphml(gt_path)
    print(f" -> Generating Strict Hierarchical Graph Overlays for {dataset_name}...")
    
    # 1. Enforce strict mathematical node depths (longest path from any root)
    roots = [n for n, d in G_gt.in_degree() if d == 0]
    if not roots: 
        roots = [list(G_gt.nodes())[0]] # Fallback if no true roots exist
        
    layers = {}
    try:
        # topological_sort guarantees we evaluate parents before children
        for node in nx.topological_sort(G_gt):
            if node in roots:
                layers[node] = 0
            else:
                layers[node] = max([layers[p] for p in G_gt.predecessors(node)]) + 1
    except nx.NetworkXUnfeasible:
        # Fallback if a cycle somehow exists in the GT
        for n in G_gt.nodes():
            layers[n] = len(nx.ancestors(G_gt, n))
            
    nx.set_node_attributes(G_gt, layers, 'layer')
    
    # 2. Map layers precisely to Y-coordinates
    pos = nx.multipartite_layout(G_gt, subset_key='layer', align='horizontal')
    for k in pos: 
        pos[k][1] = -pos[k][1]  # Invert Y to put the root (Layer 0) at the top
        
    txt_files = glob.glob(f"./results/{dataset_name}_*_condensed_closure.txt")
    if not txt_files:
        print(f" [!] No condensed closure text files found in ./results/ for {dataset_name}.")
        return
        
    tp_pattern = re.compile(r"matches GT:\s*\((.*?)\s*->\s*(.*?)\)")
    fp_pattern = re.compile(r"\[FP\]\s+(.*?)\s*->\s*(.*)$")
    
    for txt_file in txt_files:
        method_suffix = txt_file.split(f"{dataset_name}_")[-1].replace("_condensed_closure.txt", "")
        
        tp_edges = []
        fp_edges = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tp_match = tp_pattern.search(line)
                if tp_match:
                    tp_edges.append((tp_match.group(1), tp_match.group(2)))
                    continue
                fp_match = fp_pattern.search(line)
                if fp_match:
                    fp_edges.append((fp_match.group(1), fp_match.group(2)))
        
        plt.figure(figsize=(20, 14))
        
        # 3. Use rad=0.15 to curve edges around nodes, preventing straight-line overlaps
        # Base GT Edges (Transparent Gray)
        nx.draw_networkx_edges(G_gt, pos, edge_color='gray', alpha=0.15, arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.15')
        
        # Nodes
        nx.draw_networkx_nodes(G_gt, pos, node_size=60, node_color='lightblue', edgecolors='black')
        
        # Bounding box labels so the text is legible over the dense mesh
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        nx.draw_networkx_labels(G_gt, pos, font_size=7, bbox=bbox_props)
        
        # TP Edges (Vibrant Green)
        valid_tp = [e for e in tp_edges if e[0] in pos and e[1] in pos]
        if valid_tp:
            nx.draw_networkx_edges(G_gt, pos, edgelist=valid_tp, edge_color='green', alpha=0.8, width=2.5, arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.15')
            
        # FP Edges (Transparent Red)
        valid_fp = [e for e in fp_edges if e[0] in pos and e[1] in pos]
        if valid_fp:
            G_fp = nx.DiGraph()
            G_fp.add_nodes_from(G_gt.nodes())
            G_fp.add_edges_from(valid_fp)
            nx.draw_networkx_edges(G_fp, pos, edgelist=valid_fp, edge_color='red', alpha=0.3, width=1.5, arrows=True, arrowsize=12, connectionstyle='arc3,rad=0.15')
            
        plt.title(f"Hierarchy Overlay: {dataset_name} | Method: {method_suffix}\nGreen=Recovered GT (TP) | Red=Hallucinated (FP)", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        out_name = f"10_{dataset_name}_{method_suffix}_graph_vis.png"
        plt.savefig(os.path.join(VIS_DIR, out_name), dpi=300)
        plt.close()
        print(f"    -> Saved strict layered graph overlay for {method_suffix}")

def plot_reasoning_effort_comparison(df):
    """Compares Accuracy (F1) vs Compute Cost (Runtime) across reasoning efforts."""
    print(" -> Generating Reasoning Effort Comparison...")
    
    reasoning_methods = ['Our Method O(N) [low]', 'Our Method O(N) [medium]', 'Our Method O(N) [high]']
    df_res = df[df['Method'].isin(reasoning_methods)].copy()
    
    if df_res.empty:
        print("    [!] No reasoning effort runs found (low, medium, high). Skipping.")
        return

    # Ensure correct ordinal plotting
    df_res['Method'] = pd.Categorical(df_res['Method'], categories=reasoning_methods, ordered=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy (F1) Comparison
    sns.boxplot(data=df_res, x="Method", y="Primary_F1", palette="Blues", ax=ax1,
                showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=df_res, x="Method", y="Primary_F1", color=".25", size=5, alpha=0.6, jitter=True, ax=ax1)
    
    ax1.set_title("Accuracy: Condensed Closure F1 by Reasoning Effort", pad=15, fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cond Closure F1", fontweight='bold')
    ax1.set_xlabel("Reasoning Effort Level", fontweight='bold')
    ax1.set_xticklabels(['Low', 'Medium', 'High'])
    
    # Plot 2: Compute Cost (Runtime) Comparison
    sns.boxplot(data=df_res, x="Method", y="Runtime_sec", palette="Oranges", ax=ax2,
                showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=df_res, x="Method", y="Runtime_sec", color=".25", size=5, alpha=0.6, jitter=True, ax=ax2)
    
    ax2.set_title("Compute Cost: Runtime by Reasoning Effort", pad=15, fontsize=14, fontweight='bold')
    ax2.set_ylabel("Execution Time (Seconds)", fontweight='bold')
    ax2.set_xlabel("Reasoning Effort Level", fontweight='bold')
    ax2.set_xticklabels(['Low', 'Medium', 'High'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "11_reasoning_effort_comparison.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Taxonomy Extraction Visualizer")
    parser.add_argument("--vis_graph", type=str, default=None, help="Dataset name to visualize GT and Method overlays (e.g., WordNetFood_SUB)")
    args = parser.parse_args()

    if not os.path.exists("benchmark_results.json") or not os.path.exists("dataset_metrics.csv"):
        print("Error: Required files 'benchmark_results.json' or 'dataset_metrics.csv' not found.")
        print("Please ensure you have run both benchmark pipelines in this directory.")
        return

    df = load_and_merge_data()
    
    plot_method_vs_dataset_heatmap(df)
    plot_syn_exp_comparison_heatmap(df)
    plot_method_variance(df)
    plot_metric_vs_f1_scatter_grid(df)
    plot_runtime_complexity(df)
    plot_f1_metric_correlation_heatmap(df)
    plot_f1_metric_comparison(df) 
    plot_reasoning_effort_comparison(df)
    report_method_variance(df)
    generate_summary_table(df)
    
    if args.vis_graph:
        plot_graph_overlays(args.vis_graph)
    
    print(f"\n[*] All visualizations successfully saved to the '{VIS_DIR}/' directory.")

if __name__ == "__main__":
    main()
