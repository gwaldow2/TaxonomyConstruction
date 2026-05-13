import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create visualization directory
VIS_DIR = "vis"
os.makedirs(VIS_DIR, exist_ok=True)

# Set visual style for dense, academic-style figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

def load_and_merge_data(json_path="benchmark_results.json", csv_path="dataset_metrics.csv"):
    """Loads the results and metrics, normalizes dataset names, and merges them."""
    
    # 1. Load JSON (Method Performances)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    records = []
    for ds_block in json_data:
        dataset_name = ds_block.get("dataset")
        use_synsets = ds_block.get("use_synsets", False)
        explode_nodes = ds_block.get("explode_nodes", False)
        
        for res in ds_block.get("results", []):
            # Dynamically select the best metric depending on if nodes were exploded
            primary_f1 = res["Exp_Clos_F1"] if explode_nodes else res["Cond_Clos_F1"]
            
            records.append({
                "Dataset_JSON": dataset_name,
                "Use_Synsets": use_synsets,
                "Explode_Nodes": explode_nodes,
                "Method": res["method"],
                "Cond_Red_F1": res.get("Cond_Red_F1", 0),
                "Cond_Clos_F1": res.get("Cond_Clos_F1", 0),
                "Exp_Raw_F1": res.get("Exp_Raw_F1", 0),
                "Exp_Clos_F1": res.get("Exp_Clos_F1", 0),
                "Primary_F1": primary_f1 
            })
    df_perf = pd.DataFrame(records)
    
    # 2. Load CSV (Dataset Metrics)
    df_metrics = pd.read_csv(csv_path)
    
    # 3. Clean string artifacts ensuring a direct merge
    df_perf["Dataset_JSON"] = df_perf["Dataset_JSON"].str.replace('.csv', '', regex=False)
    df_metrics["Dataset"] = df_metrics["Dataset"].str.replace('.csv', '', regex=False)

    # 4. Merge
    df_merged = pd.merge(df_perf, df_metrics, left_on="Dataset_JSON", right_on="Dataset", how="inner")
    
    # --- DEBUGGING: Catch dropped datasets ---
    json_datasets = set(df_perf["Dataset_JSON"])
    csv_datasets = set(df_metrics["Dataset"])
    missing_in_csv = json_datasets - csv_datasets
    if missing_in_csv:
        print(f" [!] WARNING: These datasets are in your JSON but missing from dataset_metrics.csv: {missing_in_csv}")
    # ---------------------------------------
    
    # 5. Create Scenario Names (e.g., "WordNetFood_SUB (Syn+Exp)")
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
    print(" -> Generating Method vs Dataset Heatmap...")
    
    # Pivot the data for the heatmap using the Scenario string
    pivot_df = df.pivot(index="Method", columns="Dataset_Scenario", values="Primary_F1")
    
    # Dynamically scale width based on the number of datasets to prevent squished labels
    num_scenarios = len(pivot_df.columns)
    fig_width = max(14, num_scenarios * 1.2)
    
    plt.figure(figsize=(fig_width, 8))
    ax = sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5, vmin=0, vmax=1)
    plt.title("Method Performance (Primary F1) Across Datasets (Sub & Full Scales)", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("Extraction Method", fontweight='bold')
    plt.xlabel("Dataset Scenario", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "1_method_vs_dataset_heatmap.png"), dpi=300)
    plt.close()

def plot_method_variance(df):
    """Creates a violin/box plot showing how much each method's F1 varies across different datasets."""
    print(" -> Generating Method Variance Plot...")
    
    plt.figure(figsize=(12, 6))
    
    order = df.groupby("Method")["Primary_F1"].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x="Method", y="Primary_F1", order=order, palette="Set2", showmeans=True, 
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=df, x="Method", y="Primary_F1", order=order, color=".25", size=5, alpha=0.6, jitter=True)
    
    plt.title("F1 Score Variance per Method Across All Scenarios", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("Primary F1 Score (Cond/Exp Closure)", fontweight='bold')
    plt.xlabel("Extraction Method", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "2_method_variance_boxplot.png"), dpi=300)
    plt.close()

def plot_metric_vs_f1_scatter_grid(df):
    """Creates a 2D grid of scatter plots comparing specific dataset metrics against F1 scores, with trendlines."""
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
        
        axes[i].set_title(f"F1 Score vs {metric}", fontweight='bold')
        axes[i].set_ylim(-0.05, 1.05)
        
        if metric == "Nodes":
            axes[i].set_xscale("log") 
            
        if i == 0:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
        else:
            if axes[i].get_legend() is not None:
                axes[i].get_legend().remove()

    if len(metrics_to_plot) < len(axes):
        for j in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "3_metric_vs_f1_scatters.png"), dpi=300)
    plt.close()

def plot_f1_metric_correlation_heatmap(df):
    """Creates a heatmap showing the Spearman correlation between dataset characteristics and model success."""
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
                cbar_kws={'label': 'Spearman Rank Correlation (Metric vs F1)'})
    plt.title("Correlation: How Dataset Topology Affects Method F1 Scores", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Extraction Method", fontweight='bold')
    plt.ylabel("Dataset Topography Metric", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "4_topology_f1_correlation_heatmap.png"), dpi=300)
    plt.close()

def plot_f1_metric_comparison(df):
    """Analyzes how the evaluation metric itself (Raw vs Clos, Cond vs Exp) changes the scores."""
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
    """Generates a statistical report of the variance, std dev, and mean of F1 scores per method."""
    print(" -> Generating Method Variance Report...")
    
    var_df = df.groupby("Method")["Primary_F1"].agg(
        Mean="mean", Variance="var", Std_Dev="std", Min="min", Max="max"
    ).reset_index()
    
    var_df = var_df.sort_values(by="Mean", ascending=False).reset_index(drop=True)
    output_csv = os.path.join(VIS_DIR, "8_method_variance_report.csv")
    var_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*85)
    print("📊 METHOD VARIANCE REPORT (Primary F1) 📊")
    print("="*85)
    print(var_df.to_string(formatters={
        'Mean': '{:,.4f}'.format, 'Variance': '{:,.4f}'.format,
        'Std_Dev': '{:,.4f}'.format, 'Min': '{:,.4f}'.format, 'Max': '{:,.4f}'.format
    }, index=False))
    print("="*85)

def generate_summary_table(df):
    """Outputs a clean summary CSV of the best methods per dataset scenario."""
    print(" -> Generating Summary Table...")
    
    idx = df.groupby('Dataset_Scenario')['Primary_F1'].idxmax()
    best_df = df.loc[idx, ['Dataset_Scenario', 'Method', 'Primary_F1', 'Nodes', 'Lexical Overlap', 'Avg Hearst PPL']]
    
    best_df = best_df.sort_values(by="Dataset_Scenario").reset_index(drop=True)
    best_df.columns = ["Dataset Scenario", "Best Method", "Best F1 Score", "Total Nodes", "Lexical Overlap", "Avg Hearst PPL"]
    
    best_df.to_csv(os.path.join(VIS_DIR, "5_best_methods_summary.csv"), index=False)
    print("\n" + "="*95)
    print("🏆 BEST PERFORMING METHODS PER DATASET SCENARIO 🏆")
    print("="*95)
    print(best_df.to_string(index=False))
    print("="*95)

def main():
    if not os.path.exists("benchmark_results.json") or not os.path.exists("dataset_metrics.csv"):
        print("Error: Required files 'benchmark_results.json' or 'dataset_metrics.csv' not found.")
        print("Please ensure you have run both benchmark pipelines in this directory.")
        return

    df = load_and_merge_data()
    
    plot_method_vs_dataset_heatmap(df)
    plot_method_variance(df)
    plot_metric_vs_f1_scatter_grid(df)
    plot_f1_metric_correlation_heatmap(df)
    plot_f1_metric_comparison(df) 
    report_method_variance(df)
    generate_summary_table(df)
    
    print(f"\n[*] All visualizations successfully saved to the '{VIS_DIR}/' directory.")

if __name__ == "__main__":
    main()
