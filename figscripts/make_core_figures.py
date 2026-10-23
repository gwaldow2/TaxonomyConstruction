"""One command: core-dataset runs in, paper figures out.

Runs the existing generators in order against $TAXO_V2_CAMP (default
results/v2r2_20261021). Audit-dependent figures are not included; they need the
judge-audit legs first.

    python figscripts/make_core_figures.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
CC = f"{BASE}/results/corrected_core"

STEPS = [
    [sys.executable, f"{HERE}/merge_corrected_core.py"],
    [sys.executable, f"{HERE}/method_comparison_vis.py"],
    [sys.executable, f"{REPO}/base_comparison.py",
     "--run", f"structured:{CC}/gemma4_structured.json",
     "--run", f"single-call:{CC}/gemma4_singlecall.json",
     "--ref", "structured", "--out", f"{BASE}/paper_figures/sc_vs_ours_gemma4.png"],
    [sys.executable, f"{HERE}/khop_core.py"],
    [sys.executable, f"{REPO}/base_comparison.py",
     "--run", f"gemma4:{CC}/gemma4_structured.json",
     "--run", f"gemini:{CC}/gemini_structured.json",
     "--run", f"qwen:{CC}/qwen_structured.json",
     "--run", f"gptoss:{CC}/gptoss_structured.json",
     "--ref", "gemma4", "--out", f"{BASE}/paper_figures/base_comparison.png"],
    [sys.executable, f"{HERE}/normalizer_vis.py"],
    [sys.executable, f"{HERE}/selfpref_vis.py"],
    [sys.executable, f"{HERE}/rescore_delta_vis.py"],
    [sys.executable, f"{REPO}/base_comparison.py",
     "--run", f"Gemma4-31B:{CC}/gemma4_singlecall.json",
     "--run", f"gpt-oss-120B:{CC}/gptoss_singlecall.json",
     "--run", f"Qwen2.5-7B:{CC}/qwen_singlecall.json",
     "--run", f"Gemini 3.1 Pro:{CC}/gemini_singlecall.json",
     "--ref", "Gemma4-31B", "--out", f"{BASE}/paper_figures/sc_all_models.png"],
    [sys.executable, f"{HERE}/relation_types_core.py"],
    [sys.executable, f"{HERE}/heuristic_vis.py"],
]

for cmd in STEPS:
    print(f"\n=== {os.path.basename(cmd[1])} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
print("\n[*] core figures written to", f"{BASE}/paper_figures")
