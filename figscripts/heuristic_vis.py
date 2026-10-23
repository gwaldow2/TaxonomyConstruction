"""Heuristic-informativeness figure (18_heuristic_informativeness) on the corrected core.

Stages the seven Gemma structured-run edge-diagnostics files (the same runs behind
the tables: control-era files for the legacy three, $TAXO_V2_CAMP for the rebuilt
four) into corrected_core/diag/, then calls the repo tool
vis_benchmarks.plot_heuristic_informativeness on that directory and moves its
output into paper_figures/.
"""
import os
import shutil
import sys

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
CAMP = os.environ.get("TAXO_V2_CAMP", f"{BASE}/results/v2r2_20261021")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vis_benchmarks  # noqa: E402

DIAG = f"{BASE}/results/corrected_core/diag"
os.makedirs(DIAG, exist_ok=True)
for ds in ("WordNetFood", "CellOntology", "SemEvalFood"):
    shutil.copy(f"{BASE}/results/rescore3_20260923/work/{ds}_SUB_Our_Method_gemma4_edge_diagnostics.csv",
                f"{DIAG}/{ds}_SUB_Our_Method_gemma4_edge_diagnostics.csv")
for ds in ("SchemaOrgV2", "POV2", "OBIV2", "MatOntoV2"):
    shutil.copy(f"{CAMP}/results/{ds}_SUB_Our_Method_gemma4_v2_edge_diagnostics.csv",
                f"{DIAG}/{ds}_SUB_Our_Method_gemma4_v2_edge_diagnostics.csv")

vis_benchmarks.VIS_DIR = f"{BASE}/paper_figures"
vis_benchmarks.plot_heuristic_informativeness(results_dir=DIAG)
print(f"wrote {BASE}/paper_figures/18_heuristic_informativeness.png")
