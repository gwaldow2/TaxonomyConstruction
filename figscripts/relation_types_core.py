"""Relation-type composition on the seven-dataset core (fig:relationtypes).

Feeds the merged corrected-core audit rows (Gemma judge, the tool's default judge,
matching the legacy figure) through relation_type_audit.summarize, which prints the
proportions and draws the stacked-bar figure.
"""
import csv
import os
import sys

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relation_type_audit import summarize  # noqa: E402

rows = list(csv.DictReader(open(f"{BASE}/results/corrected_core/types_gemmaJ.csv", encoding="utf-8")))
summarize(rows,
          f"{BASE}/results/corrected_core/relation_types_summary.csv",
          f"{BASE}/paper_figures/relation_types.png")
