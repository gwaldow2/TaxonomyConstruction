"""Build results/corrected_core/: the canonical 7-dataset core with the four
LLMs4OL-derived ground truths replaced by their full-source rebuilds.

The rebuilt datasets (SchemaOrgV2/POV2/OBIV2/MatOntoV2) are renamed back to the
canonical dataset names so every downstream figure renders identically to before
the reference fix. Legacy WordNetFood/CellOntology/SemEvalFood rows are untouched
by the fix and are carried over from the original campaigns.

Outputs (under $TAXO_BASE/results/corrected_core/):
  gemma4_structured.json / gemma4_singlecall.json
  gemini_structured.json / gemini_singlecall.json
  baselines.json
  rescore/rescore_{self,cross,consensus}.csv
  types_gemmaJ.csv / types_geminiJ.csv        (judge audit rows, gt side complete;
                                               gemma-pred rows pending the audit
                                               repair legs on the server)
  gt/GT_{ds}_SUB_eval.graphml                 (all seven, corrected refs for the four)
"""
import csv
import json
import os
import shutil

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
RJ = f"{BASE}/results/results_json"
P1 = f"{BASE}/results/v2phase1_20261016/results"
P2 = f"{BASE}/results/v2phase2_20261016/results"
OUT = f"{BASE}/results/corrected_core"
os.makedirs(f"{OUT}/rescore", exist_ok=True)
os.makedirs(f"{OUT}/gt", exist_ok=True)

LEG3 = ["WordNetFood_SUB", "CellOntology_SUB", "SemEvalFood_SUB"]
REN = {"SchemaOrgV2_SUB": "LLMs4OL_SchemaOrg_SUB", "POV2_SUB": "LLMs4OL_PO_SUB",
       "OBIV2_SUB": "LLMs4OL_OBI_SUB", "MatOntoV2_SUB": "LLMs4OL_MatOnto_SUB"}
ORDER = ["WordNetFood_SUB", "CellOntology_SUB", "LLMs4OL_OBI_SUB", "LLMs4OL_PO_SUB",
         "SemEvalFood_SUB", "LLMs4OL_SchemaOrg_SUB", "LLMs4OL_MatOnto_SUB"]


def blocks(path, datasets=None, method_sub=None, rename=False):
    out = []
    for b in json.load(open(path, encoding="utf-8")):
        ds = b["dataset"]
        if rename:
            if ds not in REN:
                continue
            ds = REN[ds]
        elif datasets is not None and ds not in datasets:
            continue
        res = [r for r in b["results"] if method_sub is None or method_sub in r["method"]]
        if res:
            out.append({**b, "dataset": ds, "results": res})
    return out


def merged_json(name, leg_path, v2_path, method_sub):
    leg = blocks(leg_path, datasets=LEG3, method_sub=method_sub)
    v2 = blocks(v2_path, method_sub=method_sub, rename=True)
    got = {b["dataset"] for b in leg + v2}
    missing = [d for d in ORDER if d not in got]
    if missing:
        print(f"[!] {name}: missing {missing}")
    merged = sorted(leg + v2, key=lambda b: ORDER.index(b["dataset"]))
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1)
    print(f"wrote {name}: {len(merged)} datasets")


merged_json("gemma4_structured.json", f"{RJ}/control_gemma4N.json",
            f"{P1}/v2_phase1.json", "Our Method")
merged_json("gemma4_singlecall.json", f"{RJ}/sc_gemma4_prim_merge.json",
            f"{P1}/v2_phase1.json", "Single-Call")
merged_json("gemini_structured.json", f"{RJ}/control_gemini.json",
            f"{P2}/v2_phase2_gemini.json", "Our Method")
merged_json("gemini_singlecall.json", f"{RJ}/sc_gemini_prim_merge.json",
            f"{P2}/v2_phase2_gemini.json", "Single-Call")
merged_json("baselines.json", f"{BASE}/results/baselines_20261002/results/baselines_7core.json",
            f"{P2}/v2_baselines.json", None)

# ---- rescore CSVs (per-dataset rows; identical schema across campaigns) ----
for variant in ("self", "cross", "consensus"):
    rows = []
    for r in csv.DictReader(open(f"{BASE}/results/rescore3_20260923/rescore_{variant}.csv", encoding="utf-8")):
        if r["dataset"] in LEG3:
            rows.append(r)
    for r in csv.DictReader(open(f"{BASE}/results/v2phase2_20261016/rescore/rescore_{variant}.csv", encoding="utf-8")):
        r["dataset"] = REN[r["dataset"]]
        rows.append(r)
    rows.sort(key=lambda r: ORDER.index(r["dataset"]))
    with open(f"{OUT}/rescore/rescore_{variant}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote rescore_{variant}.csv: {len(rows)} rows")

# ---- judge audit CSVs (gt+pred rows; v2 dataset names mapped back) ----
AUD = [("types_gemmaJ.csv", f"{BASE}/results/x2_20260923/rt_gemmaJ_gemma4P.csv",
        [f"{P2}/rt_gemmaJ_v2_gemma4P.csv", f"{P2}/rt_gemmaJ_v2_gemini31P.csv"]),
       ("types_geminiJ.csv", f"{BASE}/results/x2_20260923/rt_geminiJ_gemma4P.csv",
        [f"{P2}/rt_geminiJ_v2_gemma4P.csv", f"{P2}/rt_geminiJ_v2_gemini31P.csv"])]
for name, leg_path, v2_paths in AUD:
    rows = [r for r in csv.DictReader(open(leg_path, encoding="utf-8")) if r["dataset"] in LEG3]
    seen = {(r["dataset"], r["parent"], r["child"], r.get("source")) for r in rows}
    for p in v2_paths:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            r["dataset"] = REN.get(r["dataset"], r["dataset"])
            key = (r["dataset"], r["parent"], r["child"], r.get("source"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gt = sum(1 for r in rows if r.get("source") == "gt")
    pred = len(rows) - gt
    print(f"wrote {name}: {gt} gt rows, {pred} pred rows")

# ---- GT eval graphs for all seven (corrected refs for the four) ----
for ds in LEG3:
    shutil.copy(f"{BASE}/results/battery_20260925/preds/GT_{ds}_eval.graphml",
                f"{OUT}/gt/GT_{ds}_eval.graphml")
for v2, leg in REN.items():
    shutil.copy(f"{P2}/GT_{v2}_eval.graphml", f"{OUT}/gt/GT_{leg}_eval.graphml")
print("wrote gt/: 7 GT eval graphs")
