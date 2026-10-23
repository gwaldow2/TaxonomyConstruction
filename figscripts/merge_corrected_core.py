"""Build results/corrected_core/: the canonical 7-dataset core with the four
LLMs4OL-derived ground truths replaced by their full-source rebuilds.

The rebuilt datasets (SchemaOrgV2/POV2/OBIV2/MatOntoV2) are renamed back to the
canonical dataset names so every downstream figure renders identically to before
the reference fix. Legacy WordNetFood/CellOntology/SemEvalFood rows are untouched
by the fix and are carried over from the original campaigns.

The rebuilt-four data comes from ONE campaign directory ($TAXO_V2_CAMP, default
results/v2r2_20261021: the rerun after the PR #79/#80 parser fixes). Products whose
inputs that campaign has not produced yet (judge audits, is-a rescoring) are NOT
built, and any stale copies in corrected_core/ are removed, so an older
campaign's judgments can never be silently mixed with the current references.

Outputs (under $TAXO_BASE/results/corrected_core/):
  gemma4_structured.json / gemma4_singlecall.json
  gemini_structured.json / gemini_singlecall.json
  baselines.json
  gt/GT_{ds}_SUB_eval.graphml                 (all seven)
  rescore/rescore_{self,cross,consensus}.csv  (only once the campaign has rescore/)
  types_gemmaJ.csv / types_geminiJ.csv        (only once all four audit legs exist)
"""
import csv
import json
import os
import shutil

BASE = os.environ.get("TAXO_BASE", "C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction")
RJ = f"{BASE}/results/results_json"
CAMP = os.environ.get("TAXO_V2_CAMP", f"{BASE}/results/v2r2_20261021")
CR = f"{CAMP}/results"
OUT = f"{BASE}/results/corrected_core"
os.makedirs(f"{OUT}/gt", exist_ok=True)

LEG3 = ["WordNetFood_SUB", "CellOntology_SUB", "SemEvalFood_SUB"]
REN = {"SchemaOrgV2_SUB": "LLMs4OL_SchemaOrg_SUB", "POV2_SUB": "LLMs4OL_PO_SUB",
       "OBIV2_SUB": "LLMs4OL_OBI_SUB", "MatOntoV2_SUB": "LLMs4OL_MatOnto_SUB"}
ORDER = ["WordNetFood_SUB", "CellOntology_SUB", "LLMs4OL_OBI_SUB", "LLMs4OL_PO_SUB",
         "SemEvalFood_SUB", "LLMs4OL_SchemaOrg_SUB", "LLMs4OL_MatOnto_SUB"]
print(f"[*] rebuilt-four campaign: {CAMP}")


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
        res = [r for r in b["results"]
               if method_sub is None
               or (method_sub in r["method"] and "[matched]" not in r["method"])]
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


def retire(path, why):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"[-] removed stale {os.path.relpath(path, OUT)} ({why})")


merged_json("gemma4_structured.json", f"{RJ}/control_gemma4N.json",
            f"{CR}/v2r2_gemma.json", "Our Method")
merged_json("gemma4_singlecall.json", f"{RJ}/sc_gemma4_prim_merge.json",
            f"{CR}/v2r2_gemma.json", "Single-Call")
merged_json("gemini_structured.json", f"{RJ}/control_gemini.json",
            f"{CR}/v2r2_gemini.json", "Our Method")
merged_json("gemini_singlecall.json", f"{RJ}/sc_gemini_prim_merge.json",
            f"{CR}/v2r2_gemini.json", "Single-Call")
merged_json("baselines.json", f"{BASE}/results/baselines_20261002/results/baselines_7core.json",
            f"{CR}/v2r2_baselines.json", None)

# ---- GT eval graphs for all seven (corrected refs for the four) ----
for ds in LEG3:
    shutil.copy(f"{BASE}/results/battery_20260925/preds/GT_{ds}_eval.graphml",
                f"{OUT}/gt/GT_{ds}_eval.graphml")
for v2, leg in REN.items():
    shutil.copy(f"{CR}/GT_{v2}_eval.graphml", f"{OUT}/gt/GT_{leg}_eval.graphml")
print("wrote gt/: 7 GT eval graphs")

# ---- rescore CSVs: only from this campaign's own rescore/ ----
resc_dir = f"{CAMP}/rescore"
if all(os.path.exists(f"{resc_dir}/rescore_{v}.csv") for v in ("self", "cross", "consensus")):
    os.makedirs(f"{OUT}/rescore", exist_ok=True)
    for variant in ("self", "cross", "consensus"):
        rows = [r for r in csv.DictReader(open(f"{BASE}/results/rescore3_20260923/rescore_{variant}.csv",
                                               encoding="utf-8")) if r["dataset"] in LEG3]
        for r in csv.DictReader(open(f"{resc_dir}/rescore_{variant}.csv", encoding="utf-8")):
            r["dataset"] = REN[r["dataset"]]
            rows.append(r)
        rows.sort(key=lambda r: ORDER.index(r["dataset"]))
        with open(f"{OUT}/rescore/rescore_{variant}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote rescore_{variant}.csv: {len(rows)} rows")
else:
    print(f"[ ] rescore pending: {resc_dir}/rescore_*.csv not found")
    retire(f"{OUT}/rescore", "predates current references")

# ---- judge audit CSVs: only when all four legs of this campaign exist ----
legs = {(j, p): f"{CR}/rt_{j}_v2r2_{p}.csv"
        for j in ("gemmaJ", "geminiJ") for p in ("gemma4P", "gemini31P")}
missing = [os.path.basename(p) for p in legs.values() if not os.path.exists(p)]
if not missing:
    # per-leg 2x2 CSVs (judge x pred-set), for the self-preference figure
    for (judge, pred), path in legs.items():
        leg_path = f"{BASE}/results/x2_20260923/rt_{judge}_{pred}.csv"
        rows = [r for r in csv.DictReader(open(leg_path, encoding="utf-8")) if r["dataset"] in LEG3]
        for r in csv.DictReader(open(path, encoding="utf-8")):
            r["dataset"] = REN.get(r["dataset"], r["dataset"])
            rows.append(r)
        with open(f"{OUT}/rt_{judge}_{pred}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote rt_{judge}_{pred}.csv: {len(rows)} rows")
    for path in legs.values():
        n_pred = sum(1 for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("source") == "pred")
        if n_pred == 0:
            raise SystemExit(f"[!] {os.path.basename(path)} has 0 pred rows: the audit ran without "
                             f"matching diagnostics (check --pred_label). Refusing to merge.")
    for judge, leg_path in (("gemmaJ", f"{BASE}/results/x2_20260923/rt_gemmaJ_gemma4P.csv"),
                            ("geminiJ", f"{BASE}/results/x2_20260923/rt_geminiJ_gemma4P.csv")):
        rows = [r for r in csv.DictReader(open(leg_path, encoding="utf-8")) if r["dataset"] in LEG3]
        seen = {(r["dataset"], r["parent"], r["child"], r.get("source")) for r in rows}
        for pred in ("gemma4P", "gemini31P"):
            for r in csv.DictReader(open(legs[(judge, pred)], encoding="utf-8")):
                r["dataset"] = REN.get(r["dataset"], r["dataset"])
                key = (r["dataset"], r["parent"], r["child"], r.get("source"))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
        gt = sum(1 for r in rows if r.get("source") == "gt")
        with open(f"{OUT}/types_{judge}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote types_{judge}.csv: {gt} gt rows, {len(rows) - gt} pred rows")
else:
    print(f"[ ] judge audits pending: {missing}")
    for judge in ("gemmaJ", "geminiJ"):
        retire(f"{OUT}/types_{judge}.csv", "predates current references")
        for pred in ("gemma4P", "gemini31P"):
            retire(f"{OUT}/rt_{judge}_{pred}.csv", "predates current references")
    for stale in ("judge_agreement.csv", "consensus_contamination.csv", "votes.csv"):
        retire(f"{OUT}/{stale}", "derived from stale audits")
