# figscripts — paper figure generators

Every figure in the manuscript is reproducible from data on disk: either by a tool in the
repo root (listed first) or by a script in this directory. Nothing depends on a chat
session. All scripts here read campaign data from `$TAXO_BASE` (default: the analysis
machine's repo checkout, `C:/Users/Hoengager/Documents/Claude/TaxonomyConstruction`) and
write into `paper_figures/` (V2-reference variants into `paper_figures/v2/`). Run with
plain `python <script>.py`; matplotlib + networkx (+ scipy for stats) suffice unless noted.

## Figures made by repo-root tools (not duplicated here)

| Figure | Tool | Invocation notes |
|---|---|---|
| base_comparison.png (fig:crossmodel) | `base_comparison.py` | `--run LABEL:results.json` per model, `--ref` for paired tests; one json per run (see merged-core note below) |
| sc_vs_ours_gemma4.png (fig:sc-vs-ours) | `base_comparison.py` | runs = structured json + single-call json, ref = structured |
| judge_agreement.png (fig:judgeagreement) | `judge_agreement.py` | `--types_csv` per judge audit CSV; `--results_dir` must hold the GT_*_eval.graphml + diagnostics |
| relation_types.png (fig:relationtypes) | `relation_type_audit.py` | written as `--out_png` during an audit run |
| isa_rescore data | `isa_rescore.py` | one run per filter (`--types_csv` once = single judge; twice = unanimity); rendered by rescore_delta_vis*.py here |
| 18_heuristic_informativeness, 16_precision_clawback, strategy_rank_by, 20_restructure, reconstruction | repo analysis tools (`fp_reason_analysis.py`, `strategy_comparison.py`, `prune_sweep_analysis.py`, `synonym_reconstruction.py`) | see each tool's `--help` |

Merged-core note: when a "run" spans two campaigns (e.g. legacy WNF/CL/SemEval +
SchemaOrgV2/POV2/OBIV2/MatOntoV2), build a merged json first: concatenate the dataset
blocks, keeping exactly one result entry (the wanted method) per block.

## Scripts in this directory

| Script | Figure(s) | Input data |
|---|---|---|
| `scaling_vis.py` | scaling_f1.png, scaling_cost.png | results/scaling_20261001/results/*.json + legacy control/sc jsons (N=100 points) |
| `knee_scale_vis.py` | density_knee_scale.png | results/mech_20260909/density_gemma4.json + results/scaling_20261001/results/density_sub200.json |
| `density_vis.py` | density_gemma4.png | results/mech_20260909/density_gemma4.json |
| `khop_vis.py` | khop_recall.png (three-way, N=100) | results/battery_20260925/khop_setoverlap.csv (frozen CSV incl. matched rows) |
| `khop200.py` | khop_sub200.png (+ builds khop_sub200.csv) | scaling campaign preds + benchmark graphs |
| `khop_mechanism_data.py` | builds results/scaling_20261001/khop_mechanism.csv | battery + scaling single-call preds |
| `khop_mechanism_vis.py` | khop_mechanism.png | khop_mechanism.csv |
| `m_recommend_split.py` | m_recommendation_gemma.png, m_recommendation_gemini.png | density sweep jsons + scaling fits |
| `m_recommend_simple.py` | m_recommendation_gemma_simple.png (main-text variant) | same |
| `wording_vis.py` | prompt_wording_ablation.png | legacy prompt_ablation/variants jsons |
| `selfpref_vis.py` | judge_selfpreference.png | results/x2_20260923/rt_*.csv (legacy); V2 variant pending the Gemma-preds audit legs |
| `normalizer_vis.py` | method_normalization.png | 4-model control + single-call jsons (needs gpt-oss/Qwen on V2 for a V2 variant) |
| `rescore_delta_vis.py` | isa_rescore_deltas.png (legacy) | results/rescore3_20260923/rescore_{self,cross,consensus}.csv |
| `rescore_delta_vis_v2.py` | paper_figures/v2/isa_rescore_deltas.png | results/v2phase2_20261016/rescore/rescore_*.csv |
| `v2_figures.py` | paper_figures/v2/{reference_repair, method_comparison_v2, audit_v2}.png | v2phase1 + v2phase2 campaign dirs + legacy jsons |
| `synset_dupes_build.py` | (analysis) results/synset_dupes.csv | rebuilds synset graphs from cached sources; needs nltk + obonet |

## Known gaps

- `f1_condition_deltas.png` / `f1_condition_raw_bars.png`: generators were one-off scripts
  predating this collection and were not preserved; rewrite from the per-mode F1 fields in
  the results jsons if those figures ever need regeneration.
- The N=100 three-way k-hop CSV (`battery_20260925/khop_setoverlap.csv`) is a frozen
  artifact; to rebuild from scratch, adapt `khop200.py` to the SUB (N=100) paths and append
  the matched-call rows.
