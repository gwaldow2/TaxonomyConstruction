"""Self-checks for run provenance (evaluator tagging + saved prediction graphs)
and alt_metrics. No GPU, no server, no LLM calls.

    python test_provenance.py
"""

import os
import glob
import shutil
import tempfile

import networkx as nx

import evaluator
from evaluator import evaluate_all_modes
from alt_metrics import alt_metrics, map_edges_to_gt, dataset_from_filename, selftest


def _graphs():
    G_gt = nx.DiGraph([("food", "fruit"), ("fruit", "apple")])
    G_pred = nx.DiGraph([("food", "fruit"), ("fruit", "apple"), ("food", "apple")])
    return G_pred, G_gt


def test_tagged_outputs_do_not_collide():
    """Two runs with the same method label but different tags must write disjoint files
    -- the failure mode that let successive models overwrite each other's outputs."""
    tmp = tempfile.mkdtemp()
    try:
        G_pred, G_gt = _graphs()
        for tag in ("modelA", "modelB"):
            evaluator.RUN_META = {"model": f"vendor/{tag}", "tag": tag,
                                  "results_file": "x.json", "timestamp": "t"}
            evaluate_all_modes(G_pred.copy(), G_gt.copy(), os.path.join(tmp, "D_Our_Method"))
        preds = sorted(glob.glob(os.path.join(tmp, "*_pred.graphml")))
        assert len(preds) == 2, preds
        txts = glob.glob(os.path.join(tmp, "*modelA*condensed_closure.txt"))
        assert txts, "report txts must carry the tag too"
        G = nx.DiGraph(nx.read_graphml(preds[0]))
        assert G.graph.get("model") == "vendor/modelA"
        assert set(G.edges()) == set(G_pred.edges()), "saved graph must be the scored graph"
    finally:
        evaluator.RUN_META = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_untagged_run_keeps_legacy_filenames():
    tmp = tempfile.mkdtemp()
    try:
        evaluator.RUN_META = {}
        G_pred, G_gt = _graphs()
        evaluate_all_modes(G_pred, G_gt, os.path.join(tmp, "D_Our_Method"))
        assert os.path.exists(os.path.join(tmp, "D_Our_Method_pred.graphml"))
        assert os.path.exists(os.path.join(tmp, "D_Our_Method_condensed_closure.txt"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_alt_metrics_selftest():
    selftest()


def test_lemma_normalized_mapping():
    """Predicted terms in lemma format must map onto GT nodes (the mismatch that
    silently dropped a third of Gemma's edges in the first k-hop prototype)."""
    G_gt = nx.DiGraph([("cell (cell, eukaryotic cell)", "neuron")])
    P, unmapped = map_edges_to_gt([("cell (cell, eukaryotic cell)", "neuron"),
                                   ("cell", "neuron")], G_gt)
    assert unmapped == 0
    assert P.number_of_edges() == 1


def test_dataset_from_filename_prefers_longest_match():
    tmp = tempfile.mkdtemp()
    try:
        for ds in ("D", "D_SUB"):
            nx.write_graphml(nx.DiGraph([("a", "b")]), os.path.join(tmp, f"GT_{ds}_eval.graphml"))
        assert dataset_from_filename(os.path.join(tmp, "D_SUB_Our_Method_pred.graphml")) == "D_SUB"
        assert dataset_from_filename(os.path.join(tmp, "D_Our_Method_pred.graphml")) == "D"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll provenance checks passed.")
