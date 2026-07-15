"""Offline prune-sweep analysis -- heuristic-only cutting vs the LLM prune, no LLM calls.

For each dataset it reconstructs the extracted graph from its raw edge-diagnostics CSV
(results/<ds>_Our_Method_edge_diagnostics.csv, which already carries the per-edge suspicion
components + an is_fp label) and the GT graph (results/GT_<ds>_eval.graphml), then sweeps
K = number of top-suspicious edges removed and evaluates the resulting **Cond. Closure**
P/R/F1 -- exactly as evaluator.compute_and_save_metrics does (set_overlap on synonyms) -- for:

  * each --rank_by heuristic (combined / leverage / neighborhood / self_agreement / salience),
    cutting the top-K by that score with NO model in the loop,
  * an ORACLE that removes the K highest-leverage TRUE false positives (the ceiling), and
  * the baseline (K=0).

Optionally overlays the LLM restructure_prune_only runs from a results JSON, so you can see
whether the heuristic-only cut ever beats the LLM prune. Also plots the false-positive MASS
concentration (how few edges generate most of the closure FPs).

    python prune_sweep_analysis.py --results_dir results --llm_results restr_method.json

Outputs vis/prune_sweep.png.
"""

import os
import re
import csv
import glob
import json
import argparse

import networkx as nx

VIS_DIR = "vis"
RANK_BYS = ("combined", "leverage", "neighborhood", "self_agreement", "salience")


# ---- pure copies of the two evaluator helpers (avoid importing data_manager/evaluator,
#      which pull in nltk/obonet/requests) ----
def parse_lemma_format(node_str):
    node_str = str(node_str).strip().lower()
    match = re.search(r'^([^(]+)\s*\((.*)\)$', node_str)
    if match:
        primary_text = match.group(1).strip()
        syns = [s.strip() for s in match.group(2).split(',') if s.strip()]
        if syns and syns[0] == primary_text:
            return syns
    if '|' in node_str:
        return [s.strip() for s in node_str.split('|') if s.strip()]
    return [node_str]


def _exploded_pairs(edges):
    pairs = set()
    for u, v in edges:
        for tu in parse_lemma_format(u):
            for tv in parse_lemma_format(v):
                pairs.add((tu, tv))
    return pairs


def _matches(u, v, pairs):
    for tu in parse_lemma_format(u):
        for tv in parse_lemma_format(v):
            if (tu, tv) in pairs:
                return True
    return False


def _pct_rank(values):
    n = len(values)
    if n == 0: return []
    if n == 1: return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for r, i in enumerate(order):
        ranks[i] = r / (n - 1)
    return ranks


def score_edges(rows, rank_by):
    """{edge: score} under rank_by, from the diagnostics components (matches edge_suspicion_scores)."""
    presets = {"combined": (1, 1, 1, 1), "leverage": (1, 0, 0, 0), "neighborhood": (0, 1, 0, 0),
               "self_agreement": (0, 0, 1, 0), "salience": (0, 0, 0, 1)}
    wl, wn, wa, ws = presets[rank_by]
    lev_p = _pct_rank([r["leverage"] for r in rows])
    sal_p = _pct_rank([r["salience"] for r in rows])
    out = {}
    for i, r in enumerate(rows):
        low_agree = 1.0 if r["votes"] <= 1 else 0.0
        out[(r["parent"], r["child"])] = (wl * lev_p[i] + wn * (1.0 - r["neighborhood_agreement"])
                                          + wa * low_agree + ws * (1.0 - sal_p[i]))
    return out


_FEATURES = ("leverage", "neighborhood", "self_agreement", "salience")

def edge_features(rows):
    """-> (edges, feats): per-edge suspicion features (same 4 signals, suspicion direction,
    leverage/salience percentile-normalised within the dataset)."""
    lev_p = _pct_rank([r["leverage"] for r in rows])
    sal_p = _pct_rank([r["salience"] for r in rows])
    edges, feats = [], []
    for i, r in enumerate(rows):
        edges.append((r["parent"], r["child"]))
        feats.append([lev_p[i], 1.0 - r["neighborhood_agreement"],
                      1.0 if r["votes"] <= 1 else 0.0, 1.0 - sal_p[i]])
    return edges, feats


def _fit_logreg(X, y, iters=500, lr=0.5, l2=1e-2):
    """Plain-Python logistic regression (no sklearn/numpy). Returns weights [bias, *coef]."""
    import math
    n = len(X); m = len(X[0]) if X else 0
    w = [0.0] * (m + 1)
    for _ in range(iters):
        g = [0.0] * (m + 1)
        for xi, yi in zip(X, y):
            z = w[0] + sum(w[j + 1] * xi[j] for j in range(m))
            pi = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            e = pi - yi
            g[0] += e
            for j in range(m):
                g[j + 1] += e * xi[j]
        w[0] -= lr * g[0] / max(1, n)
        for j in range(m):
            w[j + 1] -= lr * (g[j + 1] / max(1, n) + l2 * w[j + 1])
    return w


def _proba(w, x):
    import math
    z = w[0] + sum(w[j + 1] * x[j] for j in range(len(x)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def auc(scores, labels):
    """ROC AUC = P(score(FP) > score(TP)), tie-aware. NaN if a class is empty."""
    data = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(data)
    ranks = [0.0] * n
    j = 0
    while j < n:
        k = j
        while k + 1 < n and data[k + 1][0] == data[j][0]:
            k += 1
        avg = (j + k) / 2.0 + 1.0
        for t in range(j, k + 1):
            ranks[t] = avg
        j = k + 1
    n_pos = sum(1 for _, l in data if l == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(r for r, (_, l) in zip(ranks, data) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cond_clos(G_pred, gt_pairs, gt_edges):
    """Cond. Closure (P, R, F1, n_fp) of G_pred vs the GT, set_overlap on synonyms."""
    pred_edges = list(nx.transitive_closure(G_pred).edges()) if G_pred.number_of_edges() else []
    if not pred_edges:
        return (0.0, 0.0, 0.0, 0)
    pred_pairs = _exploded_pairs(pred_edges)
    tp_pred = sum(1 for (u, v) in pred_edges if _matches(u, v, gt_pairs))
    tp_gt = sum(1 for (u, v) in gt_edges if _matches(u, v, pred_pairs))
    p = tp_pred / len(pred_edges)
    r = tp_gt / len(gt_edges) if gt_edges else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return (p, r, f1, len(pred_edges) - tp_pred)


def load_dataset(csv_path, gt_dir):
    """-> (dataset, rows, G_pred, gt_pairs, gt_edges) or None if the GT graphml is missing."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            rows.append({"parent": d["parent"], "child": d["child"],
                         "leverage": float(d["leverage"]), "neighborhood_agreement": float(d["neighborhood_agreement"]),
                         "votes": float(d["votes"]), "salience": float(d["salience"]), "is_fp": int(d["is_fp"])})
    if not rows:
        return None
    dataset = rows[0].get("dataset") if "dataset" in rows[0] else None
    # dataset from the filename: <ds>_Our_Method_edge_diagnostics.csv
    ds = re.sub(r"_Our_Method_edge_diagnostics\.csv$", "", os.path.basename(csv_path))
    gt_path = os.path.join(gt_dir, f"GT_{ds}_eval.graphml")
    if not os.path.exists(gt_path):
        print(f"    [!] missing GT graphml for {ds}: {gt_path} -- skipping")
        return None
    G_gt = nx.DiGraph(nx.read_graphml(gt_path))
    if "virtual_root" in G_gt:
        G_gt.remove_node("virtual_root")
    gt_edges = list(nx.transitive_closure(G_gt).edges())
    gt_pairs = _exploded_pairs(gt_edges)
    G_pred = nx.DiGraph()
    G_pred.add_edges_from([(r["parent"], r["child"]) for r in rows])
    return (ds, rows, G_pred, gt_pairs, gt_edges)


def sweep_dataset(rows, G_pred, gt_pairs, gt_edges, ks, extra_scores=None):
    """-> {'by'->{ranker->{K->(p,r,f1)}}, 'oracle'->{K->(p,r,f1)}, 'fp_curve'->[...]}.

    extra_scores: optional {name: {edge: score}} rankers (e.g. the learned P(FP)) added
    alongside the built-in --rank_by heuristics."""
    base_p, base_r, base_f1, base_fp = cond_clos(G_pred, gt_pairs, gt_edges)
    res = {"baseline": (base_p, base_r, base_f1), "by": {}, "oracle": {}}
    rankers = {rb: score_edges(rows, rb) for rb in RANK_BYS}
    if extra_scores:
        rankers.update(extra_scores)
    for name, scores in rankers.items():
        ranked = [e for e, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        res["by"][name] = {}
        for K in ks:
            G = G_pred.copy(); G.remove_edges_from(ranked[:K])
            res["by"][name][K] = cond_clos(G, gt_pairs, gt_edges)[:3]
    # oracle: remove the K highest-leverage TRUE FPs
    fp_by_lev = [ (r["parent"], r["child"]) for r in sorted([x for x in rows if x["is_fp"] == 1],
                                                            key=lambda x: x["leverage"], reverse=True) ]
    for K in ks:
        G = G_pred.copy(); G.remove_edges_from(fp_by_lev[:K])
        res["oracle"][K] = cond_clos(G, gt_pairs, gt_edges)[:3]
    # FP-mass concentration: fraction of baseline closure-FPs eliminated as we remove FP edges (oracle order)
    curve = []
    for n in range(0, len(fp_by_lev) + 1):
        G = G_pred.copy(); G.remove_edges_from(fp_by_lev[:n])
        _, _, _, n_fp = cond_clos(G, gt_pairs, gt_edges)
        frac_gone = 1.0 - (n_fp / base_fp) if base_fp else 0.0
        curve.append((n / len(fp_by_lev) if fp_by_lev else 0.0, frac_gone))
    res["fp_curve"] = curve
    return res


def load_llm_prune(results_path):
    """-> {(K, rank_by): (meanP, meanR, meanF1)} for restructure_prune_only runs in a results JSON."""
    if not results_path or not os.path.exists(results_path):
        return {}
    agg = {}
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    for block in data:
        for r in block.get("results", []):
            m = r.get("method", "")
            if "+restructure_prune_only" not in m:
                continue
            k = int(re.search(r"_top(\d+)", m).group(1)) if re.search(r"_top(\d+)", m) else 0
            rb = re.search(r"\+rank_(\w+)", m)
            rb = rb.group(1) if rb else "combined"
            agg.setdefault((k, rb), []).append((r.get("Cond_Clos_Precision"), r.get("Cond_Clos_Recall"), r.get("Cond_Clos_F1")))
    out = {}
    for key, vals in agg.items():
        vals = [v for v in vals if None not in v]
        if vals:
            out[key] = tuple(sum(v[i] for v in vals) / len(vals) for i in range(3))
    return out


def plot(agg, llm, ks, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rankers = list(RANK_BYS) + [k for k in agg[0]["by"] if k not in RANK_BYS]   # extras (learned, taxollama_*) last
    cmap = plt.get_cmap("tab10")
    color = {rb: cmap(i % 10) for i, rb in enumerate(rankers)}
    metric_idx = {"F1": 2, "Precision": 0, "Recall": 1}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panels = [("F1", axes[0, 0]), ("Precision", axes[0, 1]), ("Recall", axes[1, 0])]

    for name, ax in panels:
        mi = metric_idx[name]
        base_m = sum(a["baseline"][mi] for a in agg) / len(agg)
        ax.axhline(base_m, ls=":", color=".5", lw=1.5, label=f"baseline ({base_m:.3f})")
        for rb in rankers:
            ys = [sum(a["by"][rb][K][mi] for a in agg) / len(agg) for K in ks]
            emph = rb not in RANK_BYS            # learned + taxollama_* stand out
            lw = 2.8 if emph else 1.3
            mk = "s" if emph else "o"
            ax.plot(ks, ys, "-", marker=mk, ms=(4 if rb == "learned" else 3),
                    color=color[rb], label=rb, alpha=0.95, lw=lw)
        oy = [sum(a["oracle"][K][mi] for a in agg) / len(agg) for K in ks]
        ax.plot(ks, oy, "--", color="black", lw=2, label="oracle (cut true FPs)")
        # overlay LLM prune points
        for (k, rb), pr in (llm or {}).items():
            ax.scatter([k], [pr[mi]], marker="*", s=180, color=color.get(rb, "red"),
                       edgecolors="black", zorder=5, label=f"LLM prune [{rb}]")
        ax.set_title(f"Cond. Closure {name} vs edges removed (top-K)", fontweight="bold")
        ax.set_xlabel("K (edges cut)", fontweight="bold"); ax.set_ylabel(name, fontweight="bold")
        ax.set_ylim(0, 1.02)
        # de-dupe legend
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=2)

    # FP-mass concentration (mean curve over a common x-grid)
    axc = axes[1, 1]
    import numpy as np
    xg = np.linspace(0, 1, 21)
    curves = []
    for a in agg:
        xs = [p[0] for p in a["fp_curve"]]; ys = [p[1] for p in a["fp_curve"]]
        curves.append(np.interp(xg, xs, ys) if len(xs) > 1 else np.zeros_like(xg))
    mean_curve = np.mean(curves, axis=0)
    axc.plot(xg, mean_curve, "-o", color="#4C72B0", ms=3)
    axc.plot([0, 1], [0, 1], "--", color=".6", lw=1)   # diffuse reference (proportional)
    axc.set_title("FP-mass concentration (oracle order)\nsteep = few edges make most closure FPs",
                  fontweight="bold")
    axc.set_xlabel("fraction of FP edges removed", fontweight="bold")
    axc.set_ylabel("fraction of closure FPs eliminated", fontweight="bold")
    axc.set_xlim(0, 1); axc.set_ylim(0, 1.02)

    fig.suptitle("Prune sweep: heuristic-only cut vs LLM prune vs oracle (Cond. Closure)",
                 y=1.01, fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Optional TaxoLLaMA per-edge is-a plausibility (perplexity) as an FP signal
# ----------------------------------------------------------------------------
def _primary(node):
    return parse_lemma_format(node)[0]


def taxollama_needed_pairs(D):
    """Ordered (hyponym, hypernym) primary-term pairs needed: for each edge parent->child
    the forward (child, parent) and the reverse (parent, child)."""
    pairs = set()
    for d in D:
        for (p, c) in d["edges"]:
            pp, cc = _primary(p), _primary(c)
            pairs.add((cc, pp)); pairs.add((pp, cc))
    return pairs


def _load_taxollama(device):
    """Load Llama-2-7b + the TaxoLLaMA adapter (4-bit) and return a masked-PPL scorer --
    the same prompt/masking as taxollama_method.precompute_taxollama_ppl."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    base_id, adapter_id = "meta-llama/Llama-2-7b-hf", "VityaVitalich/TaxoLLaMA"
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base_id, token=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=qc, device_map="auto", token=True)
    model = PeftModel.from_pretrained(base, adapter_id); model.eval()
    sysp = ("[INST] <<SYS>> You are a helpfull assistant. List all the possible words divided with a "
            "coma. Your answer should not include anything except the words divided by a coma<</SYS>>\n")

    def masked_ppl(child, parent):
        prompt = f"{sysp}hyponym: {child} | hypernyms: [/INST]"
        ids = tok(prompt + f" {parent}", return_tensors="pt").input_ids.to(device)
        plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
        labels = ids.clone(); m = min(plen, labels.shape[1] - 1); labels[0, :m] = -100
        with torch.no_grad():
            return torch.exp(model(ids, labels=labels).loss).item()
    return masked_ppl


def compute_ppl_cache(pairs, device, cache_path):
    """{(hyponym, hypernym): ppl}, loading/saving a JSON cache so re-runs skip the model."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        for k, v in json.load(open(cache_path, encoding="utf-8")).items():
            hypo, hyper = k.split("\t")
            cache[(hypo, hyper)] = v
    todo = [pr for pr in pairs if pr not in cache]
    if todo:
        from tqdm import tqdm
        ppl = _load_taxollama(device)
        for (hypo, hyper) in tqdm(todo, desc="  -> TaxoLLaMA PPL"):
            cache[(hypo, hyper)] = ppl(child=hypo, parent=hyper)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            json.dump({f"{h}\t{y}": v for (h, y), v in cache.items()}, open(cache_path, "w", encoding="utf-8"))
    return cache


def attach_taxollama(D, cache):
    """Set per-edge taxo_ppl (forward PPL) and taxo_ratio (forward/reverse), and append the
    percentile-ranked ppl & ratio as two extra suspicion features (high = suspect: high PPL =
    implausible is-a; high ratio = the reverse direction is more plausible)."""
    for d in D:
        edges = d["edges"]
        fwd, rat = {}, {}
        for (p, c) in edges:
            pp, cc = _primary(p), _primary(c)
            f = cache.get((cc, pp)); r = cache.get((pp, cc))
            fwd[(p, c)] = f
            rat[(p, c)] = (f / r) if (f is not None and r not in (None, 0)) else None
        d["taxo_ppl"], d["taxo_ratio"] = fwd, rat
        ppl_pct = _pct_rank([fwd[e] if fwd[e] is not None else 0.0 for e in edges])
        rat_pct = _pct_rank([rat[e] if rat[e] is not None else 1.0 for e in edges])
        for i, e in enumerate(edges):
            d["feats"][i] = d["feats"][i] + [ppl_pct[i], rat_pct[i]]


def main():
    ap = argparse.ArgumentParser(description="Offline prune-sweep: heuristic vs LLM vs oracle.")
    ap.add_argument("--results_dir", default="results",
                    help="Dir with <ds>_Our_Method_edge_diagnostics.csv and GT_<ds>_eval.graphml.")
    ap.add_argument("--llm_results", default=None, help="Results JSON to overlay LLM prune runs (optional).")
    ap.add_argument("--ks", default="0,1,2,3,5,7,10,15,20,25,30,40,50,75,100",
                    help="Comma-separated K values to sweep.")
    ap.add_argument("--taxollama", action="store_true",
                    help="Add TaxoLLaMA per-edge is-a perplexity as FP signals (taxollama_ppl & "
                         "taxollama_ratio rankers + learned features). Needs a GPU + transformers/peft.")
    ap.add_argument("--ppl_cache", default=os.path.join("results", "taxollama_ppl_cache.json"),
                    help="JSON cache of TaxoLLaMA perplexities (reused across runs).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "prune_sweep.png"))
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",")})
    csvs = sorted(glob.glob(os.path.join(args.results_dir, "*_Our_Method_edge_diagnostics.csv")))
    if not csvs:
        print(f"[!] No *_Our_Method_edge_diagnostics.csv in {args.results_dir}. "
              f"Run our_method (raw) so it writes edge diagnostics.")
        return

    # Load every dataset first (need the pooled feature set for the learned ranker).
    D = []
    for c in csvs:
        loaded = load_dataset(c, args.results_dir)
        if loaded is None:
            continue
        ds, rows, G_pred, gt_pairs, gt_edges = loaded
        edges, feats = edge_features(rows)
        D.append(dict(ds=ds, rows=rows, G_pred=G_pred, gt_pairs=gt_pairs, gt_edges=gt_edges,
                      edges=edges, feats=feats, labels=[r["is_fp"] for r in rows]))
    if not D:
        print("[!] No datasets evaluated (missing GT graphmls?).")
        return

    # Optional TaxoLLaMA per-edge is-a plausibility -> extra features + rankers.
    feature_names = list(_FEATURES)
    if args.taxollama:
        print("[*] computing TaxoLLaMA per-edge perplexity (loads Llama-2-7b + adapter; cached)...")
        cache = compute_ppl_cache(taxollama_needed_pairs(D), args.device, args.ppl_cache)
        attach_taxollama(D, cache)
        feature_names += ["taxollama_ppl", "taxollama_ratio"]

    # Learned FP scorer: leave-one-dataset-out logistic regression on the features.
    nf = len(D[0]["feats"][0])
    for i, d in enumerate(D):
        Xtr = [f for j, o in enumerate(D) if j != i for f in o["feats"]]
        ytr = [l for j, o in enumerate(D) if j != i for l in o["labels"]]
        w = _fit_logreg(Xtr, ytr) if (0 < sum(ytr) < len(ytr)) else [0.0] * (nf + 1)
        d["learned"] = {e: _proba(w, f) for e, f in zip(d["edges"], d["feats"])}

    # AUC of each signal as an FP detector (pooled across datasets).
    yall = [l for d in D for l in d["labels"]]
    print("\n[*] FP-detection AUC (pooled; 0.5 = useless, higher = more separable):")
    for fi, name in enumerate(feature_names):
        print(f"      {name:16s} {auc([f[fi] for d in D for f in d['feats']], yall):.3f}")
    print(f"      {'combined(hand)':16s} "
          f"{auc([s for d in D for s in [score_edges(d['rows'], 'combined')[e] for e in d['edges']]], yall):.3f}")
    print(f"      {'learned(LOO)':16s} {auc([d['learned'][e] for d in D for e in d['edges']], yall):.3f}")
    w_all = _fit_logreg([f for d in D for f in d["feats"]], yall)
    ceil_auc = auc([_proba(w_all, f) for d in D for f in d["feats"]], yall)
    print(f"      {'learned(in-samp)':16s} {ceil_auc:.3f}   <- ceiling of these {len(feature_names)} features")

    # Sweep every dataset (heuristics + learned + any TaxoLLaMA rankers).
    agg = []
    for d in D:
        extra = {"learned": d["learned"]}
        if args.taxollama:
            extra["taxollama_ppl"] = {e: (d["taxo_ppl"][e] or 0.0) for e in d["edges"]}
            extra["taxollama_ratio"] = {e: (d["taxo_ratio"][e] or 0.0) for e in d["edges"]}
        agg.append(sweep_dataset(d["rows"], d["G_pred"], d["gt_pairs"], d["gt_edges"], ks,
                                 extra_scores=extra))
        b = agg[-1]["baseline"]
        n_fp = sum(d["labels"])
        print(f"    {d['ds']:26s} edges={len(d['rows']):4d} (FP={n_fp:4d})  "
              f"baseline P/R/F1={b[0]:.3f}/{b[1]:.3f}/{b[2]:.3f}")

    llm = load_llm_prune(args.llm_results)
    if llm:
        print("[*] overlaying LLM prune point(s): "
              + ", ".join(f"top{k}/{rb}={v[2]:.3f}F1" for (k, rb), v in sorted(llm.items())))
    n = len(agg)
    best = lambda rb: max(sum(a["by"][rb][K][2] for a in agg) / n for K in ks)
    base_f1 = sum(a["baseline"][2] for a in agg) / n
    print(f"[*] best mean F1 over sweep  baseline={base_f1:.3f}  "
          f"combined={best('combined'):.3f}  learned={best('learned'):.3f}  "
          f"oracle={max(sum(a['oracle'][K][2] for a in agg)/n for K in ks):.3f}")
    print("    verdict: if learned AUC approaches the in-sample ceiling AND its F1 curve nears the "
          "oracle -> learn the weights. If the in-sample ceiling itself is low (~<0.75) -> these 4 "
          "features can't separate FP/TP; you need new per-edge signals.")
    plot(agg, llm, ks, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
