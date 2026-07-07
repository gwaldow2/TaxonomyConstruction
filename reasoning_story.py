"""Tell the reasoning-effort story from the --debug_parse artifacts.

Reads the per-response parse-debug JSONL files (written by main.py --debug_parse) and,
optionally, the benchmark results JSON, then draws one figure linking:
  reasoning tokens (chars)  ->  edges the model emits  ->  taxonomy accuracy (F1).

The point it visualises: under a prompt that suppresses deliberation, responses cluster
at low reasoning and zero edges, which drags recall/F1 down. Run several prompt variants
with --debug_parse (e.g. full, full_json, legacy_json) to compare them on one axis.

    python main.py --method our_method --prompt_variant full full_json legacy_json \
        --debug_parse --results_file prompt_ablation.json
    python reasoning_story.py --results prompt_ablation.json

Outputs vis/reasoning_story.png.
"""

import os
import re
import json
import glob
import argparse
from collections import defaultdict

VIS_DIR = "vis"

# results/<Dataset>_Our_Method_<variant>_parse_debug.jsonl
_DEBUG_RE = re.compile(r"(.+?)_Our_Method_(.+?)_parse_debug\.jsonl$")


def load_debug_records(debug_glob):
    """Return a flat list of per-response dicts: dataset, variant, reasoning_chars,
    edges (n_strict), produced (edges > 0)."""
    rows = []
    for path in sorted(glob.glob(debug_glob)):
        m = _DEBUG_RE.search(os.path.basename(path))
        if not m:
            continue
        dataset, variant = m.group(1), m.group(2)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # Edges this response contributed. JSON variants populate n_strict, the
                # '<=' variants populate n_line, and the other parser returns 0 -- so max()
                # gives the real count regardless of the variant's output format.
                diag = r.get("diag", {})
                edges = max(int(diag.get("n_strict", 0) or 0), int(diag.get("n_line", 0) or 0))
                rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    "reasoning_chars": int(r.get("reasoning_chars", 0) or 0),
                    "edges": edges,
                    "produced": edges > 0,
                })
    return rows


def load_f1(results_path):
    """Map (dataset, variant) -> Cond. Closure F1 from the benchmark JSON (clawback=0 runs)."""
    out = {}
    if not results_path or not os.path.exists(results_path):
        return out
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    for block in data:
        ds = str(block.get("dataset", "")).replace(".csv", "")
        for res in block.get("results", []):
            meth = res.get("method", "")
            mm = re.search(r"Our Method \[([^\]]+)\]", meth)
            if mm and "clawback=0" in meth:
                out[(ds, mm.group(1))] = res.get("Cond_Clos_F1")
    return out


def aggregate(rows, f1map):
    """Per (dataset, variant): mean reasoning, % of responses that produced an edge,
    total edges, and F1 (if available)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["variant"])].append(r)
    agg = []
    for (ds, v), rs in sorted(groups.items()):
        rc = [x["reasoning_chars"] for x in rs]
        agg.append({
            "dataset": ds,
            "variant": v,
            "n": len(rs),
            "mean_reasoning": sum(rc) / len(rc) if rc else 0.0,
            "pct_produced": 100.0 * sum(1 for x in rs if x["produced"]) / len(rs) if rs else 0.0,
            "total_edges": sum(x["edges"] for x in rs),
            "f1": f1map.get((ds, v)),
        })
    return agg


def plot_story(rows, agg, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = sorted({r["variant"] for r in rows})
    cmap = plt.get_cmap("tab10")
    color = {v: cmap(i % 10) for i, v in enumerate(variants)}
    rng = np.random.default_rng(0)   # deterministic jitter

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # Panel 1: per-response reasoning vs edges (jittered), one colour per variant.
    ax = axes[0]
    for v in variants:
        xs = np.array([r["reasoning_chars"] for r in rows if r["variant"] == v], dtype=float)
        ys = np.array([r["edges"] for r in rows if r["variant"] == v], dtype=float)
        ys = ys + rng.uniform(-0.25, 0.25, size=len(ys))   # jitter integer edge counts
        ax.scatter(xs, ys, s=16, alpha=0.45, color=color[v], label=v, edgecolors="none")
    ax.set_title("Per response: reasoning -> edges emitted", fontsize=12, fontweight="bold")
    ax.set_xlabel("reasoning chars", fontweight="bold")
    ax.set_ylabel("edges emitted (n_strict)", fontweight="bold")
    ax.legend(title="variant", fontsize=8)

    # Panel 2: reasoning distribution by variant, split empty vs produced-an-edge.
    ax = axes[1]
    labels, data, box_colors = [], [], []
    for v in variants:
        empt = [r["reasoning_chars"] for r in rows if r["variant"] == v and not r["produced"]]
        prod = [r["reasoning_chars"] for r in rows if r["variant"] == v and r["produced"]]
        if empt:
            labels.append(f"{v}\n(0 edges)"); data.append(empt); box_colors.append(color[v])
        if prod:
            labels.append(f"{v}\n(>=1 edge)"); data.append(prod); box_colors.append(color[v])
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.45)
    ax.set_title("Reasoning per response: empty vs productive", fontsize=12, fontweight="bold")
    ax.set_ylabel("reasoning chars", fontweight="bold")
    ax.tick_params(axis="x", labelsize=7, labelrotation=0)

    # Panel 3: per-dataset mean reasoning vs F1 (the accuracy link).
    ax = axes[2]
    have_f1 = any(a["f1"] is not None for a in agg)
    if have_f1:
        for v in variants:
            xs = [a["mean_reasoning"] for a in agg if a["variant"] == v and a["f1"] is not None]
            ys = [a["f1"] for a in agg if a["variant"] == v and a["f1"] is not None]
            ax.scatter(xs, ys, s=70, alpha=0.8, color=color[v], label=v, edgecolors="black", linewidths=0.5)
        ax.set_ylabel("Cond. Closure F1", fontweight="bold")
        ax.set_title("Per dataset: mean reasoning -> F1", fontsize=12, fontweight="bold")
    else:
        for v in variants:
            xs = [a["mean_reasoning"] for a in agg if a["variant"] == v]
            ys = [a["pct_produced"] for a in agg if a["variant"] == v]
            ax.scatter(xs, ys, s=70, alpha=0.8, color=color[v], label=v, edgecolors="black", linewidths=0.5)
        ax.set_ylabel("% responses producing >=1 edge", fontweight="bold")
        ax.set_title("Per dataset: mean reasoning -> productivity\n(no F1 in --results)",
                     fontsize=11, fontweight="bold")
    ax.set_xlabel("mean reasoning chars (per dataset)", fontweight="bold")
    ax.legend(title="variant", fontsize=8)

    fig.suptitle("Reasoning effort drives edges emitted, and edges drive accuracy",
                 y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Reasoning-effort story from --debug_parse artifacts.")
    ap.add_argument("--debug_glob", default="results/*_parse_debug.jsonl",
                    help="Glob for the parse-debug JSONL files.")
    ap.add_argument("--results", default="benchmark_results.json",
                    help="Benchmark JSON, for the per-dataset F1 in panel 3 (optional).")
    ap.add_argument("--out", default=os.path.join(VIS_DIR, "reasoning_story.png"))
    args = ap.parse_args()

    rows = load_debug_records(args.debug_glob)
    if not rows:
        print(f"[!] No parse-debug records matched {args.debug_glob!r}. "
              f"Run main.py with --debug_parse first.")
        return
    f1map = load_f1(args.results)
    agg = aggregate(rows, f1map)

    variants = sorted({r["variant"] for r in rows})
    print(f"[*] {len(rows)} responses across {len({r['dataset'] for r in rows})} datasets, "
          f"variants: {', '.join(variants)}")
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        empt = [r["reasoning_chars"] for r in rs if not r["produced"]]
        prod = [r["reasoning_chars"] for r in rs if r["produced"]]
        med = lambda xs: int(sorted(xs)[len(xs) // 2]) if xs else 0
        print(f"    {v:12s}: {len(rs):4d} resp | {len(prod)} productive "
              f"(median reasoning {med(prod)}) vs {len(empt)} empty (median {med(empt)})")
    plot_story(rows, agg, args.out)
    print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
