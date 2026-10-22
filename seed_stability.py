"""Seed stability: is the structured method less sensitive than the single call to
WHICH terms a benchmark happens to sample?

Each seed draws an independent 100-term subset of every dataset's frozen FULL graph
(scale_benchmarks.py -> {ds}_SUB100S{seed}); both methods run on every subset; the
spread of their scores across seeds is the stability measure. The legacy SUB
(seed 42) sample is deliberately not one of the seeds.

    python seed_stability.py prepare                      # build the seeded subsets (no LLM)
    python seed_stability.py run --tag gemma4 --provider vllm --model google/gemma-4-31b-it
    python seed_stability.py analyze --tag gemma4         # table + figure

`run` resumes: a seed whose results file already holds both methods for a dataset
skips that dataset. Each seed writes its own results file
(results/seedstab/{tag}_s{seed}.json), so runs never collide and `--jobs N` can run
N seeds concurrently (per-seed logs then go to results/seedstab/logs/).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CORE = ["WordNetFood", "CellOntology", "SemEvalFood",
        "SchemaOrgV2", "POV2", "OBIV2", "MatOntoV2"]
SEEDS = list(range(1, 11))
N = 100
OUT_DIR = os.path.join("results", "seedstab")
METHODS = {"structured": "Our Method", "single-call": "Single-Call Baseline"}
DISPLAY = {"SchemaOrgV2": "SchemaOrg", "POV2": "PO", "OBIV2": "OBI", "MatOntoV2": "MatOnto",
           "WordNetFood": "WordNet", "CellOntology": "CL", "SemEvalFood": "SemEval"}


def token(seed):
    return f"SUB{N}S{seed}"


def results_path(tag, seed):
    return os.path.join(OUT_DIR, f"{tag}_s{seed}.json")


def load_scores(tag, seed):
    """-> {dataset: {method_key: result_row}} for one seed's results file."""
    path = results_path(tag, seed)
    if not os.path.exists(path):
        return {}
    out = {}
    for block in json.load(open(path, encoding="utf-8")):
        ds = block["dataset"].rsplit("_", 1)[0]
        for r in block["results"]:
            for key, label in METHODS.items():
                if r["method"].startswith(label) and "[matched]" not in r["method"]:
                    out.setdefault(ds, {})[key] = r
    return out


# ---------------- prepare ----------------

def cmd_prepare(args):
    cmd = [sys.executable, "scale_benchmarks.py", "--target_nodes", str(N),
           "--seeds", *map(str, args.seeds), "--datasets", *args.datasets]
    print("[*]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------- run ----------------

def run_seed(seed, args, log):
    done = load_scores(args.tag, seed)
    todo = [ds for ds in args.datasets if set(done.get(ds, {})) != set(METHODS)]
    if not todo:
        return seed, "already complete"
    cmd = [sys.executable, "main.py", "--method", "our_method", "single_call",
           "--datasets", *todo, "--scale", token(seed).lower(),
           "--provider", args.provider, "--model", args.model,
           "--results_file", results_path(args.tag, seed),
           "--diag_tag", f"{args.tag}_seedstab"]
    if args.dry_run:
        return seed, " ".join(cmd)
    if log:
        os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
        with open(os.path.join(OUT_DIR, "logs", f"{args.tag}_s{seed}.log"), "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd).returncode
    missing = [ds for ds in todo if set(load_scores(args.tag, seed).get(ds, {})) != set(METHODS)]
    status = f"exit {rc}" + (f", still missing {missing}" if missing else ", complete")
    return seed, status


def cmd_run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[*] {args.tag}: seeds {args.seeds} x {len(args.datasets)} datasets, jobs={args.jobs}")
    if args.jobs <= 1:
        for seed in args.seeds:
            print(f"\n=== seed {seed} ===", flush=True)
            print(f"[seed {seed}] {run_seed(seed, args, log=False)[1]}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for seed, status in pool.map(lambda s: run_seed(s, args, log=True), args.seeds):
                print(f"[seed {seed}] {status}", flush=True)


# ---------------- analyze ----------------

def cmd_analyze(args):
    import csv
    import numpy as np
    from scipy import stats

    data = {}  # (ds, method) -> {seed: row}
    for seed in args.seeds:
        for ds, by_m in load_scores(args.tag, seed).items():
            if ds in args.datasets:
                for m, r in by_m.items():
                    data.setdefault((ds, m), {})[seed] = r
    if not data:
        raise SystemExit(f"[!] no results under {OUT_DIR} for tag '{args.tag}'")

    metric = "Cond_Clos_F1"
    rows, sd = [], {}
    print(f"\n{'dataset':14s} {'method':12s} {'n':>3s} {'mean F1':>8s} {'SD':>7s} {'CV':>6s} "
          f"{'SD prec':>8s} {'SD rec':>7s}")
    for ds in args.datasets:
        for m in METHODS:
            vals = data.get((ds, m), {})
            if len(vals) < 2:
                continue
            f1 = np.array([v[metric] for v in vals.values()])
            p = np.array([v["Cond_Clos_Precision"] for v in vals.values()])
            r = np.array([v["Cond_Clos_Recall"] for v in vals.values()])
            s = f1.std(ddof=1)
            sd[(ds, m)] = s
            rows.append({"dataset": ds, "method": m, "n_seeds": len(f1), "mean_f1": f1.mean(),
                         "sd_f1": s, "cv_f1": s / f1.mean() if f1.mean() else float("nan"),
                         "sd_precision": p.std(ddof=1), "sd_recall": r.std(ddof=1)})
            print(f"{ds:14s} {m:12s} {len(f1):3d} {f1.mean():8.3f} {s:7.3f} {rows[-1]['cv_f1']:6.3f} "
                  f"{rows[-1]['sd_precision']:8.3f} {rows[-1]['sd_recall']:7.3f}")

    # per dataset: Brown-Forsythe test of equal variance across seeds
    print(f"\nper-dataset equal-variance test (Brown-Forsythe, F1 across seeds):")
    tests = []
    for ds in args.datasets:
        a, b = data.get((ds, "structured"), {}), data.get((ds, "single-call"), {})
        if len(a) >= 3 and len(b) >= 3:
            w, pv = stats.levene([v[metric] for v in a.values()],
                                 [v[metric] for v in b.values()], center="median")
            ratio = sd[(ds, "structured")] / sd[(ds, "single-call")] if sd[(ds, "single-call")] else float("nan")
            tests.append((ds, ratio, pv))
            print(f"  {ds:14s} SD ratio structured/single-call = {ratio:5.2f}  p = {pv:.3f}")

    # pooled: is structured's SD smaller across datasets?
    paired = [(sd[(ds, "structured")], sd[(ds, "single-call")]) for ds in args.datasets
              if (ds, "structured") in sd and (ds, "single-call") in sd]
    pooled = ""
    if len(paired) >= 5:
        s_st, s_sc = np.array(paired).T
        wst = stats.wilcoxon(s_st, s_sc)
        gm = float(np.exp(np.mean(np.log(s_st / s_sc))))
        lower = int((s_st < s_sc).sum())
        pooled = (f"structured SD lower on {lower}/{len(paired)} datasets; geometric-mean SD ratio "
                  f"{gm:.2f}; Wilcoxon signed-rank p = {wst.pvalue:.3f}")
        print(f"\npooled: {pooled}")

    # the headline number: core-mean F1 per seed
    core = {m: [] for m in METHODS}
    for seed in args.seeds:
        if all(seed in data.get((ds, m), {}) for ds in args.datasets for m in METHODS):
            for m in METHODS:
                core[m].append(np.mean([data[(ds, m)][seed][metric] for ds in args.datasets]))
    if len(core["structured"]) >= 2:
        st, sc = np.array(core["structured"]), np.array(core["single-call"])
        wins = int((st > sc).sum())
        print(f"\ncore-mean F1 over {len(st)} complete seeds:")
        print(f"  structured  {st.mean():.3f} +/- {st.std(ddof=1):.3f}  (range {st.min():.3f}-{st.max():.3f})")
        print(f"  single-call {sc.mean():.3f} +/- {sc.std(ddof=1):.3f}  (range {sc.min():.3f}-{sc.max():.3f})")
        print(f"  structured ahead on {wins}/{len(st)} seeds")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"summary_{args.tag}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[*] wrote {out_csv}")
    plot(args, data, sd, pooled, core)


def plot(args, data, sd, pooled, core):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    style = {"structured": ("C0", "o", "structured (ours)"),
             "single-call": ("C1", "s", "single-call")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    xs = np.arange(len(args.datasets))
    for off, m in ((-0.17, "structured"), (0.17, "single-call")):
        col, mk, label = style[m]
        for i, ds in enumerate(args.datasets):
            vals = [v["Cond_Clos_F1"] for v in data.get((ds, m), {}).values()]
            if vals:
                ax1.scatter(np.full(len(vals), i + off), vals, color=col, marker=mk, s=22,
                            alpha=0.55, label=label if i == 0 else None)
                ax1.plot([i + off - 0.1, i + off + 0.1], [np.mean(vals)] * 2, color=col, lw=2.4)
        ax2.bar(xs + off, [sd.get((ds, m), np.nan) for ds in args.datasets], width=0.34,
                color=col, label=label)
    names = [DISPLAY.get(ds, ds) for ds in args.datasets]
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Cond. Closure F1 (one point per seed)", fontweight="bold")
    ax1.grid(axis="y", color="0.92")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("SD of F1 across seeds", fontweight="bold")
    ax2.grid(axis="y", color="0.92")
    n = max((len(v) for v in data.values()), default=0)
    head = f"Score stability across {n} independently sampled {N}-term subsets per dataset ({args.tag})"
    sub = pooled
    if len(core["structured"]) >= 2:
        sub += (f"\ncore-mean F1: structured {np.mean(core['structured']):.3f} +/- "
                f"{np.std(core['structured'], ddof=1):.3f}, single-call {np.mean(core['single-call']):.3f} "
                f"+/- {np.std(core['single-call'], ddof=1):.3f}")
    fig.suptitle(head + ("\n" + sub if sub else ""), fontsize=10.5, fontweight="bold", y=1.06)
    fig.tight_layout()
    os.makedirs("vis", exist_ok=True)
    out = os.path.join("vis", f"seed_stability_{args.tag}.png")
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"[*] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
        p.add_argument("--datasets", nargs="+", default=CORE)
        if name in ("run", "analyze"):
            p.add_argument("--tag", required=True,
                           help="Run label carried into every file name, e.g. gemma4 / gemini31")
        if name == "run":
            p.add_argument("--provider", default="vllm")
            p.add_argument("--model", required=True)
            p.add_argument("--jobs", type=int, default=1,
                           help="Seeds to run concurrently (each its own process and results file)")
            p.add_argument("--dry_run", action="store_true", help="Print the main.py calls only")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
