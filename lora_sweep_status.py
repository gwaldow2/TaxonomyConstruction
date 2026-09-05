"""Which LoRA sweep runs finished, which crashed, and the exact commands to finish the rest.

A run is COMPLETE iff its out_dir contains train_config.json: lora_train.py writes that file
LAST, after the adapter and tokenizer are saved, so its presence proves a clean finish. A dir
with only checkpoint-* subdirs is a PARTIAL run (killed mid-training); the emitted commands
delete it and retrain from scratch -- resuming a crashed run across a cosine schedule and an
early-stopping state is not worth the ambiguity on runs this short. A dir that does not exist
was never started.

Also cross-checks the matched-N design on whatever already finished: the in/cross pair must
report the same n_examples, and the cross arm's n_examples must equal the CURRENT line count
of sft/<domain>.jsonl -- a mismatch there means sft/ was regenerated after that arm trained,
and the arm belongs to a different sweep.

    python lora_sweep_status.py                              # report on the gemma4V2 sweep
    python lora_sweep_status.py --emit_script finish_lora.sh # also write the finish script

No GPU, no LLM calls, no model downloads -- pure filesystem inspection.
"""

import os
import json
import glob
import argparse

CORE_DATASETS = ["WordNetFood", "CellOntology", "SemEvalFood", "LLMs4OL_OBI",
                 "LLMs4OL_PO", "LLMs4OL_SchemaOrg", "LLMs4OL_MatOnto"]


def classify(out_dir):
    """-> (status, info): 'complete' (with the parsed train_config), 'partial' (with the
    latest checkpoint step), or 'missing'."""
    if not os.path.isdir(out_dir):
        return "missing", {}
    cfg = os.path.join(out_dir, "train_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                return "complete", json.load(f)
        except (OSError, json.JSONDecodeError):
            return "partial", {"note": "train_config.json unreadable"}
    steps = []
    for p in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        tail = p.rsplit("-", 1)[-1]
        if tail.isdigit():
            steps.append(int(tail))
    return "partial", {"last_checkpoint_step": max(steps) if steps else 0,
                       "n_checkpoints": len(steps)}


def sft_count(sft_dir, domain):
    """Line count of sft/<domain>.jsonl, or None if the file is absent."""
    p = os.path.join(sft_dir, f"{domain}.jsonl")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def train_command(base, arm, domain, n_examples, out_dir):
    """The exact lora_train.py invocation for one arm, N baked in as a literal."""
    if arm == "in":
        return f"python lora_train.py --base {base} --train {domain} --out_dir {out_dir}"
    return (f"python lora_train.py --base {base} --train all --exclude {domain} "
            f"--max_examples {n_examples} --out_dir {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Report LoRA sweep completion; emit finish commands.")
    ap.add_argument("--adapters_dir", default="adapters")
    ap.add_argument("--prefix", default="gemma4V2",
                    help="Adapter dir prefix: <prefix>_in_<D> and <prefix>_cross_<D>.")
    ap.add_argument("--base", default="gemma4")
    ap.add_argument("--sft_dir", default="sft")
    ap.add_argument("--datasets", nargs="+", default=CORE_DATASETS)
    ap.add_argument("--emit_script", default=None,
                    help="Write the remaining runs to this bash script.")
    args = ap.parse_args()

    todo, n_done, n_partial, n_missing = [], 0, 0, 0
    completed_n = {}

    print(f"[*] sweep {args.prefix} in {args.adapters_dir}/ "
          f"({len(args.datasets)} datasets x 2 arms)\n")
    for d in args.datasets:
        n = sft_count(args.sft_dir, d)
        for arm in ("in", "cross"):
            out_dir = os.path.join(args.adapters_dir, f"{args.prefix}_{arm}_{d}")
            status, info = classify(out_dir)
            if status == "complete":
                n_done += 1
                completed_n[(d, arm)] = info.get("n_examples")
                extra = (f"n={info.get('n_examples')} "
                         f"stopped_epoch={round(info['stopped_epoch'], 2) if info.get('stopped_epoch') else '?'} "
                         f"best_eval_loss={round(info['best_eval_loss'], 4) if info.get('best_eval_loss') else 'n/a'}")
                print(f"  [done]    {d:22s} {arm:5s} {extra}")
                continue
            if status == "partial":
                n_partial += 1
                print(f"  [PARTIAL] {d:22s} {arm:5s} crashed at step "
                      f"{info.get('last_checkpoint_step', '?')} -- will delete and retrain")
            else:
                n_missing += 1
                print(f"  [missing] {d:22s} {arm:5s} not started")
            if n is None:
                print(f"      [!] {args.sft_dir}/{d}.jsonl missing -- run lora_data_prep.py "
                      f"before this arm can train")
                continue
            if status == "partial":
                todo.append(f"rm -rf {out_dir}")
            todo.append(train_command(args.base, arm, d, n, out_dir))

    print(f"\n[*] {n_done} complete | {n_partial} partial | {n_missing} not started")

    # Matched-N integrity on whatever already finished.
    for d in args.datasets:
        n_in, n_cross = completed_n.get((d, "in")), completed_n.get((d, "cross"))
        if n_in is not None and n_cross is not None and n_in != n_cross:
            print(f"    [warn] {d}: in-domain trained on {n_in} examples but cross-domain on "
                  f"{n_cross} -- the pair is NOT volume-matched")
        n_now = sft_count(args.sft_dir, d)
        for arm in ("in", "cross"):
            n_arm = completed_n.get((d, arm))
            if n_arm is not None and n_now is not None and n_arm != n_now:
                print(f"    [warn] {d}/{arm}: trained on {n_arm} examples but "
                      f"{args.sft_dir}/{d}.jsonl now has {n_now} -- sft/ was regenerated "
                      f"after this arm trained; retrain it to keep the sweep consistent")

    if not todo:
        print("[*] nothing left to run -- the sweep is complete.")
        return
    print(f"\n[*] {sum(1 for t in todo if not t.startswith('rm '))} run(s) remaining:")
    for t in todo:
        print(f"    {t}")
    if args.emit_script:
        with open(args.emit_script, "w", encoding="utf-8", newline="\n") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("# Finish the interrupted LoRA sweep. Run in the taxolora conda env,\n")
            f.write("# inside tmux; no vLLM server is needed for training.\n")
            f.write("set -e\n\n")
            f.write("\n".join(todo) + "\n")
        print(f"\n[*] wrote {args.emit_script} -- review it, then:  bash {args.emit_script}")


if __name__ == "__main__":
    main()
