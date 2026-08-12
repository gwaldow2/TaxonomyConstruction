"""QLoRA fine-tuning for taxonomy construction on the SFT files from lora_data_prep.py.

Loss is masked to the completion only -- the prompt (instructions + candidate list) is context,
not something to memorize. 4-bit NF4 base + bf16 compute fits a 7-31B model on one H100.

--base selects which model to start from, so the same SFT data can be tuned from either and
the results compared (see lora_comparison.py):

  --base qwen     Qwen/Qwen2.5-7B-Instruct  -- small, ungated, fast to iterate on
  --base gemma4   google/gemma-4-31b-it     -- the model the method itself uses

Gemma 4 needs peft>=0.19.0 and NO explicit target_modules. Its vision/audio encoders wrap
projections in Gemma4ClippableLinear, which subclasses nn.Module rather than nn.Linear, so
PEFT's type allowlist rejects it. A leaf-name list like ["q_proj", ...] matches those encoder
projections too (SigLIP reuses the names) and fails on the wrapper -- the language model's own
layers were never the problem. peft>=0.19 ships Gemma 4 defaults that scope adapters to the
language model by regex, so leaving target_modules unset is the fix.

Whatever base you pick, evaluate the UNTUNED SAME base as the control -- a control from a
different model measures the model, not the tuning.

  in-domain     --train WordNetFood                        (eval on WordNetFood)
  cross-domain  --train all --exclude WordNetFood           (eval on WordNetFood)

Use --max_examples to make those two comparable. Unmatched, cross-domain trains on every other
ontology and in-domain on one, so a difference between them confounds domain match with data
volume. Setting --max_examples to the in-domain count gives both arms the same number of
examples and the same number of optimizer steps:

    for D in WordNetFood CellOntology LLMs4OL_OBI; do
        N=$(wc -l < sft/$D.jsonl)
        python lora_train.py --base gemma4 --train $D --out_dir adapters/in_$D
        python lora_train.py --base gemma4 --train all --exclude $D --max_examples $N \
            --out_dir adapters/cross_$D
    done

Do not regenerate sft/ partway through a sweep: the cross-domain pool would change between
rows and "cross-domain" would no longer mean the same thing for every dataset.

    python lora_train.py --base qwen   --train WordNetFood --out_dir adapters/qwen_in_WordNetFood
    python lora_train.py --base gemma4 --train WordNetFood --out_dir adapters/gemma4_in_WordNetFood

Serve the adapter with:  vllm serve <base> --enable-lora --lora-modules taxo=<out_dir>
then evaluate with:      python main.py --model taxo ...
"""

import os
import json
import glob
import random
import argparse
from collections import Counter


# --base presets. target_modules=None means "let PEFT choose": required for gemma4 (its
# defaults skip the Gemma4ClippableLinear vision/audio wrappers), harmless elsewhere.
BASE_PRESETS = {
    "qwen": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "min_peft": None,
    },
    "gemma4": {
        "model": "google/gemma-4-31b-it",
        "target_modules": None,          # MUST stay None -- see the module docstring
        "min_peft": (0, 19, 0),
    },
}


def _version_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (3 - len(out)))


def resolve_base(args):
    """-> (model_id, target_modules_or_None). CLI overrides win over the preset."""
    preset = BASE_PRESETS[args.base]
    model_id = args.base_model or preset["model"]
    if args.target_modules:
        targets = None if args.target_modules == ["auto"] else args.target_modules
    else:
        targets = preset["target_modules"]
    if preset["min_peft"]:
        import peft
        need, have = preset["min_peft"], _version_tuple(peft.__version__)
        if have < need:
            raise SystemExit(
                f"[!] --base {args.base} needs peft>={'.'.join(map(str, need))} (have {peft.__version__}). "
                f"Older PEFT cannot attach LoRA to Gemma 4: its encoder projections are "
                f"Gemma4ClippableLinear wrappers. Run: pip install -U 'peft>=0.19.0'")
    return model_id, targets


def load_records(sft_dir, domains, exclude):
    """Read sft/<domain>.jsonl for the requested domains. 'all' means every file present."""
    if domains == ["all"]:
        paths = sorted(glob.glob(os.path.join(sft_dir, "*.jsonl")))
    else:
        paths = [os.path.join(sft_dir, f"{d}.jsonl") for d in domains]
    recs = []
    for p in paths:
        d = os.path.splitext(os.path.basename(p))[0]
        if d in exclude:
            print(f"    [-] excluding {d}")
            continue
        if not os.path.exists(p):
            print(f"    [!] missing {p} -- skipping")
            continue
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        for r in rows:
            r["domain"] = d
        recs.extend(rows)
        print(f"    [+] {d:26s} {len(rows):5d} examples")
    return recs


def subsample(records, n, seed):
    """Cut the POOLED record list to n examples, shuffled with `seed`.

    This is what makes in-domain vs cross-domain a controlled comparison. Unmatched, the
    cross-domain arm trains on every other ontology -- 6k-20k examples against in-domain's
    156-3005 -- so a difference between them confounds domain match with data volume, and the
    volume advantage is not even constant across datasets (it ranged 2.2x to 60x in practice).
    Matching the counts makes both arms take the same number of optimizer steps, leaving the
    source of the data as the only difference.

    Sampling is from the POOL, not per source, so the mixture reflects the natural sizes of the
    other ontologies. Equalising per source is a different design (it also controls source
    diversity) and would belong in a separate arm.
    """
    if not n or n >= len(records):
        if n:
            print(f"    [i] --max_examples {n} >= {len(records)} available -- using all of them")
        return records
    rng = random.Random(seed)
    out = records[:]
    rng.shuffle(out)
    out = out[:n]
    kept = Counter(r["domain"] for r in out)
    print(f"    [*] subsampled to {n} of {len(records)} examples (seed={seed})")
    for d, k in sorted(kept.items(), key=lambda kv: -kv[1]):
        print(f"        {d:26s} {k:5d}")
    return out


class SFTDataset:
    """Tokenizes prompt+completion and masks the prompt tokens out of the loss."""

    def __init__(self, records, tokenizer, max_len):
        self.records, self.tok, self.max_len = records, tokenizer, max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        prompt_ids = self.tok(r["prompt"], add_special_tokens=True).input_ids
        eos = self.tok.eos_token or ""
        comp_ids = self.tok(r["completion"] + eos, add_special_tokens=False).input_ids
        # Truncate the PROMPT from the left if needed so the completion always survives intact;
        # a truncated completion would train the model to stop mid-answer.
        room = self.max_len - len(comp_ids)
        if room < 1:
            comp_ids = comp_ids[:self.max_len - 1]
            room = 1
        prompt_ids = prompt_ids[-room:]
        return {"input_ids": prompt_ids + comp_ids,
                "labels": [-100] * len(prompt_ids) + comp_ids}


def make_collator(pad_id):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}
    return collate


def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tune for taxonomy construction.")
    ap.add_argument("--sft_dir", default="sft")
    ap.add_argument("--train", nargs="+", required=True, help="Domains to train on, or 'all'.")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="Domains to drop (use for leave-one-domain-out with --train all).")
    ap.add_argument("--max_examples", type=int, default=0,
                    help="Cap the POOLED training set at N examples (0 = use all). Set it to the "
                         "in-domain arm's example count so cross-domain trains on the same amount "
                         "of data, making domain match the only difference between the two. "
                         "Sampling is seeded by --seed.")
    ap.add_argument("--base", choices=sorted(BASE_PRESETS), default="qwen",
                    help="Which base model to start from. Sets the model id and the LoRA "
                         "target modules appropriate for that architecture.")
    ap.add_argument("--base_model", default=None,
                    help="Override the preset's model id (keeps the preset's target modules).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=4096,
                    help="Qwen2.5 handles 32k; 4096 clears the long high-fan-out completions "
                         "(e.g. CellOntology) that 2048 truncated.")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="+", default=None,
                    help="Override the preset's LoRA target modules. Pass 'auto' to let PEFT "
                         "pick its architecture defaults (this is what gemma4 uses).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true", help="Load + tokenize + report, then exit.")
    args = ap.parse_args()

    print(f"[*] loading SFT data from {args.sft_dir}/")
    records = load_records(args.sft_dir, args.train, set(args.exclude))
    if not records:
        raise SystemExit("[!] no training records -- run lora_data_prep.py first.")
    n_pool = len(records)
    records = subsample(records, args.max_examples, args.seed)
    doms = sorted({r["domain"] for r in records})
    print(f"[*] {len(records)} examples across {len(doms)} domain(s): {', '.join(doms)}")

    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base_model, target_modules = resolve_base(args)
    print(f"[*] base '{args.base}' -> {base_model} | target_modules="
          f"{target_modules if target_modules else 'PEFT defaults (auto)'}")

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(records, tok, args.max_seq_len)

    # UNtruncated length distribution, so clipping is visible instead of silent. A truncated
    # completion teaches the model to stop mid-answer, so any example over the cap is a warning.
    eos = tok.eos_token or ""
    sample = records if args.dry_run else records[:min(len(records), 1000)]
    raw, comp_lens = [], []
    for r in sample:
        p = len(tok(r["prompt"], add_special_tokens=True).input_ids)
        c = len(tok(r["completion"] + eos, add_special_tokens=False).input_ids)
        raw.append(p + c); comp_lens.append(c)
    over = sum(1 for L in raw if L > args.max_seq_len)
    print(f"[*] untruncated length over {len(sample)} examples: mean={sum(raw)//len(raw)} "
          f"max={max(raw)} | completion max={max(comp_lens)} | cap={args.max_seq_len}")
    if over:
        print(f"    [warn] {over}/{len(sample)} ({100*over/len(sample):.1f}%) exceed the cap and WILL be "
              f"truncated -- raise --max_seq_len (Qwen2.5 handles up to 32k) to keep completions intact.")
    else:
        print(f"    [ok] no truncation: every sampled example fits under {args.max_seq_len}.")

    if args.dry_run:
        ex = ds[0]
        n_sup = sum(1 for x in ex["labels"] if x != -100)
        print(f"[*] dry run -- example 0: {len(ex['input_ids'])} tokens, {n_sup} supervised (completion only)")
        print("---- prompt (first 400 chars) ----")
        print(records[0]["prompt"][:400])
        print("---- completion ----")
        print(records[0]["completion"][:400])
        return

    print(f"[*] loading {base_model} in 4-bit NF4")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     bias="none", task_type="CAUSAL_LM",
                     **({"target_modules": target_modules} if target_modules else {}))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    # Report WHERE the adapters landed. On a multimodal base this is the check that catches
    # adapters attaching to the vision/audio tower instead of (or as well as) the LM.
    adapted = sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_modules() if ".lora_A" in n})
    if adapted:
        leaves = sorted({a.rsplit(".", 1)[-1] for a in adapted})
        print(f"[*] LoRA attached to {len(adapted)} modules; leaf types: {', '.join(leaves)}")
        print(f"    e.g. {adapted[0]}")
        stray = [a for a in adapted if any(k in a for k in ("vision", "audio", "multi_modal"))]
        if stray:
            print(f"    [warn] {len(stray)} adapter(s) are on vision/audio modules, e.g. {stray[0]} "
                  f"-- for a text-only task these add trainable params that never see gradient signal.")

    trainer = Trainer(
        model=model, train_dataset=ds, data_collator=make_collator(tok.pad_token_id),
        args=TrainingArguments(
            output_dir=args.out_dir, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
            logging_steps=10, save_strategy="epoch", save_total_limit=2,
            report_to=[], seed=args.seed, remove_unused_columns=False))
    trainer.train()

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "resolved_base_model": base_model,
                   "resolved_target_modules": target_modules, "train_domains": doms,
                   "n_examples": len(records), "n_pool_before_subsample": n_pool,
                   "subsampled": len(records) < n_pool}, f, indent=2)
    print(f"[*] adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()
