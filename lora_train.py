"""QLoRA fine-tuning for taxonomy construction on the SFT files from lora_data_prep.py.

Loss is masked to the completion only -- the prompt (instructions + candidate list) is context,
not something to memorize. 4-bit NF4 base + bf16 compute fits a 27-31B model on one H100.

  in-domain     --train WordNetFood                        (eval on WordNetFood)
  cross-domain  --train CellOntology LLMs4OL_SchemaOrg      (eval on WordNetFood)

    python lora_train.py --sft_dir sft --train WordNetFood --out_dir adapters/in_WordNetFood

Serve the adapter with:  vllm serve <base> --enable-lora --lora-modules taxo=<out_dir>
then evaluate with:      python main.py --model taxo ...
"""

import os
import json
import glob
import argparse


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
    ap.add_argument("--base_model", default="google/gemma-4-31b-it")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true", help="Load + tokenize + report, then exit.")
    args = ap.parse_args()

    print(f"[*] loading SFT data from {args.sft_dir}/")
    records = load_records(args.sft_dir, args.train, set(args.exclude))
    if not records:
        raise SystemExit("[!] no training records -- run lora_data_prep.py first.")
    doms = sorted({r["domain"] for r in records})
    print(f"[*] {len(records)} examples across {len(doms)} domain(s): {', '.join(doms)}")

    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(records, tok, args.max_seq_len)
    lens = [len(ds[i]["input_ids"]) for i in range(min(len(ds), 200))]
    print(f"[*] token length (first {len(lens)}): mean={sum(lens)/len(lens):.0f} max={max(lens)} "
          f"(cap {args.max_seq_len})")
    if args.dry_run:
        ex = ds[0]
        n_sup = sum(1 for x in ex["labels"] if x != -100)
        print(f"[*] dry run -- example 0: {len(ex['input_ids'])} tokens, {n_sup} supervised (completion only)")
        print("---- prompt (first 400 chars) ----")
        print(records[0]["prompt"][:400])
        print("---- completion ----")
        print(records[0]["completion"][:400])
        return

    print(f"[*] loading {args.base_model} in 4-bit NF4")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

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
        json.dump({**vars(args), "train_domains": doms, "n_examples": len(records)}, f, indent=2)
    print(f"[*] adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()
