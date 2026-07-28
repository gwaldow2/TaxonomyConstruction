"""Check that a fresh LoRA env can actually train, before committing hours of GPU time.

Exercises every API lora_train.py touches -- versions, CUDA, bitsandbytes 4-bit, the real
gemma4 tokenizer, TrainingArguments, peft -- and finishes with a genuine 4-bit LoRA
forward+backward on a tiny model. `--dry_run` in lora_train.py does NOT cover this: it
stops after tokenizing and never builds TrainingArguments or the peft model.

    python verify_lora_env.py
    python verify_lora_env.py --base gemma4
"""

import argparse
import traceback

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_results = []


def check(name):
    """Decorator: run a check, record pass/fail, keep going on failure."""
    def wrap(fn):
        print(f"\n=== {name} ===")
        try:
            fn()
            _results.append((name, None))
            print(f"    [ok] {name}")
        except Exception as e:
            _results.append((name, e))
            print(f"    [FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        return fn
    return wrap


def main():
    from lora_train import BASE_PRESETS, resolve_base       # single source of truth for presets

    ap = argparse.ArgumentParser(description="Verify a LoRA training environment.")
    ap.add_argument("--base", choices=sorted(BASE_PRESETS), default="qwen",
                    help="Verify the environment for this --base preset from lora_train.py.")
    ap.add_argument("--base_model", default=None, help="Override the preset's model id.")
    ap.add_argument("--skip_tokenizer", action="store_true", help="Skip the base-model tokenizer download.")
    ap.add_argument("--skip_smoke", action="store_true", help="Skip the tiny-model 4-bit train-step test.")
    ap.add_argument("--skip_base_lora", action="store_true",
                    help="Skip attaching LoRA to the REAL base (downloads/loads it). "
                         "This is the check that catches architecture gaps like Gemma 4's wrapped linears.")
    args = ap.parse_args()

    base_model = args.base_model or BASE_PRESETS[args.base]["model"]
    lora_targets = BASE_PRESETS[args.base]["target_modules"]
    print(f"[*] verifying --base {args.base} -> {base_model}")
    print(f"    target_modules: {lora_targets if lora_targets else 'PEFT defaults (auto)'}")

    @check("versions")
    def _versions():
        import torch, transformers, peft, accelerate, bitsandbytes
        print(f"    torch        {torch.__version__}")
        print(f"    transformers {transformers.__version__}")
        print(f"    peft         {peft.__version__}")
        print(f"    accelerate   {accelerate.__version__}")
        print(f"    bitsandbytes {bitsandbytes.__version__}")
        major = int(transformers.__version__.split(".")[0])
        assert major >= 5, "gemma4 needs transformers >= 5.14.1"
        # Enforces the preset's own floor (gemma4 needs peft>=0.19 for its LoRA defaults).
        resolve_base(argparse.Namespace(base=args.base, base_model=args.base_model,
                                        target_modules=None))

    @check("cuda")
    def _cuda():
        import torch
        assert torch.cuda.is_available(), "no CUDA device visible"
        print(f"    device     {torch.cuda.get_device_name(0)}")
        print(f"    capability {torch.cuda.get_device_capability(0)}")
        print(f"    total VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
        assert torch.cuda.is_bf16_supported(), "bf16 unsupported (the trainer uses bf16)"
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"    free VRAM  {free:.0f} GB")
        if free < 60:
            print("    [warn] under 60 GB free -- is vLLM still running? training will OOM.")

    @check("bitsandbytes 4-bit")
    def _bnb():
        import torch
        from transformers import BitsAndBytesConfig
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        import bitsandbytes as bnb
        # Actually move a 4-bit tensor onto the GPU -- import alone does not prove the CUDA
        # binary matches the installed toolkit, which is the usual failure after a CUDA bump.
        lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16).cuda()
        out = lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
        assert out.shape == (2, 64), out.shape
        print("    4-bit matmul on GPU OK")

    @check("TrainingArguments")
    def _targs():
        from transformers import TrainingArguments
        TrainingArguments(
            output_dir="/tmp/_verify_lora", num_train_epochs=1.0,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
            bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
            logging_steps=10, save_strategy="epoch", save_total_limit=2,
            report_to=[], seed=42, remove_unused_columns=False)
        print("    every kwarg lora_train.py passes is accepted")

    @check("peft LoraConfig")
    def _peft():
        from peft import LoraConfig
        LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"])
        print("    LoraConfig OK")

    if not args.skip_tokenizer:
        @check(f"tokenizer: {base_model}")
        def _tok():
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(base_model)
            print(f"    class {type(t).__name__}, vocab {len(t)}")
            ids = t("apple <= fruit").input_ids
            assert ids, "tokenizer produced no ids"
            print(f"    eos={t.eos_token!r} pad={t.pad_token!r}")

    if not args.skip_smoke:
        @check("4-bit LoRA train step (tiny model)")
        def _smoke():
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from lora_train import SFTDataset, make_collator

            tok = AutoTokenizer.from_pretrained(TINY)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                TINY, quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
                dtype=torch.bfloat16, device_map="auto")
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, LoraConfig(
                r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM"))

            recs = [{"prompt": "Candidates:\n- fruit\n\nRelationships:\n", "completion": "apple <= fruit"},
                    {"prompt": "Candidates:\n- food\n\nRelationships:\n", "completion": "none"}]
            ds = SFTDataset(recs, tok, 128)
            batch = make_collator(tok.pad_token_id)([ds[0], ds[1]])
            # Prompt tokens must be masked out; only the completion is supervised.
            n_sup = int((batch["labels"] != -100).sum())
            assert 0 < n_sup < batch["labels"].numel(), f"bad masking: {n_sup}"
            print(f"    loss masking: {n_sup}/{batch['labels'].numel()} tokens supervised")

            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert grads, "no LoRA parameter received a gradient"
            print(f"    forward+backward OK: loss={loss.item():.4f}, {len(grads)} LoRA tensors got grads")

    if not args.skip_base_lora:
        @check(f"attach LoRA to real base: {base_model}")
        def _base_attach():
            # The definitive check: a tiny stand-in model can't tell you whether PEFT can wrap
            # THIS architecture's modules. Gemma 4, for one, wraps its vision/audio encoder
            # projections in Gemma4ClippableLinear, which PEFT's type allowlist refuses -- and
            # only loading the actual base surfaces that. The tiny smoke test cannot.
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            print(f"    loading {base_model} in 4-bit (downloads if not cached)...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
                dtype=torch.bfloat16, device_map="auto")
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            try:
                model = get_peft_model(model, LoraConfig(
                    r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                    **({"target_modules": lora_targets} if lora_targets else {})))
            except ValueError as e:
                raise AssertionError(
                    f"PEFT cannot attach LoRA to {base_model} with "
                    f"target_modules={lora_targets or 'PEFT defaults'}. If this is a multimodal "
                    f"base, an explicit leaf-name list also matches its vision/audio encoder "
                    f"projections; leaving target_modules unset lets PEFT scope to the language "
                    f"model. Otherwise upgrade PEFT or pick a supported base. Original: {e}") from e
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert n_train > 0, "no trainable LoRA parameters were created"
            adapted = sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_modules() if ".lora_A" in n})
            leaves = sorted({a.rsplit(".", 1)[-1] for a in adapted})
            print(f"    LoRA attached: {n_train:,} trainable params over {len(adapted)} modules")
            print(f"    leaf types: {', '.join(leaves)}")
            print(f"    e.g. {adapted[0] if adapted else '(none)'}")
            stray = [a for a in adapted if any(k in a for k in ("vision", "audio", "multi_modal"))]
            if stray:
                print(f"    [warn] {len(stray)} adapter(s) on vision/audio modules, e.g. {stray[0]}")

    failed = [(n, e) for n, e in _results if e is not None]
    print("\n" + "=" * 60)
    for n, e in _results:
        print(f"  {'FAIL' if e else 'ok  '}  {n}")
    if failed:
        print(f"\n{len(failed)} check(s) failed -- do not start training yet.")
        raise SystemExit(1)
    print("\nEnvironment looks good. Next: python lora_train.py ... --dry_run")


if __name__ == "__main__":
    main()
