"""Self-checks for lora_train's encoding, split and placement logic. No GPU, no model
download, no transformers -- the tokenizer is stubbed and torch is only needed for nn.Linear.

    python test_lora_train.py
"""

from lora_train import SFTDataset, split_train_val, all_linear_targets, PROJ_LEAVES


class _Tok:
    """Stub tokenizer: one id per whitespace token, chat template adds framing tokens.

    The template satisfies the prefix property SFTDataset relies on: rendering the
    conversation WITH the assistant turn extends the generation-prompt rendering exactly.
    """
    eos_token = "<eos>"
    chat_template = "stub"

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]

    def __call__(self, text, add_special_tokens=True):
        return type("O", (), {"input_ids": self._ids(text)})()

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        parts = ["<bos>"]
        for m in msgs:
            parts += [f"<{m['role']}>", m["content"], "<end>"]
        if add_generation_prompt:
            parts.append("<assistant>")
        return self._ids(" ".join(parts))


class _BrokenTok(_Tok):
    """Template that violates the prefix property (re-renders the whole conversation)."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        ids = super().apply_chat_template(msgs, add_generation_prompt, tokenize)
        return ids if add_generation_prompt else list(reversed(ids))


REC = {"prompt": "identify relations for apple", "completion": "apple <= fruit"}


def test_chat_encoding_masks_prompt_and_keeps_end_of_turn():
    tok = _Tok()
    ds = SFTDataset([REC], tok, max_len=4096)
    assert ds.use_chat_template
    p_ids, c_ids = ds.encode(REC)
    # prompt ends with the generation prompt, completion ends with the end-of-turn token
    assert p_ids[-1] == tok.vocab["<assistant>"]
    assert c_ids[-1] == tok.vocab["<end>"]
    assert tok.vocab["apple"] in c_ids            # the completion text itself is supervised
    ex = ds[0]
    assert ex["labels"][:len(p_ids)] == [-100] * len(p_ids)
    assert ex["labels"][len(p_ids):] == c_ids
    assert ex["input_ids"] == p_ids + c_ids


def test_prefix_violation_falls_back_to_raw():
    tok = _BrokenTok()
    ds = SFTDataset([REC], tok, max_len=4096)
    p_ids, c_ids = ds.encode(REC)
    assert ds._warned_prefix
    assert p_ids == tok(REC["prompt"]).input_ids
    assert c_ids == tok(REC["completion"] + tok.eos_token).input_ids


def test_no_template_and_optout_use_raw():
    bare = _Tok()
    bare.chat_template = None
    for ds in (SFTDataset([REC], bare, 4096),
               SFTDataset([REC], _Tok(), 4096, use_chat_template=False)):
        assert not ds.use_chat_template
        p_ids, c_ids = ds.encode(REC)
        assert p_ids == ds.tok(REC["prompt"]).input_ids
        assert c_ids == ds.tok(REC["completion"] + ds.tok.eos_token).input_ids


def test_truncation_preserves_completion():
    ds = SFTDataset([REC], _Tok(), max_len=6)
    ex = ds[0]
    _, c_ids = ds.encode(REC)
    assert len(ex["input_ids"]) <= 6
    assert ex["input_ids"][-len(c_ids):] == c_ids          # completion intact, prompt clipped
    assert sum(1 for x in ex["labels"] if x != -100) == len(c_ids)


def _recs(**counts):
    return [{"domain": d, "prompt": f"{d}{i}", "completion": "x"}
            for d, n in counts.items() for i in range(n)]


def test_split_is_stratified_seeded_and_lossless():
    recs = _recs(A=20, B=10, C=1)
    tr, va = split_train_val(recs, 0.1, seed=42)
    assert (tr, va) == split_train_val(recs, 0.1, seed=42)      # deterministic
    by_dom = lambda rows, d: [r for r in rows if r["domain"] == d]
    assert len(by_dom(va, "A")) == 2 and len(by_dom(va, "B")) == 1
    assert len(by_dom(va, "C")) == 0                            # singleton stays in train
    key = lambda r: (r["domain"], r["prompt"])
    assert sorted(map(key, tr + va)) == sorted(map(key, recs))  # nothing lost or duplicated
    assert split_train_val(recs, 0.1, seed=7) != (tr, va)       # seed matters


def test_split_zero_fraction_disables_validation():
    recs = _recs(A=5)
    tr, va = split_train_val(recs, 0.0, seed=42)
    assert tr == recs and va == []


def test_all_linear_targets_scans_lm_and_skips_towers():
    import torch.nn as nn

    class _Wrapper(nn.Module):
        """Stands in for Gemma4ClippableLinear: an nn.Module that is NOT an nn.Linear."""

    modules = [
        ("model.layers.0.self_attn.q_proj", nn.Linear(2, 2)),
        ("model.layers.0.self_attn.v_proj", nn.Linear(2, 2)),
        ("model.layers.0.mlp.up_proj", nn.Linear(2, 2)),
        ("model.layers.0.self_attn.rotary_emb", nn.Linear(2, 2)),   # leaf not a projection
        ("vision_tower.blocks.0.attn.q_proj", nn.Linear(2, 2)),     # multimodal marker
        ("model.audio_encoder.layers.0.v_proj", nn.Linear(2, 2)),   # multimodal marker
        ("encoder.blocks.0.q_proj", _Wrapper()),                    # wrapper, not nn.Linear
    ]
    got = all_linear_targets(modules)
    assert got == ["model.layers.0.mlp.up_proj",
                   "model.layers.0.self_attn.q_proj",
                   "model.layers.0.self_attn.v_proj"], got
    assert all(t.rsplit(".", 1)[-1] in PROJ_LEAVES for t in got)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll lora_train checks passed.")
