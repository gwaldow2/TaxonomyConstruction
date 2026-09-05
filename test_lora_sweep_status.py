"""Self-checks for lora_sweep_status. No GPU, no LLM -- fake dir trees in a tempdir.

    python test_lora_sweep_status.py
"""

import os
import json
import shutil
import tempfile

from lora_sweep_status import classify, sft_count, train_command


def _tmp():
    return tempfile.mkdtemp()


def test_complete_requires_train_config():
    tmp = _tmp()
    try:
        d = os.path.join(tmp, "adapter")
        os.makedirs(os.path.join(d, "checkpoint-80"))
        status, _ = classify(d)
        assert status == "partial"                      # checkpoints alone are not completion
        with open(os.path.join(d, "train_config.json"), "w") as f:
            json.dump({"n_examples": 156, "stopped_epoch": 4.0, "best_eval_loss": 0.31}, f)
        status, info = classify(d)
        assert status == "complete" and info["n_examples"] == 156
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_partial_reports_latest_checkpoint():
    tmp = _tmp()
    try:
        d = os.path.join(tmp, "adapter")
        os.makedirs(os.path.join(d, "checkpoint-40"))
        os.makedirs(os.path.join(d, "checkpoint-120"))
        status, info = classify(d)
        assert status == "partial" and info["last_checkpoint_step"] == 120
        assert info["n_checkpoints"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_and_unreadable_config():
    tmp = _tmp()
    try:
        assert classify(os.path.join(tmp, "nope"))[0] == "missing"
        d = os.path.join(tmp, "bad")
        os.makedirs(d)
        with open(os.path.join(d, "train_config.json"), "w") as f:
            f.write("{not json")
        assert classify(d)[0] == "partial"              # a corrupt config is not a clean finish
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sft_count_skips_blank_lines():
    tmp = _tmp()
    try:
        with open(os.path.join(tmp, "X.jsonl"), "w") as f:
            f.write('{"a":1}\n\n{"b":2}\n')
        assert sft_count(tmp, "X") == 2
        assert sft_count(tmp, "Y") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_commands_match_the_sweep_loop():
    assert train_command("gemma4", "in", "WordNetFood", 1553, "adapters/gemma4V2_in_WordNetFood") == \
        "python lora_train.py --base gemma4 --train WordNetFood --out_dir adapters/gemma4V2_in_WordNetFood"
    c = train_command("gemma4", "cross", "WordNetFood", 1553, "adapters/gemma4V2_cross_WordNetFood")
    assert "--train all --exclude WordNetFood" in c
    assert "--max_examples 1553" in c                   # N baked in as a literal


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"    [ok] {name}")
    print("\nAll lora_sweep_status checks passed.")
