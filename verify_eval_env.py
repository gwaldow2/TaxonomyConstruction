"""Check which parts of the benchmark environment actually work, before a long run.

Imports each method path SEPARATELY and reports pass/fail per group instead of dying on the
first error, so a broken optional dependency (e.g. torchcodec missing FFmpeg, or a
libstdc++/CXXABI mismatch breaking sqlite3 -> nltk) is localised rather than mistaken for a
code bug. --method our_method needs only the "core" and "loaders" groups to pass.

    python verify_eval_env.py
    python verify_eval_env.py --server        # also ping the vLLM endpoint

Companion to verify_lora_env.py (training side).
"""

import argparse
import traceback

_results = []


def check(name, required=True):
    def wrap(fn):
        try:
            fn()
            _results.append((name, required, None))
            print(f"    [ok]   {name}")
        except Exception as e:
            _results.append((name, required, e))
            tag = "FAIL" if required else "warn"
            print(f"    [{tag}] {name}: {type(e).__name__}: {e}")
        return fn
    return wrap


def main():
    ap = argparse.ArgumentParser(description="Verify the benchmark/eval environment.")
    ap.add_argument("--server", action="store_true", help="Also ping the vLLM endpoint.")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="woohoo")
    ap.add_argument("--verbose", action="store_true", help="Print full tracebacks on failure.")
    args = ap.parse_args()

    print("\n=== core (required for --method our_method) ===")

    @check("stdlib sqlite3")
    def _sqlite():
        # Canary for the libstdc++/CXXABI mismatch: conda's ICU/sqlite compiled against a
        # newer C++ runtime than the loader provides. nltk imports sqlite3, so this failing
        # takes down data_manager and looks like a code error.
        import sqlite3
        sqlite3.connect(":memory:").close()

    @check("openai / networkx / pandas / numpy / tqdm")
    def _core():
        import openai, networkx, pandas, numpy, tqdm  # noqa: F401

    @check("our_method + evaluator (no torch needed)")
    def _method():
        import our_method, evaluator  # noqa: F401
        assert hasattr(our_method, "build_prompt")

    print("\n=== dataset loaders (required: main.py loads benchmark graphs) ===")

    @check("nltk / obonet / requests")
    def _loaders():
        import nltk, obonet, requests  # noqa: F401

    @check("data_manager")
    def _dm():
        import data_manager  # noqa: F401
        assert hasattr(data_manager, "load_benchmark_graph")

    @check("main.py imports")
    def _main():
        import importlib.util
        spec = importlib.util.spec_from_file_location("_m", "main.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert hasattr(m, "ALL_PROMPT_VARIANTS")

    print("\n=== visualization (required for vis_benchmarks / *_comparison) ===")

    @check("matplotlib / seaborn")
    def _vis():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot, seaborn  # noqa: F401

    print("\n=== optional baselines (only needed for those --method values) ===")

    @check("torch", required=False)
    def _torch():
        import torch
        print(f"           torch {torch.__version__}, cuda={torch.cuda.is_available()}")

    @check("sentence-transformers  (--method vector/lexical/sbu_embedding)", required=False)
    def _st():
        import transformers, sentence_transformers  # noqa: F401
        major = int(transformers.__version__.split(".")[0])
        print(f"           transformers {transformers.__version__}, "
              f"sentence-transformers {sentence_transformers.__version__}")
        assert major < 5, ("sentence-transformers requires transformers<5 -- this env has "
                           f"{transformers.__version__}. Training needs >=5.14, which is why "
                           "training and eval must be separate envs.")

    @check("peft / bitsandbytes  (--method taxollama)", required=False)
    def _taxo():
        import peft, bitsandbytes  # noqa: F401

    if args.server:
        print("\n=== vLLM endpoint ===")

        @check(f"models at {args.base_url}", required=False)
        def _srv():
            from openai import OpenAI
            served = [m.id for m in OpenAI(base_url=args.base_url,
                                           api_key=args.api_key).models.list().data]
            print(f"           serving: {served}")

    req_fail = [(n, e) for n, r, e in _results if e is not None and r]
    opt_fail = [(n, e) for n, r, e in _results if e is not None and not r]

    print("\n" + "=" * 62)
    for n, r, e in _results:
        print(f"  {'FAIL' if (e and r) else 'warn' if e else 'ok  '}  {n}")
    if args.verbose:
        for n, e in req_fail + opt_fail:
            print(f"\n--- {n} ---")
            traceback.print_exception(type(e), e, e.__traceback__)

    if opt_fail and not req_fail:
        print(f"\n{len(opt_fail)} OPTIONAL group(s) unavailable -- those --method values will not "
              f"run, but --method our_method is fine.")
    if req_fail:
        print(f"\n{len(req_fail)} REQUIRED check(s) failed -- fix these before running main.py.")
        raise SystemExit(1)
    print("\nEval environment OK.")


if __name__ == "__main__":
    main()
