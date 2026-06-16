"""Command-line interface: ``taxochunk terms.txt --base-url ... --model ...``.

Reads one term per line (lemma format allowed), builds the taxonomy against an
OpenAI-compatible endpoint, and writes the result as a GraphML file or as a
parent<TAB>child edge list on stdout.
"""

import sys
import argparse

import networkx as nx

from . import __version__
from .extract import build_taxonomy


def _read_terms(path):
    stream = sys.stdin if path in (None, "-") else open(path, "r", encoding="utf-8")
    try:
        return [ln.strip() for ln in stream if ln.strip()]
    finally:
        if stream is not sys.stdin:
            stream.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="taxochunk", description="Build an is-a taxonomy from a term list via chunked LLM prompting.")
    ap.add_argument("terms", nargs="?", default="-", help="File with one term per line (default: stdin)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--api-key", default="EMPTY", help="API key (any non-empty string for local servers)")
    ap.add_argument("--model", required=True, help="Model id, e.g. openai/gpt-oss-120b")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--alt-prompt", action="store_true", help="Use the JSON parent/child prompt")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--no-isolated", action="store_true", help="Drop terms that get no relations (benchmark parity)")
    ap.add_argument("--out", help="Write GraphML here (default: print edge list to stdout)")
    ap.add_argument("--progress", action="store_true", help="Show a progress bar (needs tqdm)")
    ap.add_argument("--version", action="version", version=f"taxochunk {__version__}")
    args = ap.parse_args(argv)

    try:
        from openai import OpenAI
    except ImportError:
        ap.error("the CLI needs the 'openai' package: pip install 'taxochunk[openai]'")

    terms = _read_terms(args.terms)
    if not terms:
        ap.error("no terms provided")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    G = build_taxonomy(
        terms,
        client=client,
        model=args.model,
        chunk_size=args.chunk_size,
        alt_prompt=args.alt_prompt,
        max_retries=args.max_retries,
        keep_isolated=not args.no_isolated,
        show_progress=args.progress,
        verbose=True,
    )

    if args.out:
        nx.write_graphml(G, args.out)
        print(f"Wrote {G.number_of_nodes()} nodes / {G.number_of_edges()} edges to {args.out}", file=sys.stderr)
    else:
        for u, v in G.edges():
            print(f"{u}\t{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
