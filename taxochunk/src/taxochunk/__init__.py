"""taxochunk -- chunked LLM taxonomy construction.

Build an is-a taxonomy (a parent->child DAG) from a flat list of terms by asking
an LLM, chunk by chunk, which terms are parents/children of each target.

Quick start
-----------
    from taxochunk import build_taxonomy
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
    G = build_taxonomy(["food", "fruit", "apple", "granny smith"],
                       client=client, model="openai/gpt-oss-120b")
    print(list(G.edges()))

No LLM handy? Inject any ``prompt -> text`` callable via ``respond=...`` (great for tests).
"""

from .extract import build_taxonomy, make_openai_responder
from .graph import enforce_dag, cluster_synonyms_and_enforce_dag
from .text import clean_term, get_primary_term, parse_lemma_format, to_lemma_format
from .prompts import build_prompt, parse_response

__version__ = "0.1.0"

__all__ = [
    "build_taxonomy",
    "make_openai_responder",
    "enforce_dag",
    "cluster_synonyms_and_enforce_dag",
    "clean_term",
    "get_primary_term",
    "parse_lemma_format",
    "to_lemma_format",
    "build_prompt",
    "parse_response",
    "__version__",
]
