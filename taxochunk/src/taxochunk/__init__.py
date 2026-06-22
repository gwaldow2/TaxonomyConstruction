"""taxochunk -- chunked LLM taxonomy construction.

Build an is-a taxonomy (a parent->child DAG) from a flat list of terms by asking
an LLM, chunk by chunk, which terms are parents/children of each target.

Quick start
-----------
    from taxochunk import TaxonomyBuilder

    builder = TaxonomyBuilder.from_endpoint(
        base_url="http://localhost:8000/v1", api_key="EMPTY",
        model="openai/gpt-oss-120b",
    )
    G = builder.build(["food", "fruit", "apple", "granny smith"])
    builder.build_and_save(["food", "fruit", "apple"], "taxonomy.graphml")

Or stay functional:
    from taxochunk import build_taxonomy, openai_client
    client = openai_client("http://localhost:8000/v1", "EMPTY")
    G = build_taxonomy(terms, client=client, model="openai/gpt-oss-120b")

No LLM handy? Inject any ``prompt -> text`` callable via ``respond=...`` (great for tests).
"""

from .extract import build_taxonomy, make_openai_responder
from .builder import TaxonomyBuilder
from .client import openai_client
from .export import save_taxonomy, SUPPORTED_FORMATS
from .graph import enforce_dag, cluster_synonyms_and_enforce_dag
from .text import clean_term, get_primary_term, parse_lemma_format, to_lemma_format
from .prompts import build_prompt, parse_response

__version__ = "0.1.0"

__all__ = [
    "TaxonomyBuilder",
    "build_taxonomy",
    "openai_client",
    "make_openai_responder",
    "save_taxonomy",
    "SUPPORTED_FORMATS",
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
