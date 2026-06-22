"""A small object wrapper so the method can be *instantiated* and reused.

    from taxochunk import TaxonomyBuilder

    # one-liner: build the OpenAI client for you
    builder = TaxonomyBuilder.from_endpoint(
        base_url="http://localhost:8000/v1", api_key="EMPTY",
        model="openai/gpt-oss-120b", chunk_size=1000,
    )
    G = builder.build(terms)
    builder.build_and_save(terms, "taxonomy.graphml")

    # or pass your own client
    from openai import OpenAI
    builder = TaxonomyBuilder(client=OpenAI(...), model="openai/gpt-oss-120b")
"""

from .extract import build_taxonomy
from .client import openai_client
from .export import save_taxonomy

__all__ = ["TaxonomyBuilder"]


class TaxonomyBuilder:
    """Holds the LLM connection + extraction settings; build taxonomies on demand."""

    def __init__(self, client=None, model=None, *, respond=None,
                 chunk_size=1000, alt_prompt=False, max_retries=3, retry_delay=2.0,
                 temperature=0.0, max_tokens=16328, keep_isolated=True,
                 show_progress=False, verbose=False):
        if respond is None and client is None:
            raise ValueError("Provide a client (with model=...) or a respond= callable.")
        if respond is None and model is None:
            raise ValueError("model= is required when using a client.")
        self.client = client
        self.model = model
        self.respond = respond
        self.options = dict(
            chunk_size=chunk_size, alt_prompt=alt_prompt, max_retries=max_retries,
            retry_delay=retry_delay, temperature=temperature, max_tokens=max_tokens,
            keep_isolated=keep_isolated, show_progress=show_progress, verbose=verbose,
        )

    @classmethod
    def from_endpoint(cls, base_url="http://localhost:8000/v1", api_key="EMPTY",
                      model=None, **options):
        """Build the OpenAI-compatible client for you, then return a builder."""
        if model is None:
            raise ValueError("model= is required.")
        return cls(client=openai_client(base_url, api_key), model=model, **options)

    def build(self, terms, **overrides):
        """Build and return the taxonomy (a ``networkx.DiGraph``).

        Any keyword in ``overrides`` temporarily overrides the stored option
        (e.g. ``builder.build(terms, chunk_size=100)``).
        """
        opts = {**self.options, **overrides}
        return build_taxonomy(terms, respond=self.respond, client=self.client,
                              model=self.model, **opts)

    def build_and_save(self, terms, path, fmt=None, **overrides):
        """Build the taxonomy and write it to ``path`` (GraphML by default)."""
        G = self.build(terms, **overrides)
        save_taxonomy(G, path, fmt=fmt)
        return G
