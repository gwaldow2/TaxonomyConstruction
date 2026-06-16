"""Core algorithm: build an is-a taxonomy from a flat list of terms via chunked LLM prompting.

For every target term, the candidate vocabulary is split into chunks (to fit the
model's context window) and the model is asked which candidates are parents/children
of the target. Parsed relations are accumulated into a directed graph, then
synonym-clustered and reduced to a DAG.
"""

import time

import networkx as nx

from .text import get_primary_term
from .prompts import build_prompt, parse_response
from .graph import cluster_synonyms_and_enforce_dag

__all__ = ["build_taxonomy", "make_openai_responder"]


def make_openai_responder(client, model, temperature=0.0, max_tokens=16328):
    """Wrap an OpenAI-compatible ``client`` into a ``prompt -> text`` callable.

    ``client`` only needs a ``client.chat.completions.create(...)`` method, so this
    works with the official ``openai`` SDK or any compatible server (e.g. vLLM). Both
    ``content`` and any ``reasoning`` field are concatenated, matching the benchmark.
    """
    def respond(prompt):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = resp.choices[0].message
        content = getattr(message, "content", "") or ""
        reasoning = getattr(message, "reasoning", "") or getattr(message, "reasoning_content", "") or ""
        return (str(reasoning) + "\n" + str(content)).strip()

    return respond


def build_taxonomy(
    terms,
    *,
    respond=None,
    client=None,
    model=None,
    chunk_size=1000,
    alt_prompt=False,
    max_retries=3,
    retry_delay=2.0,
    temperature=0.0,
    max_tokens=16328,
    keep_isolated=True,
    show_progress=False,
    verbose=False,
):
    """Build a taxonomy (a ``networkx.DiGraph`` of parent -> child edges) from ``terms``.

    Parameters
    ----------
    terms : iterable of str
        The vocabulary. Terms may carry synonyms in lemma format,
        e.g. ``"apple (apple, malus pumila)"``.
    respond : callable, optional
        A ``prompt:str -> text:str`` function. Supply this to use any LLM (or a
        stub for testing). If omitted, ``client`` and ``model`` must be given and
        an OpenAI-compatible responder is built via :func:`make_openai_responder`.
    client, model :
        An OpenAI-compatible client and model id (used only when ``respond`` is None).
    chunk_size : int
        Number of candidate terms per prompt (lower = more focused, more calls).
    alt_prompt : bool
        Use the JSON ``[["parent","child"], ...]`` prompt instead of ``child <= parent`` lines.
    max_retries, retry_delay :
        Per-chunk retry budget and back-off (seconds) on empty/malformed/error responses.
    keep_isolated : bool
        If True (default) every input term appears in the result even with no relations.
        Set False to reproduce the benchmark's edge-only graph.
    show_progress : bool
        Show a tqdm bar over targets if ``tqdm`` is installed.
    verbose : bool
        Print a message when a chunk exhausts its retries.

    Returns
    -------
    networkx.DiGraph
        Synonym-clustered, acyclic parent -> child taxonomy.
    """
    if respond is None:
        if client is None or model is None:
            raise ValueError("Provide respond=callable, or both client= and model=.")
        respond = make_openai_responder(client, model, temperature, max_tokens)

    terms = list(terms)
    primary_to_full = {get_primary_term(n): n for n in terms}
    primary_nodes = list(primary_to_full.keys())
    vocab_set = set(primary_nodes)

    dag = nx.DiGraph()
    if keep_isolated:
        dag.add_nodes_from(terms)

    targets = primary_nodes
    if show_progress:
        try:
            from tqdm import tqdm
            targets = tqdm(primary_nodes, desc="taxochunk", leave=False)
        except ImportError:
            pass

    for target in targets:
        candidates = [t for t in primary_nodes if t != target]
        for i in range(0, len(candidates), chunk_size):
            chunk = candidates[i:i + chunk_size]
            prompt = build_prompt(target, chunk, alt_prompt=alt_prompt)

            for attempt in range(max_retries):
                try:
                    text = respond(prompt)
                    if not text:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        break
                    try:
                        pairs = parse_response(text, vocab_set, alt_prompt=alt_prompt)
                    except ValueError:   # json.JSONDecodeError is a subclass of ValueError
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        break
                    for parent, child in pairs:
                        dag.add_edge(primary_to_full[parent], primary_to_full[child])
                    break
                except Exception as e:               # network / API hiccups
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    elif verbose:
                        print(f"[taxochunk] target '{target}' chunk {i // chunk_size} "
                              f"failed after {max_retries} attempts: {e}")

    return cluster_synonyms_and_enforce_dag(dag)
