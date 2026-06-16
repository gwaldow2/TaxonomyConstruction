"""Text normalization and synonym-cluster ("lemma") formatting.

A node can carry several synonymous surface forms. They are stored in a single
canonical string -- the "lemma format" -- as  ``primary (primary, syn2, syn3)``.
These helpers are pure (no third-party deps) and mirror the conventions used
across the TaxonomyConstruction benchmark so packaged output stays compatible.
"""

import re

__all__ = ["clean_term", "get_primary_term", "parse_lemma_format", "to_lemma_format"]


def clean_term(term):
    """Lowercase, strip punctuation (keeping word chars, spaces, hyphens), collapse spaces."""
    term = str(term).strip().lower()
    term = re.sub(r"[^\w\s\-]", "", term)
    term = re.sub(r"\s+", " ", term)
    return term.strip()


def get_primary_term(node_str):
    """Return the primary surface form of a (possibly multi-synonym) node string.

    ``"apple (apple, malus pumila)"`` -> ``"apple"``; a plain term is returned as-is.
    """
    node_str = str(node_str).strip().lower()
    match = re.search(r"^([^(]+)\s*\((.*)\)$", node_str)
    if match:
        primary_text = match.group(1).strip()
        syns = [s.strip() for s in match.group(2).split(",") if s.strip()]
        if syns and syns[0] == primary_text:
            return primary_text
    return node_str


def parse_lemma_format(node_str):
    """Return the list of surface forms merged into a node string.

    Accepts both ``"primary (primary, syn2)"`` and pipe-delimited ``"a|b|c"``.
    """
    node_str = str(node_str).strip().lower()
    match = re.search(r"^([^(]+)\s*\((.*)\)$", node_str)
    if match:
        primary_text = match.group(1).strip()
        syns = [s.strip() for s in match.group(2).split(",") if s.strip()]
        if syns and syns[0] == primary_text:
            return syns
    if "|" in node_str:
        return [s.strip() for s in node_str.split("|") if s.strip()]
    return [node_str]


def to_lemma_format(terms):
    """Collapse a collection of terms/synonyms into one canonical lemma-format string."""
    if not terms:
        return ""
    unique_terms = []
    for t in terms:
        for st in parse_lemma_format(str(t)):
            st = clean_term(st)
            if st and st not in unique_terms:
                unique_terms.append(st)

    if len(unique_terms) == 1:
        return unique_terms[0]
    primary = unique_terms[0]
    syns = ", ".join(unique_terms)
    return f"{primary} ({syns})"
