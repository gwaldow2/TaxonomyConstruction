"""Prompt templates and response parsing for the two prompting strategies.

standard : the model emits ``child <= parent`` lines (one relation per line).
alt      : the model emits a JSON list of ``["parent", "child"]`` pairs.
"""

import re
import json

__all__ = ["build_prompt", "parse_response"]


def build_prompt(target, candidates_chunk, alt_prompt=False):
    """Build the per-target prompt over one chunk of candidate terms."""
    if alt_prompt:
        vocab = [target] + list(candidates_chunk)
        return f"""You are an expert ontologist building a hierarchical taxonomy.
                You are given a vocabulary of {len(vocab)} terms.

                A parent is a broader concept, a child is a more specific concept.
                ONLY use terms EXACTLY as they appear in the vocabulary list.

                Candidates: [{list(candidates_chunk)}]
                ONLY output relationships involving '{target}'. Do NOT output relationships between the candidates themselves.
                Format Example:
                [
                ["parent_term_1", "child_term_1"],
                ["parent_term_2", "child_term_2"]
                ]

                Output your answer strictly as a list of arrays. Do not add conversational text.
                """
    instructions = (
        f"You are identifying hierarchical relationships for the target entity: '{target}'.\n"
        f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
        f"between the target and the candidates.\n"
        f"- If every entity labeled with '{target}' could logically also be labeled with a candidate 'C', output '{target} <= C'\n"
        f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{target}', output 'C <= {target}'\n"
        f"ONLY output relationships involving '{target}'. Do NOT output relationships between the candidates themselves. "
        f"Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
        f"Example: 'anucleate cell' <= 'cell'\n"
        f"Candidates:\n"
    )
    return instructions + "\n".join([f"- {c}" for c in candidates_chunk]) + "\n\nRelationships:\n"


def parse_response(text, vocab_set, alt_prompt=False):
    """Parse model text into a list of (parent, child) term pairs.

    Only pairs whose BOTH terms are in ``vocab_set`` are kept. Returns the raw
    term strings (the caller maps them back to full node names). Raises
    ``json.JSONDecodeError`` on malformed alt-prompt JSON so the caller can retry.
    """
    pairs = []
    if alt_prompt:
        match = re.search(r"\[\s*\[.*\]\s*\]", text, re.DOTALL)
        json_str = match.group(0) if match else text
        relationships = json.loads(json_str)          # may raise -> caller retries
        for pair in relationships:
            if len(pair) == 2:
                parent, child = pair[0].strip(), pair[1].strip()
                if child in vocab_set and parent in vocab_set:
                    pairs.append((parent, child))
    else:
        for line in text.lower().split("\n"):
            if "<=" in line:
                parts = line.split("<=")
                if len(parts) == 2:
                    child = parts[0].strip().strip("-'\" ")
                    parent = parts[1].strip().strip("-'\" ")
                    if child in vocab_set and parent in vocab_set:
                        pairs.append((parent, child))
    return pairs
