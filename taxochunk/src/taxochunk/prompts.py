"""Prompt templates and response parsing.

The default and alternate prompts are IDENTICAL except for the relationship-request
block -- a linguistic ablation (mirrors our_method.py in the benchmark):

  * default   : asks for ALL ANCESTORS via subsumption
                ("every entity labeled with X could logically also be labeled with C").
  * alternate : asks only for DIRECT parent/child connections, SBU-style
                ("C is a parent / child concept of X").

Both emit the same ``child <= parent`` lines, so a single parser handles both.
"""

__all__ = ["build_prompt", "parse_response"]


def build_prompt(target, candidates_chunk, alt_prompt=False):
    """Build the per-target prompt over one chunk of candidate terms."""
    # ===== ABLATION: the ONLY text that differs between the two prompts =====
    if alt_prompt:
        relationship_rules = (
            f"- If a candidate 'C' is a parent concept of '{target}', output '{target} <= C'\n"
            f"- If a candidate 'C' is a child concept of '{target}', output 'C <= {target}'\n"
        )
    else:
        relationship_rules = (
            f"- If every entity labeled with '{target}' could logically also be labeled with a candidate 'C', output '{target} <= C'\n"
            f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{target}', output 'C <= {target}'\n"
        )
    # =======================================================================
    instructions = (
        f"You are identifying hierarchical relationships for the target entity: '{target}'.\n"
        f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
        f"between the target and the candidates.\n"
        + relationship_rules +
        f"ONLY output relationships involving '{target}'. Do NOT output relationships between the candidates themselves. "
        f"Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
        f"Example: 'anucleate cell' <= 'cell'\n"
        f"Candidates:\n"
    )
    return instructions + "\n".join([f"- {c}" for c in candidates_chunk]) + "\n\nRelationships:\n"


def parse_response(text, vocab_set):
    """Parse ``child <= parent`` lines into a list of (parent, child) term pairs.

    Only pairs whose BOTH terms are in ``vocab_set`` are kept. Returns raw term
    strings (the caller maps them back to full node names).
    """
    pairs = []
    for line in text.lower().split("\n"):
        if "<=" in line:
            parts = line.split("<=")
            if len(parts) == 2:
                child = parts[0].strip().strip("-'\" ")
                parent = parts[1].strip().strip("-'\" ")
                if child in vocab_set and parent in vocab_set:
                    pairs.append((parent, child))
    return pairs
