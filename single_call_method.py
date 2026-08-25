"""Single-Call Baseline: the whole taxonomy from ONE prompt, written to current best practice.

This is what llm_zero was meant to be, built as a NEW method so llm_zero and every result
produced with it stay untouched as historical controls. Differences from llm_zero, each one a
lesson this project paid for:

  * explicit is-a definition with the extensional test ("every instance of the child is also an
    instance of the parent") -- the wording the prompt ablation found matters;
  * DIRECT-edges-only instruction, so the closure isn't fed redundant ancestor claims;
  * few-shot examples from a variety of NON-benchmark domains (music, medicine/buildings),
    demonstrating chains, multiple roots and unrelated distractor terms;
  * the vocabulary it shows is complete -- the legacy prompt's withheld-target defect caused an
    84% refusal rate, so the count and the rendered list here always agree;
  * the committed answer (``content``) is parsed first and the reasoning scratchpad is only a
    fallback, matching our_method's policy instead of concatenating both;
  * case-INSENSITIVE vocabulary matching -- the legacy JSON parser's case-sensitivity was a
    real edge-loss source.

``merge_synonyms`` controls whether mutually-asserted edges are condensed into synonym clusters
(the same condense_synonyms our_method uses) or left to plain DAG enforcement. Weak models
assert is-a in both directions freely -- untuned Qwen produced 60 reciprocal pairs on one
dataset, fusing a third of the vocabulary into one cluster -- so the flag lets the merge's
contribution be measured instead of assumed.

One API call per dataset: the n_calls=1 "lazy" comparison point for our method.
"""

import re

import networkx as nx

from data_manager import get_primary_term
from our_method import condense_synonyms, enforce_dag, _llm_call, EXTRACT_MAX_TOKENS

# A single ["parent", "child"] pair in either quote style. Findall over the whole text is
# deliberately lenient about the surrounding array: a missing closing bracket or trailing prose
# should not cost every edge (json.loads-or-nothing was another legacy failure mode).
_PAIR_RE = re.compile(r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]')

_FEW_SHOT = """EXAMPLE 1
Vocabulary (6 terms): [musical instrument, string instrument, guitar, electric guitar, percussion instrument, drum]
Output:
[
  ["musical instrument", "string instrument"],
  ["string instrument", "guitar"],
  ["guitar", "electric guitar"],
  ["musical instrument", "percussion instrument"],
  ["percussion instrument", "drum"]
]

EXAMPLE 2
Vocabulary (7 terms): [healthcare worker, nurse, doctor, surgeon, building, hospital, school]
Output:
[
  ["healthcare worker", "nurse"],
  ["healthcare worker", "doctor"],
  ["doctor", "surgeon"],
  ["building", "hospital"],
  ["building", "school"]
]"""


def build_single_call_prompt(primary_terms):
    """One prompt holding the ENTIRE vocabulary. The stated count and the rendered list are the
    same object, so they cannot disagree."""
    vocab = ", ".join(primary_terms)
    return f"""You are an expert ontologist constructing an is-a taxonomy.

TASK: given a vocabulary of terms, output every DIRECT parent-child (is-a) relationship that
holds between them.

RULES:
- "child is-a parent" means: every instance of the child is also an instance of the parent.
- Output DIRECT relationships only. If A > B > C, output ["A","B"] and ["B","C"], never ["A","C"].
- Use terms EXACTLY as they appear in the vocabulary. Do not invent, merge or reword terms.
- Not every term has a relationship; unrelated terms are simply omitted.
- Output ONLY a JSON list of ["parent", "child"] pairs. If there are none, output [].

{_FEW_SHOT}

NOW THE REAL TASK
Vocabulary ({len(primary_terms)} terms): [{vocab}]
Output:
"""


def parse_pairs(text, primary_to_full_map):
    """-> [(parent_node, child_node)] for pairs whose BOTH terms are in the vocabulary."""
    out = []
    for p, c in _PAIR_RE.findall(text or ""):
        pk, ck = p.strip().lower(), c.strip().lower()
        if pk in primary_to_full_map and ck in primary_to_full_map and pk != ck:
            out.append((primary_to_full_map[pk], primary_to_full_map[ck]))
    return out


def method_single_call(nodes, client, model_name, merge_synonyms=True,
                       max_tokens=EXTRACT_MAX_TOKENS, max_retries=3):
    """One LLM call over the whole vocabulary -> taxonomy DAG.

    merge_synonyms=True runs the same mutual-edge condensation as our_method; False leaves
    reciprocal assertions to enforce_dag's cycle-breaking, so the merge's effect is measurable.
    """
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    prompt = build_single_call_prompt(sorted(primary_to_full_map))

    content, reasoning = _llm_call(client, model_name, prompt, max_tokens, max_retries)
    # Committed answer first; the scratchpad only rescues an empty answer.
    edges = parse_pairs(content, primary_to_full_map) or parse_pairs(reasoning, primary_to_full_map)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print(f"    [Single-Call] {len(edges)} in-vocabulary edges parsed "
          f"({'content' if parse_pairs(content, primary_to_full_map) else 'reasoning fallback' if edges else 'nothing'})")

    if merge_synonyms:
        condensed, _ = condense_synonyms(G)
        return enforce_dag(condensed)
    return enforce_dag(G)
