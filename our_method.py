import re
import time
from collections import Counter, defaultdict

import networkx as nx
from tqdm import tqdm

from data_manager import get_primary_term, to_lemma_format

def enforce_dag(G):
    G_dag = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = next(nx.simple_cycles(G_dag))
            G_dag.remove_edge(cycle[-1], cycle[0])
    except StopIteration:
        pass
    return G_dag

def condense_synonyms(G):
    """Merge mutually-pointing nodes into one cluster node.

    Returns (condensed_dag, node_mapping) where node_mapping maps each original
    node to its cluster name. (DAG enforcement is applied separately so callers
    can map per-edge metadata onto the condensed graph first.)
    """
    bidirectional_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    G_sym = nx.Graph()
    G_sym.add_nodes_from(G.nodes())
    G_sym.add_edges_from(bidirectional_edges)
    clusters = list(nx.connected_components(G_sym))

    condensed_dag = nx.DiGraph()
    node_mapping = {}
    for cluster in clusters:
        new_node_name = to_lemma_format(sorted(list(cluster)))
        condensed_dag.add_node(new_node_name)
        for node in cluster:
            node_mapping[node] = new_node_name

    for u, v in G.edges():
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        if new_u != new_v:
            condensed_dag.add_edge(new_u, new_v)

    return condensed_dag, node_mapping

def cluster_synonyms_and_enforce_dag(G):
    condensed_dag, _ = condense_synonyms(G)
    return enforce_dag(condensed_dag)

# ---- Relationship-rule blocks: the line(s) that tell the model what to output ----
def _rules_full(t):  # original "our approach": extensional subsumption, both directions
    return (f"- If every entity labeled with '{t}' could logically also be labeled with a candidate 'C', output '{t} <= C'\n"
            f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{t}', output 'C <= {t}'\n")

def _rules_direct(t):  # minimal-edit alternate: direct parent/child concept (SBU-style)
    return (f"- If a candidate 'C' is a parent concept of '{t}', output '{t} <= C'\n"
            f"- If a candidate 'C' is a child concept of '{t}', output 'C <= {t}'\n")

def _rules_isa(t):  # plain "is a kind of" -- relational but not extensional/instance-based
    return (f"- If '{t}' is a kind of a candidate 'C', output '{t} <= C'\n"
            f"- If a candidate 'C' is a kind of '{t}', output 'C <= {t}'\n")

def _rules_no_quantifier(t):  # full subsumption WITHOUT the "every entity labeled with" universal
    return (f"- If '{t}' could logically also be labeled with a candidate 'C', output '{t} <= C'\n"
            f"- If a candidate 'C' could logically also be labeled with '{t}', output 'C <= {t}'\n")

def _rules_oneway(t):  # full subsumption but only the superclass direction
    return (f"- If every entity labeled with '{t}' could logically also be labeled with a candidate 'C', output '{t} <= C'\n")

# Each variant ablates exactly ONE element of the original ('full') prompt, so a run
# of all of them isolates which element drives Our Method's gain.
PROMPT_VARIANTS = {
    "full":           {"rules": _rules_full,          "restriction": True,  "example": True},   # original (control)
    "direct":         {"rules": _rules_direct,        "restriction": True,  "example": True},   # subsumption -> direct parent/child concept
    "isa":            {"rules": _rules_isa,           "restriction": True,  "example": True},   # subsumption -> plain "is a kind of"
    "no_quantifier":  {"rules": _rules_no_quantifier, "restriction": True,  "example": True},   # drop the "every entity labeled with" universal
    "oneway":         {"rules": _rules_oneway,        "restriction": True,  "example": True},   # ask only the superclass direction
    "no_example":     {"rules": _rules_full,          "restriction": True,  "example": False},  # remove the worked example
    "no_restriction": {"rules": _rules_full,          "restriction": False, "example": True},   # remove the ONLY/Do-NOT scoping
}

def build_prompt(target_raw, candidates_chunk, alt_prompt=False, variant=None):
    """Build the per-target chunk prompt for a given prompt VARIANT.

    Variants are ablations of the original ('full') prompt -- each changes one
    element so a sweep over them identifies which element matters (see
    PROMPT_VARIANTS). `variant` overrides `alt_prompt`; `alt_prompt=True` is exactly
    `variant='direct'`. 'full' and 'direct' reproduce the previous default and
    alternate prompts byte-for-byte. Every variant emits the same
    'subordinate <= superordinate' format, so parsing is shared.
    """
    if variant is None:
        variant = "direct" if alt_prompt else "full"
    spec = PROMPT_VARIANTS[variant]
    instructions = (
        f"You are identifying hierarchical relationships for the target entity: '{target_raw}'.\n"
        f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
        f"between the target and the candidates.\n"
        + spec["rules"](target_raw)
    )
    if spec["restriction"]:
        instructions += (f"ONLY output relationships involving '{target_raw}'. "
                         f"Do NOT output relationships between the candidates themselves. ")
    instructions += "Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
    if spec["example"]:
        instructions += "Example: 'anucleate cell' <= 'cell'\n"
    instructions += "Candidates:\n"
    return instructions + "\n".join([f"- {c}" for c in candidates_chunk]) + "\n\nRelationships:\n"

def _llm_text(client, model_name, prompt, max_tokens, max_retries):
    """Single chat completion -> concatenated (reasoning + content), with retries.
    Returns "" if it never produced text."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            content = getattr(message, 'content', '') or ""
            reasoning = getattr(message, 'reasoning', '') or getattr(message, 'reasoning_content', '') or ""
            text = (str(reasoning) + "\n" + str(content)).strip()
            if text:
                return text
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(2)
    return ""

def _extract_condensed_with_votes(nodes, client, model_name, chunk_size=1000, max_retries=3,
                                  alt_prompt=False, variant=None):
    """Run the chunked extraction and return (condensed_dag, edge_votes).

    edge_votes[(parent, child)] is the GENERATION SELF-AGREEMENT signal: of the
    edge's TWO endpoints, how many -- when used as the prompt target -- asserted the
    edge (0, 1, or 2). It is NOT a raw emission count: a relation restated within one
    response, or several synonym members, never push it past 2.
    """
    raw_dag = nx.DiGraph()
    edge_targets = defaultdict(set)   # raw edge -> set of target terms that asserted it (deduped)
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())

    desc_label = f"Our Method [{variant}]" if variant else ("Our Method (alt. Prompt)" if alt_prompt else "Our Method")

    for target_raw in tqdm(primary_nodes, desc=f"  -> [{desc_label}] ChunkSize={chunk_size}", leave=False):
        all_candidates = [t for t in primary_nodes if t != target_raw]

        for i in range(0, len(all_candidates), chunk_size):
            candidates_chunk = all_candidates[i:i + chunk_size]
            prompt = build_prompt(target_raw, candidates_chunk, alt_prompt=alt_prompt, variant=variant)
            full_text = _llm_text(client, model_name, prompt, max_tokens=16328, max_retries=max_retries)
            if not full_text:
                continue

            for line in full_text.lower().split('\n'):
                if '<=' in line:
                    parts = line.split('<=')
                    if len(parts) == 2:
                        sub_raw = parts[0].strip().strip("-'\" ")
                        sup_raw = parts[1].strip().strip("-'\" ")
                        if sub_raw in primary_to_full_map and sup_raw in primary_to_full_map:
                            actual_sub = primary_to_full_map[sub_raw]
                            actual_sup = primary_to_full_map[sup_raw]
                            raw_dag.add_edge(actual_sup, actual_sub)
                            edge_targets[(actual_sup, actual_sub)].add(target_raw)

    condensed, node_mapping = condense_synonyms(raw_dag)
    final = enforce_dag(condensed)

    # Self-agreement votes (0/1/2): for each condensed edge U->W, did a target in
    # cluster U assert it (parent side) and/or a target in cluster W (child side)?
    edge_sides = defaultdict(set)
    for (a, c), targets in edge_targets.items():
        u, w = node_mapping.get(a), node_mapping.get(c)
        if u is None or w is None or u == w or not final.has_edge(u, w):
            continue
        for t in targets:
            t_cluster = node_mapping.get(primary_to_full_map.get(t))
            if t_cluster == u:
                edge_sides[(u, w)].add("parent")
            elif t_cluster == w:
                edge_sides[(u, w)].add("child")
    edge_votes = Counter({e: len(edge_sides.get(e, ())) for e in final.edges()})
    return final, edge_votes

# ----------------------------------------------------------------------------
# Precision clawback
# ----------------------------------------------------------------------------
def _pct_rank(values):
    """Map values to percentile ranks in [0, 1] (largest value -> 1.0)."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for r, i in enumerate(order):
        ranks[i] = r / (n - 1)
    return ranks

def edge_component_records(G, edge_votes):
    """Per-edge values of the three suspicion-heuristic components (no ground truth).

    Returns one dict per edge with the raw component values used by
    rank_suspicious_edges -- leverage, neighborhood_agreement, votes -- so they can
    be paired with an FP/TP label downstream to check whether each component is
    informative of errors.

      * leverage               -- (|ancestors(a)|+1)*(|descendants(c)|+1): closure
                                  pairs that route through the edge.
      * neighborhood_agreement -- of c's descendants, the fraction that ALSO have a
                                  raw edge directly from a. If a->c is a real
                                  ancestor relation then every descendant of c is
                                  also a descendant of a, and (since the method
                                  emits ancestor edges directly) a well-supported
                                  edge has many of them link straight back to a.
                                  Few/none agreeing => structurally uncorroborated
                                  => suspicious. Leaf children (no descendants)
                                  score 1.0 (nothing to corroborate).
      * votes                  -- how many of the two endpoint prompts asserted the
                                  edge (generation self-agreement).
    """
    if G.number_of_edges() == 0:
        return []
    closure = nx.transitive_closure(G)
    rows = []
    for (a, c) in G.edges():
        n_anc = len(list(closure.predecessors(a)))
        descendants_c = list(closure.successors(c))
        leverage = (n_anc + 1) * (len(descendants_c) + 1)
        if descendants_c:
            agree = sum(1 for d in descendants_c if G.has_edge(a, d))
            neighborhood = agree / len(descendants_c)
        else:
            neighborhood = 1.0   # leaf child: no neighborhood to corroborate
        rows.append({
            "parent": a,
            "child": c,
            "leverage": leverage,
            "neighborhood_agreement": neighborhood,
            "votes": edge_votes.get((a, c), 1),
        })
    return rows

def rank_suspicious_edges(G, edge_votes, w_leverage=1.0, w_neighborhood=1.0, w_agreement=1.0):
    """Order the edges of G from most to least suspicious for removal.

    Combines three GT-free heuristics (each a positive 'suspicion' contribution):
      * leverage     -- (|ancestors(a)|+1)*(|descendants(c)|+1): closure pairs that
                        route through the edge. A wrong high-leverage edge is what
                        damages closure precision the most.
      * neighborhood -- 1 - neighborhood_agreement: an edge a->c whose descendants
                        do NOT independently link back to a is structurally
                        uncorroborated, hence suspicious.
      * agreement    -- edges asserted by only ONE of the two endpoint prompts (a
                        single generation vote) are weakly corroborated -> suspicious.

    leverage is percentile-normalised; neighborhood disagreement (1 - agreement)
    and the single-vote bonus are already in [0, 1].
    """
    records = edge_component_records(G, edge_votes)
    if not records:
        return []
    lev_p = _pct_rank([r["leverage"] for r in records])
    scored = []
    for idx, r in enumerate(records):
        low_agreement = 1.0 if r["votes"] <= 1 else 0.0
        disagreement = 1.0 - r["neighborhood_agreement"]
        score = w_leverage * lev_p[idx] + w_neighborhood * disagreement + w_agreement * low_agreement
        scored.append((score, (r["parent"], r["child"])))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [edge for _, edge in scored]

def _decide_sever(text):
    """Return True iff the LLM's verdict text ends on SEVER (default: keep)."""
    tokens = re.findall(r'\b(keep|sever)\b', text.lower())
    return bool(tokens) and tokens[-1] == 'sever'

def _scrutiny_context(G, parent, child, max_items=6):
    """Neighbor-context lines + explicit 'neighborhood disagreement' statements for
    the edge parent -> child, to ground the LLM's audit in local structure."""
    P, C = get_primary_term(parent), get_primary_term(child)

    def _join(nodes):
        names = [get_primary_term(n) for n in nodes]
        shown = ", ".join(f"'{n}'" for n in names[:max_items])
        return shown + (f", ... (+{len(names) - max_items} more)" if len(names) > max_items else "")

    parent_parents = list(G.predecessors(parent))
    parent_children = [n for n in G.successors(parent) if n != child]
    child_parents = [n for n in G.predecessors(child) if n != parent]
    child_children = list(G.successors(child))

    lines = []
    if parent_parents:  lines.append(f"- '{P}' is a kind of: {_join(parent_parents)}.")
    if parent_children: lines.append(f"- '{P}' is also a parent of: {_join(parent_children)}.")
    if child_parents:   lines.append(f"- '{C}' is also a kind of: {_join(child_parents)}.")
    if child_children:  lines.append(f"- '{C}' is a parent of: {_join(child_children)}.")

    # Neighborhood disagreements: direct children d of child that do NOT independently
    # link back to parent (no raw edge parent -> d), stated as the heuristic sees them.
    inconsistencies = []
    for d in G.successors(child):
        if not G.has_edge(parent, d):
            inconsistencies.append(
                f"- You said '{P}' -> '{C}' and '{C}' -> '{get_primary_term(d)}', "
                f"but you did not say '{P}' -> '{get_primary_term(d)}'.")
        if len(inconsistencies) >= max_items:
            break
    return lines, inconsistencies

def scrutinize_edge(client, model_name, parent, child, G=None, max_retries=3, max_tokens=4096):
    """Ask the LLM to briefly reason about a single is-a edge and return 'KEEP'/'SEVER'.

    If the predicted graph G is given, the prompt includes local neighbor context
    around both endpoints and any 'neighborhood disagreements' -- descendants of the
    child that do not independently link back to the parent -- stated explicitly.
    """
    P, C = get_primary_term(parent), get_primary_term(child)
    context_block = ""
    if G is not None and G.has_node(parent) and G.has_node(child):
        lines, inconsistencies = _scrutiny_context(G, parent, child)
        if lines:
            context_block += "\nNeighborhood:\n" + "\n".join(lines) + "\n"
        context_block += ("\nInconsistencies:\n"
                          + ("\n".join(inconsistencies) if inconsistencies else "- none") + "\n")
    prompt = (
        f"You are auditing one is-a edge in a taxonomy.\n"
        f"Proposed relationship: '{C}' is a kind of '{P}'.\n"
        f"{context_block}"
        f"\nTaking the neighborhood and any inconsistencies above into account, briefly reason "
        f"(1-2 sentences) about whether every '{C}' is necessarily a type of '{P}'.\n"
        f"Then, on a new FINAL line, output exactly one word: "
        f"KEEP if the relationship is valid, or SEVER if it is incorrect and should be removed."
    )
    text = _llm_text(client, model_name, prompt, max_tokens=max_tokens, max_retries=max_retries)
    return "SEVER" if _decide_sever(text) else "KEEP"   # conservative default: KEEP

def precision_clawback(G, edge_votes, client, model_name, suspicion_candidates=0, max_retries=3):
    """Sever the LLM-confirmed-wrong edges among the top `suspicion_candidates` suspects.

    suspicion_candidates == 0 -> no clawback (returns G unchanged).
    """
    if not suspicion_candidates or suspicion_candidates <= 0:
        return G
    ranked = rank_suspicious_edges(G, edge_votes)
    G_out = G.copy()
    for (a, c) in ranked[:suspicion_candidates]:
        if scrutinize_edge(client, model_name, a, c, G, max_retries=max_retries) == "SEVER" and G_out.has_edge(a, c):
            G_out.remove_edge(a, c)
    return G_out

def method_our_approach(nodes, client, model_name, chunk_size=1000, max_retries=3,
                        alt_prompt=False, suspicion_candidates=0, variant=None):
    final_dag, edge_votes = _extract_condensed_with_votes(
        nodes, client, model_name, chunk_size=chunk_size, max_retries=max_retries,
        alt_prompt=alt_prompt, variant=variant)
    return precision_clawback(final_dag, edge_votes, client, model_name,
                              suspicion_candidates=suspicion_candidates, max_retries=max_retries)

def method_our_approach_sweep(nodes, client, model_name, suspicion_candidates_list,
                              chunk_size=1000, max_retries=3, alt_prompt=False, variant=None):
    """Extract ONCE, then return ({K: graph}, edge_components).

    {K: graph} is one taxonomy per K in suspicion_candidates_list. The suspicion
    ranking is fixed, so the top-K suspects are nested; each edge's LLM verdict is
    computed at most once (for the largest K) and reused across all sweep points.
    This makes a full F1-vs-K sweep cost one extraction plus max(K) clawback calls
    -- not a re-extraction per K.

    edge_components is the per-edge heuristic-component table for the BASE
    (pre-clawback) graph (see edge_component_records); pair it with an FP/TP label
    downstream to test whether the components are informative of errors.
    """
    final_dag, edge_votes = _extract_condensed_with_votes(
        nodes, client, model_name, chunk_size=chunk_size, max_retries=max_retries,
        alt_prompt=alt_prompt, variant=variant)
    ks = sorted({int(k) for k in suspicion_candidates_list})
    ranked = rank_suspicious_edges(final_dag, edge_votes)

    kmax = max(ks) if ks else 0
    verdicts = {}
    for (a, c) in ranked[:kmax]:
        verdicts[(a, c)] = scrutinize_edge(client, model_name, a, c, final_dag, max_retries=max_retries)

    out = {}
    for K in ks:
        G = final_dag.copy()
        for edge in ranked[:max(0, K)]:
            if verdicts.get(edge) == "SEVER" and G.has_edge(*edge):
                G.remove_edge(*edge)
        out[K] = G
    return out, edge_component_records(final_dag, edge_votes)
