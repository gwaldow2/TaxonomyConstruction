import re
import time
from collections import Counter

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

def build_prompt(target_raw, candidates_chunk, alt_prompt=False):
    """Build the per-target chunk prompt.

    The default and alternate prompts are IDENTICAL except for the single
    relationship-request block flagged below. This is the linguistic ABLATION:

      * default (our approach): asks for ALL ANCESTORS via logical subsumption --
        "every entity labeled with X could logically also be labeled with C".
      * alternate: asks only for DIRECT parent/child connections, the way the SBU
        batch method does -- "C is a parent / child concept of X".

    Both prompts elicit the same 'subordinate <= superordinate' output format, so
    everything else (instructions, example, candidate list, parsing) is shared and
    the only variable is the linguistic framing of the relationship.
    """
    # ===== ABLATION: the ONLY text that differs between the two prompts =====
    if alt_prompt:
        relationship_rules = (
            f"- If a candidate 'C' is a parent concept of '{target_raw}', output '{target_raw} <= C'\n"
            f"- If a candidate 'C' is a child concept of '{target_raw}', output 'C <= {target_raw}'\n"
        )
    else:
        relationship_rules = (
            f"- If every entity labeled with '{target_raw}' could logically also be labeled with a candidate 'C', output '{target_raw} <= C'\n"
            f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{target_raw}', output 'C <= {target_raw}'\n"
        )
    # =======================================================================
    instructions = (
        f"You are identifying hierarchical relationships for the target entity: '{target_raw}'.\n"
        f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
        f"between the target and the candidates.\n"
        + relationship_rules +
        f"ONLY output relationships involving '{target_raw}'. Do NOT output relationships between the candidates themselves. "
        f"Output each relationship on a new line. If there are no relationships, output 'none'.\n\n"
        f"Example: 'anucleate cell' <= 'cell'\n"
        f"Candidates:\n"
    )
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

def _extract_condensed_with_votes(nodes, client, model_name, chunk_size=1000, max_retries=3, alt_prompt=False):
    """Run the chunked extraction and return (condensed_dag, edge_votes).

    edge_votes[(parent, child)] counts how many of the (up to 2) independent
    prompts -- one with each endpoint as the target -- asserted that edge. This
    is the generation self-agreement signal used by the precision-clawback step.
    """
    raw_dag = nx.DiGraph()
    raw_votes = Counter()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())

    desc_label = "Our Method (alt. Prompt)" if alt_prompt else "Our Method"

    for target_raw in tqdm(primary_nodes, desc=f"  -> [{desc_label}] ChunkSize={chunk_size}", leave=False):
        all_candidates = [t for t in primary_nodes if t != target_raw]

        for i in range(0, len(all_candidates), chunk_size):
            candidates_chunk = all_candidates[i:i + chunk_size]
            prompt = build_prompt(target_raw, candidates_chunk, alt_prompt=alt_prompt)
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
                            raw_votes[(actual_sup, actual_sub)] += 1

    condensed, node_mapping = condense_synonyms(raw_dag)
    final = enforce_dag(condensed)

    edge_votes = Counter()
    for (a, c), v in raw_votes.items():
        u, w = node_mapping.get(a), node_mapping.get(c)
        if u is not None and w is not None and u != w and final.has_edge(u, w):
            edge_votes[(u, w)] += v
    return final, edge_votes

# ----------------------------------------------------------------------------
# Precision clawback
# ----------------------------------------------------------------------------
def _node_depths(G):
    """Longest-path depth of each node from a root (0 for roots)."""
    depth = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth

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

def rank_suspicious_edges(G, edge_votes, w_leverage=1.0, w_depth_skip=1.0, w_agreement=1.0):
    """Order the edges of G from most to least suspicious for removal.

    Combines three GT-free heuristics (each a positive 'suspicion' contribution):
      * leverage   -- (|ancestors(a)|+1)*(|descendants(c)|+1): closure pairs that
                      route through the edge. A wrong high-leverage edge is what
                      damages closure precision the most.
      * depth_skip -- depth(c) - depth(a): how many hierarchy levels the edge spans
                      (over-generalisation candidates skip many levels).
      * agreement  -- edges asserted by only ONE of the two endpoint prompts (a
                      single generation vote) are weakly corroborated -> suspicious.

    leverage and depth_skip are percentile-normalised so they are comparable; the
    agreement term is 1.0 for single-vote edges, else 0.0.
    """
    edges = list(G.edges())
    if not edges:
        return []
    depth = _node_depths(G)
    leverage = [(len(nx.ancestors(G, a)) + 1) * (len(nx.descendants(G, c)) + 1) for a, c in edges]
    depth_skip = [depth[c] - depth[a] for a, c in edges]
    lev_p = _pct_rank(leverage)
    skip_p = _pct_rank(depth_skip)

    scored = []
    for idx, (a, c) in enumerate(edges):
        votes = edge_votes.get((a, c), 1)
        low_agreement = 1.0 if votes <= 1 else 0.0
        score = w_leverage * lev_p[idx] + w_depth_skip * skip_p[idx] + w_agreement * low_agreement
        scored.append((score, (a, c)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [edge for _, edge in scored]

def _decide_sever(text):
    """Return True iff the LLM's verdict text ends on SEVER (default: keep)."""
    tokens = re.findall(r'\b(keep|sever)\b', text.lower())
    return bool(tokens) and tokens[-1] == 'sever'

def scrutinize_edge(client, model_name, parent, child, max_retries=3, max_tokens=4096):
    """Ask the LLM to briefly reason about a single is-a edge and return 'KEEP'/'SEVER'."""
    p_disp = get_primary_term(parent)
    c_disp = get_primary_term(child)
    prompt = (
        f"You are auditing one is-a edge in a taxonomy.\n"
        f"Proposed relationship: '{c_disp}' is a kind of '{p_disp}'.\n"
        f"Briefly reason (1-2 sentences) about whether every '{c_disp}' is necessarily a type of '{p_disp}'.\n"
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
        if scrutinize_edge(client, model_name, a, c, max_retries=max_retries) == "SEVER" and G_out.has_edge(a, c):
            G_out.remove_edge(a, c)
    return G_out

def method_our_approach(nodes, client, model_name, chunk_size=1000, max_retries=3,
                        alt_prompt=False, suspicion_candidates=0):
    final_dag, edge_votes = _extract_condensed_with_votes(
        nodes, client, model_name, chunk_size=chunk_size, max_retries=max_retries, alt_prompt=alt_prompt)
    return precision_clawback(final_dag, edge_votes, client, model_name,
                              suspicion_candidates=suspicion_candidates, max_retries=max_retries)

def method_our_approach_sweep(nodes, client, model_name, suspicion_candidates_list,
                              chunk_size=1000, max_retries=3, alt_prompt=False):
    """Extract ONCE, then return {K: graph} for each K in suspicion_candidates_list.

    The suspicion ranking is fixed, so the top-K suspects are nested; each edge's
    LLM verdict is computed at most once (for the largest K) and reused across all
    sweep points. This makes a full F1-vs-K sweep cost one extraction plus
    max(K) clawback calls -- not a re-extraction per K.
    """
    final_dag, edge_votes = _extract_condensed_with_votes(
        nodes, client, model_name, chunk_size=chunk_size, max_retries=max_retries, alt_prompt=alt_prompt)
    ks = sorted({int(k) for k in suspicion_candidates_list})
    ranked = rank_suspicious_edges(final_dag, edge_votes)

    kmax = max(ks) if ks else 0
    verdicts = {}
    for (a, c) in ranked[:kmax]:
        verdicts[(a, c)] = scrutinize_edge(client, model_name, a, c, max_retries=max_retries)

    out = {}
    for K in ks:
        G = final_dag.copy()
        for edge in ranked[:max(0, K)]:
            if verdicts.get(edge) == "SEVER" and G.has_edge(*edge):
                G.remove_edge(*edge)
        out[K] = G
    return out
