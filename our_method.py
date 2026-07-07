import re
import json
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

# 'legacy_json' is NOT a one-line ablation of 'full' -- it is the ORIGINAL alternate
# prompt from before the ablation was formalized: a different template ("expert
# ontologist") that asks for a JSON [["parent","child"], ...] answer parsed by
# _parse_relations_json. It lives outside PROMPT_VARIANTS (different template + parser)
# but is a valid --prompt_variant choice so the old default-vs-alt gap can be replicated.
ALL_PROMPT_VARIANTS = list(PROMPT_VARIANTS) + ["legacy_json", "full_json"]

def _build_legacy_json_prompt(target_raw, candidates_chunk):
    """Faithful reproduction of the ORIGINAL alt prompt (pre-formalization).

    Reproduced quirks-and-all so the original score is replicable: the 16-space
    indentation on every line and the candidate list rendered as a Python list repr
    (``Candidates: [[...]]``) are exactly as they were. The reply is JSON, parsed
    CASE-SENSITIVELY by _parse_relations_json -- unlike the '<=' variants, whose
    parser lowercases first.
    """
    vocab = [target_raw] + candidates_chunk
    return f"""You are an expert ontologist building a hierarchical taxonomy.
                You are given a vocabulary of {len(vocab)} terms.

                A parent is a broader concept, a child is a more specific concept.
                ONLY use terms EXACTLY as they appear in the vocabulary list.

                Candidates: [{candidates_chunk}]
                ONLY output relationships involving '{target_raw}'. Do NOT output relationships between the candidates themselves. 
                Format Example:
                [
                ["parent_term_1", "child_term_1"],
                ["parent_term_2", "child_term_2"]
                ]

                Output your answer strictly as a list of arrays. Do not add conversational text.
                """

def _build_full_json_prompt(target_raw, candidates_chunk):
    """The contemporary 'full' prompt -- identical framing, per-candidate subsumption
    wording, restriction, and CLEAN candidate list -- but requesting the legacy JSON
    [[parent, child], ...] OUTPUT (parsed by _parse_relations_json) instead of '<=' lines.

    It differs from 'full' ONLY in output format (+ parser), so full_json-vs-full isolates
    the effect of the JSON format, while full_json-vs-legacy_json isolates the effect of the
    'expert ontologist' wording and garbled candidate rendering.
    """
    t = target_raw
    instructions = (
        f"You are identifying hierarchical relationships for the target entity: '{t}'.\n"
        f"Below is a list of candidate entities. Identify any subclass or superclass relationships "
        f"between the target and the candidates.\n"
        f"- If every entity labeled with '{t}' could logically also be labeled with a candidate 'C', output the pair [\"C\", \"{t}\"]\n"
        f"- If every entity labeled with a candidate 'C' could logically also be labeled with '{t}', output the pair [\"{t}\", \"C\"]\n"
        f"ONLY output relationships involving '{t}'. Do NOT output relationships between the candidates themselves. "
        f"Output your answer strictly as a JSON list of [parent, child] arrays. If there are no relationships, output [].\n\n"
        f"Example: [[\"cell\", \"anucleate cell\"]]\n"
        f"Candidates:\n"
    )
    return instructions + "\n".join([f"- {c}" for c in candidates_chunk]) + "\n\nRelationships:\n"

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
    if variant == "legacy_json":
        return _build_legacy_json_prompt(target_raw, candidates_chunk)
    if variant == "full_json":
        return _build_full_json_prompt(target_raw, candidates_chunk)
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

def _llm_call(client, model_name, prompt, max_tokens, max_retries):
    """One chat completion -> (content, reasoning), with retries.

    Keeps the committed final answer (``content``) and the model's scratchpad
    (``reasoning``) SEPARATE so callers can prefer the answer. Returns ("","") if it
    never produced text.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            content = str(getattr(message, 'content', '') or "")
            reasoning = str(getattr(message, 'reasoning', '') or getattr(message, 'reasoning_content', '') or "")
            if content.strip() or reasoning.strip():
                return content, reasoning
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(2)
    return "", ""

def _llm_text(client, model_name, prompt, max_tokens, max_retries):
    """Concatenated (reasoning + content) text -- used by edge scrutiny."""
    content, reasoning = _llm_call(client, model_name, prompt, max_tokens, max_retries)
    return (reasoning + "\n" + content).strip()

def _parse_relations(text, primary_to_full_map):
    """Parse 'sub <= sup' lines into a list of (parent, child) edges (with repeats)."""
    out = []
    for line in text.lower().split('\n'):
        if '<=' in line:
            parts = line.split('<=')
            if len(parts) == 2:
                sub_raw = parts[0].strip().strip("-'\" ")
                sup_raw = parts[1].strip().strip("-'\" ")
                if sub_raw in primary_to_full_map and sup_raw in primary_to_full_map:
                    out.append((primary_to_full_map[sup_raw], primary_to_full_map[sub_raw]))
    return out

_JSON_BLOCK_RE = re.compile(r'\[\s*\[.*\]\s*\]', re.DOTALL)
# A single ["a", "b"] pair (either quote style); used by the lenient diagnostic to
# recover pairs even when the outer array is malformed / won't json.loads.
_JSON_PAIR_RE = re.compile(r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]')

def _parse_relations_json(text, primary_to_full_map):
    """Parse the legacy [["parent","child"], ...] JSON answer into (parent, child) edges.

    Faithful to the original alt path: it locates the outermost [[...]] block, json.loads
    it, and matches each pair's terms CASE-SENSITIVELY against the vocabulary (the '<='
    parser, by contrast, lowercases the whole text first). That asymmetry is a real
    edge-loss source whenever the model capitalises a term, which the parse diagnostics
    below quantify.
    """
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return []
    try:
        relationships = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    out = []
    if isinstance(relationships, list):
        for pair in relationships:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                sup_raw = str(pair[0]).strip()
                sub_raw = str(pair[1]).strip()
                if sub_raw in primary_to_full_map and sup_raw in primary_to_full_map:
                    out.append((primary_to_full_map[sup_raw], primary_to_full_map[sub_raw]))
    return out

def _parse_diagnostics(text, primary_to_full_map):
    """Cross-parse one response several ways to localise where edges are lost.

    Returns counts for the same text under: the faithful strict JSON parser (what feeds
    the graph), json.loads with CASE-INSENSITIVE vocab matching, a lenient pair-regex
    (ignores json.loads entirely), and the '<=' line parser. Comparing them separates a
    parsing artifact (edges recoverable by a laxer parser) from a genuine model/prompt
    effect (few well-formed in-vocab pairs to begin with).
    """
    vocab = primary_to_full_map                       # keys are already lowercased
    n_strict = len(_parse_relations_json(text, vocab))

    match = _JSON_BLOCK_RE.search(text)
    regex_match = match is not None
    json_ok, n_json_ci = False, 0
    if match:
        try:
            rel = json.loads(match.group(0))
            json_ok = True
            if isinstance(rel, list):
                for pair in rel:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
                        if a in vocab and b in vocab:
                            n_json_ci += 1
        except (json.JSONDecodeError, ValueError):
            json_ok = False

    n_regex_ci, n_pairs_wellformed = 0, 0
    for a, b in _JSON_PAIR_RE.findall(text):
        n_pairs_wellformed += 1
        if a.strip().lower() in vocab and b.strip().lower() in vocab:
            n_regex_ci += 1

    return {
        "regex_match": regex_match,
        "json_ok": json_ok,
        "n_strict": n_strict,              # feeds the graph (case-sensitive)
        "n_json_ci": n_json_ci,            # json.loads + case-insensitive match
        "n_regex_ci": n_regex_ci,          # pair-regex + case-insensitive (json.loads-free)
        "n_line": len(_parse_relations(text, vocab)),   # '<=' cross-check
        "n_pairs_wellformed": n_pairs_wellformed,       # ["a","b"] pairs seen (vocab-agnostic)
    }

def _emit_parse_debug(records, n_chunks, n_empty, label, debug_path):
    """Print an aggregate parse-diagnostics summary and, if given a path, dump the
    per-response records (raw output + parse counts) as JSONL for manual inspection."""
    n = len(records)
    s = lambda k: sum(r["diag"][k] for r in records)
    strict, json_ci, regex_ci, line, wf = (s("n_strict"), s("n_json_ci"),
                                            s("n_regex_ci"), s("n_line"), s("n_pairs_wellformed"))
    case_loss = json_ci - strict            # edges strict dropped purely on case
    jsonfail_recovery = regex_ci - json_ci  # edges only recoverable if json.loads is bypassed
    unknown = wf - regex_ci                 # well-formed pairs whose terms aren't in vocab
    print(f"  [parse-diag {label}] chunks={n_chunks} empty={n_empty} nonempty={n}")
    print(f"      JSON regex-match: {sum(r['diag']['regex_match'] for r in records)}/{n}   "
          f"json.loads ok: {sum(r['diag']['json_ok'] for r in records)}/{n}")
    print(f"      edges STRICT (case-sensitive, feeds graph): {strict}")
    print(f"      edges json.loads + case-INSENSITIVE:        {json_ci}   (+{case_loss} recoverable by lowercasing)")
    print(f"      edges lenient pair-regex (json.loads-free):  {regex_ci}   (+{jsonfail_recovery} recoverable if json.loads bypassed)")
    print(f"      edges '<=' line parser (format cross-check): {line}")
    print(f"      well-formed [\"a\",\"b\"] pairs seen: {wf}   out-of-vocab (model reworded): {unknown}")
    if wf == 0 and line == 0:
        hint = "model produced ~no parseable relations -> prompt/model issue, NOT the parser."
    elif line > regex_ci and line > strict:
        hint = "model emitted '<=' lines, not JSON -> FORMAT mismatch (JSON parser can't read it)."
    elif (case_loss + jsonfail_recovery) > max(1, strict) * 0.15:
        hint = "many edges recoverable by a laxer parser -> PARSING is suppressing the score."
    else:
        hint = "few edges recoverable beyond strict -> low score is NOT a parsing artifact."
    print(f"      VERDICT hint: {hint}")
    if debug_path:
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"      wrote raw parse debug -> {debug_path} ({n} responses)")
        except Exception as e:
            print(f"      [!] could not write parse debug: {e}")

def _extract_condensed_with_votes(nodes, client, model_name, chunk_size=1000, max_retries=3,
                                  alt_prompt=False, variant=None, debug_parse=False, debug_path=None):
    """Run the chunked extraction and return (condensed_dag, edge_votes, edge_salience).

    The GRAPH is built from the model's COMMITTED final answer (``content``); the
    reasoning scratchpad is used only as a fallback when the answer parsed no edges,
    so edges the model merely considered and discarded don't enter the taxonomy.

    edge_votes[(parent, child)] is the GENERATION SELF-AGREEMENT signal: of the
    edge's TWO endpoints, how many -- when used as the prompt target -- asserted the
    edge (0, 1, or 2). It is NOT an emission count: repeats or synonym members never
    push it past 2.

    edge_salience[(parent, child)] is a separate, richer signal: how many times the
    edge was asserted ANYWHERE in the responses (committed answer + reasoning). This
    is unbounded; it is reported as a diagnostic feature, not used by the heuristic.
    """
    raw_dag = nx.DiGraph()
    edge_targets = defaultdict(set)   # raw edge -> set of target terms that asserted it (deduped)
    raw_salience = Counter()          # raw edge -> total assertions across answer + reasoning
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())

    # legacy_json and full_json return a JSON [["parent","child"], ...] answer -> JSON parser.
    parse_fn = _parse_relations_json if variant in ("legacy_json", "full_json") else _parse_relations
    dbg_records = [] if debug_parse else None
    dbg_chunks = dbg_empty = 0

    desc_label = f"Our Method [{variant}]" if variant else ("Our Method (alt. Prompt)" if alt_prompt else "Our Method")

    for target_raw in tqdm(primary_nodes, desc=f"  -> [{desc_label}] ChunkSize={chunk_size}", leave=False):
        all_candidates = [t for t in primary_nodes if t != target_raw]

        for i in range(0, len(all_candidates), chunk_size):
            candidates_chunk = all_candidates[i:i + chunk_size]
            prompt = build_prompt(target_raw, candidates_chunk, alt_prompt=alt_prompt, variant=variant)
            content, reasoning = _llm_call(client, model_name, prompt, max_tokens=16328, max_retries=max_retries)
            dbg_chunks += 1
            if not (content or reasoning):
                dbg_empty += 1
                continue

            committed = parse_fn(content, primary_to_full_map)
            deliberated = parse_fn(reasoning, primary_to_full_map)
            # Graph + votes from the COMMITTED answer; fall back to the scratchpad only
            # if the answer itself parsed no edges.
            graph_edges = committed if committed else deliberated
            for (sup, sub) in graph_edges:
                raw_dag.add_edge(sup, sub)
                edge_targets[(sup, sub)].add(target_raw)
            # Salience counts every assertion anywhere in the response.
            for (sup, sub) in committed + deliberated:
                raw_salience[(sup, sub)] += 1

            if debug_parse:
                diag_text = content if content.strip() else reasoning
                dbg_records.append({
                    "target": target_raw,
                    "chunk": i // chunk_size,
                    "content": content[:4000],
                    "reasoning_chars": len(reasoning),
                    "diag": _parse_diagnostics(diag_text, primary_to_full_map),
                })

    if debug_parse:
        _emit_parse_debug(dbg_records, dbg_chunks, dbg_empty, desc_label, debug_path)

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

    edge_salience = Counter()
    for (a, c), cnt in raw_salience.items():
        u, w = node_mapping.get(a), node_mapping.get(c)
        if u is not None and w is not None and u != w and final.has_edge(u, w):
            edge_salience[(u, w)] += cnt
    return final, edge_votes, edge_salience

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

def edge_component_records(G, edge_votes, edge_salience=None):
    """Per-edge values of the suspicion-heuristic components (no ground truth).

    Returns one dict per edge with the raw component values used by
    rank_suspicious_edges -- leverage, neighborhood_agreement, votes -- plus
    salience (a diagnostic-only feature), so they can be paired with an FP/TP label
    downstream to check whether each is informative of errors.

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
                                  edge (generation self-agreement; 0/1/2).
      * salience               -- total assertions of the edge across answer +
                                  reasoning (unbounded; diagnostic only).
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
            "salience": (edge_salience.get((a, c), 0) if edge_salience is not None else 0),
        })
    return rows

def rank_suspicious_edges(G, edge_votes, edge_salience=None,
                          w_leverage=1.0, w_neighborhood=1.0, w_agreement=1.0, w_salience=1.0):
    """Order the edges of G from most to least suspicious for removal.

    Combines GT-free heuristics (each a positive 'suspicion' contribution):
      * leverage     -- (|ancestors(a)|+1)*(|descendants(c)|+1): closure pairs that
                        route through the edge. A wrong high-leverage edge is what
                        damages closure precision the most.
      * neighborhood -- 1 - neighborhood_agreement: an edge a->c whose descendants
                        do NOT independently link back to a is structurally
                        uncorroborated, hence suspicious.
      * agreement    -- edges asserted by only ONE of the two endpoint prompts (a
                        single generation vote) are weakly corroborated -> suspicious.
      * salience     -- LOW total-assertion count correlates with false positives, so a
                        low salience percentile raises suspicion. Only applied when
                        ``edge_salience`` is provided (otherwise this term is 0).

    leverage and salience are percentile-normalised; neighborhood disagreement, the
    single-vote bonus, and the low-salience term are all in [0, 1].
    """
    records = edge_component_records(G, edge_votes, edge_salience)
    if not records:
        return []
    lev_p = _pct_rank([r["leverage"] for r in records])
    use_sal = edge_salience is not None
    sal_p = _pct_rank([r["salience"] for r in records]) if use_sal else [0.0] * len(records)
    scored = []
    for idx, r in enumerate(records):
        low_agreement = 1.0 if r["votes"] <= 1 else 0.0
        disagreement = 1.0 - r["neighborhood_agreement"]
        low_salience = (1.0 - sal_p[idx]) if use_sal else 0.0
        score = (w_leverage * lev_p[idx] + w_neighborhood * disagreement
                 + w_agreement * low_agreement + w_salience * low_salience)
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

def precision_clawback(G, edge_votes, client, model_name, suspicion_candidates=0, max_retries=3,
                       edge_salience=None):
    """Sever the LLM-confirmed-wrong edges among the top `suspicion_candidates` suspects.

    suspicion_candidates == 0 -> no clawback (returns G unchanged).
    """
    if not suspicion_candidates or suspicion_candidates <= 0:
        return G
    ranked = rank_suspicious_edges(G, edge_votes, edge_salience)
    G_out = G.copy()
    for (a, c) in ranked[:suspicion_candidates]:
        if scrutinize_edge(client, model_name, a, c, G, max_retries=max_retries) == "SEVER" and G_out.has_edge(a, c):
            G_out.remove_edge(a, c)
    return G_out

# ----------------------------------------------------------------------------
# Whole-graph restructuring pass
# ----------------------------------------------------------------------------
def _restructure_prompt(G):
    """Prompt that shows the ENTIRE taxonomy at once and asks for an optional rewrite.

    Edges are presented as 'child <= parent' (matching the extraction output format so
    parsing is shared) and the allowed vocabulary is stated explicitly so the model
    re-parents rather than inventing concepts.
    """
    prim = sorted({get_primary_term(n) for n in G.nodes()})
    edges = sorted((get_primary_term(c), get_primary_term(p)) for (p, c) in G.edges())
    edge_lines = "\n".join(f"{c} <= {p}" for (c, p) in edges) or "(no relationships yet)"
    return (
        "You are an expert ontologist reviewing a complete is-a taxonomy that was drafted "
        "automatically. It is written as 'child <= parent', meaning every child is a kind of "
        "the parent.\n\n"
        "Review the taxonomy AS A WHOLE and, where needed, restructure it to follow sound "
        "taxonomy design:\n"
        "- Keep only correct is-a (subclass) relationships; drop any edge that is not one, and "
        "fix the direction if it is reversed.\n"
        "- Re-parent mis-placed concepts under their most specific correct parent.\n"
        "- Remove redundant shortcuts: if 'A <= B' is already implied by a chain A <= ... <= B, "
        "drop the direct edge.\n"
        "- Avoid cycles; prefer a single connected hierarchy.\n"
        "- Use ONLY the concepts listed below -- do not invent, rename, split, or merge concepts.\n\n"
        f"Concepts ({len(prim)}): {', '.join(prim)}\n\n"
        f"Current relationships:\n{edge_lines}\n\n"
        "Output the FINAL restructured taxonomy as 'child <= parent' lines, one per line, using "
        "the concept names exactly as given above. Output only those lines and nothing else.\n\n"
        "Restructured taxonomy:\n"
    )

def restructure_taxonomy(G, client, model_name, max_retries=3, max_tokens=16328):
    """Whole-graph restructuring variant: show the LLM the ENTIRE extracted taxonomy at
    once and let it optionally rewrite the is-a edges to better follow taxonomy
    best-practices (re-parent mis-placed concepts, drop illogical/redundant links, fix
    reversed edges), then rebuild the DAG from its answer.

    The output is constrained to G's existing vocabulary: edges over unknown terms are
    dropped. If the model's answer parses to no edges, G is returned unchanged
    (conservative no-op), so a non-answer never wipes the taxonomy. Cycles in the
    rewrite are broken via enforce_dag. The concept set is preserved; only the edges
    (structure) change.
    """
    if G.number_of_edges() == 0:
        return G
    primary_to_full_map = {get_primary_term(n): n for n in G.nodes()}
    content, reasoning = _llm_call(client, model_name, _restructure_prompt(G),
                                   max_tokens=max_tokens, max_retries=max_retries)
    if not (content or reasoning):
        return G
    # Prefer the committed answer; fall back to the scratchpad only if it parsed nothing.
    edges = _parse_relations(content, primary_to_full_map) or _parse_relations(reasoning, primary_to_full_map)
    if not edges:
        return G
    G_new = nx.DiGraph()
    G_new.add_nodes_from(G.nodes())      # preserve the concept set; only structure changes
    for (parent, child) in edges:
        if parent != child:
            G_new.add_edge(parent, child)
    return enforce_dag(G_new)

def _ranked_restructure_prompt(G, records, ranked, topk):
    """Whole-graph restructure prompt whose edges are ORDERED by clawback suspicion and
    annotated with the per-edge heuristics, so the model concentrates on likely false
    positives rather than restructuring blindly. topk>0 marks only the top-K edges as
    removable ([SUSPECT]) and the rest as [keep] to protect recall."""
    rec_by_edge = {(r["parent"], r["child"]): r for r in records}
    lines = []
    for rank_i, (p, c) in enumerate(ranked):
        r = rec_by_edge.get((p, c))
        if r is None:
            continue
        P, C = get_primary_term(p), get_primary_term(c)
        ann = (f"agree={r['neighborhood_agreement']:.2f} votes={r['votes']} "
               f"salience={r['salience']} leverage={r['leverage']}")
        if topk and rank_i < topk:
            lines.append(f"[SUSPECT] {C} <= {P}    ({ann})")
        elif topk:
            lines.append(f"[keep]    {C} <= {P}")
        else:
            lines.append(f"{C} <= {P}    ({ann})")
    edge_block = "\n".join(lines) or "(no relationships)"

    focus = ("Only the edges marked [SUSPECT] may be removed or re-parented; reproduce every "
             "[keep] edge unchanged.\n" if topk else
             "Concentrate on the most suspicious edges (listed first).\n")
    return (
        "You are an expert ontologist auditing an is-a taxonomy that was drafted "
        "automatically. Each line is an edge 'child <= parent' (every child is a kind of the "
        "parent), annotated with automatic suspicion signals estimated WITHOUT ground truth:\n"
        "  - agree    = neighborhood agreement in [0,1]; LOW means the child's descendants do "
        "not independently trace back to the parent -> the edge is likely WRONG.\n"
        "  - votes    = generation self-agreement (0/1/2); LOW means only weakly corroborated -> suspect.\n"
        "  - salience = how many times the edge was asserted across the model's outputs; "
        "LOW means weakly asserted -> suspect.\n"
        "  - leverage = how many ancestor-descendant pairs route through the edge (blast radius if wrong).\n"
        "Edges are ordered from MOST to LEAST suspicious.\n\n"
        "Correct the taxonomy: remove or re-parent edges that are NOT valid is-a relationships. "
        + focus +
        "Do NOT delete well-supported edges just to simplify -- change only edges that are "
        "genuinely wrong, so correct relations are preserved (protecting recall). Use ONLY the "
        "concept names shown; do not invent, rename, split, or merge concepts.\n\n"
        f"Edges:\n{edge_block}\n\n"
        "Output the FINAL corrected taxonomy as 'child <= parent' lines, one per line, using the "
        "concept names exactly as given above. Output only those lines and nothing else.\n\n"
        "Corrected taxonomy:\n"
    )

def restructure_taxonomy_ranked(G, edge_votes, client, model_name, edge_salience=None,
                                topk=0, max_retries=3, max_tokens=16328):
    """Heuristic-guided whole-graph restructuring: show the LLM every edge, but ranked by
    the clawback suspicion score (leverage + low neighborhood-agreement + low self-agreement)
    and annotated with each edge's heuristics (incl. salience), so it focuses its edits on
    likely false positives instead of restructuring indiscriminately.

    Same safety as restructure_taxonomy: vocabulary-bound, a conservative no-op if the answer
    parses no edges, cycles broken. topk>0 restricts the removable set to the top-K suspects.
    """
    if G.number_of_edges() == 0:
        return G
    records = edge_component_records(G, edge_votes, edge_salience)
    ranked = rank_suspicious_edges(G, edge_votes, edge_salience)
    primary_to_full_map = {get_primary_term(n): n for n in G.nodes()}
    content, reasoning = _llm_call(client, model_name, _ranked_restructure_prompt(G, records, ranked, topk),
                                   max_tokens=max_tokens, max_retries=max_retries)
    if not (content or reasoning):
        return G
    edges = _parse_relations(content, primary_to_full_map) or _parse_relations(reasoning, primary_to_full_map)
    if not edges:
        return G
    G_new = nx.DiGraph()
    G_new.add_nodes_from(G.nodes())
    for (parent, child) in edges:
        if parent != child:
            G_new.add_edge(parent, child)
    return enforce_dag(G_new)

def method_our_approach(nodes, client, model_name, chunk_size=1000, max_retries=3,
                        alt_prompt=False, suspicion_candidates=0, variant=None, restructure=False,
                        restructure_ranked=False, restructure_topk=0,
                        debug_parse=False, debug_path=None):
    final_dag, edge_votes, edge_salience = _extract_condensed_with_votes(
        nodes, client, model_name, chunk_size=chunk_size, max_retries=max_retries,
        alt_prompt=alt_prompt, variant=variant, debug_parse=debug_parse, debug_path=debug_path)
    if restructure_ranked:
        final_dag = restructure_taxonomy_ranked(final_dag, edge_votes, client, model_name,
                                                edge_salience=edge_salience, topk=restructure_topk,
                                                max_retries=max_retries)
    elif restructure:
        final_dag = restructure_taxonomy(final_dag, client, model_name, max_retries=max_retries)
    return precision_clawback(final_dag, edge_votes, client, model_name,
                              suspicion_candidates=suspicion_candidates, max_retries=max_retries,
                              edge_salience=edge_salience)

def method_our_approach_sweep(nodes, client, model_name, suspicion_candidates_list,
                              chunk_size=1000, max_retries=3, alt_prompt=False, variant=None,
                              restructure=False, restructure_ranked=False, restructure_topk=0,
                              debug_parse=False, debug_path=None):
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
    final_dag, edge_votes, edge_salience = _extract_condensed_with_votes(
        nodes, client, model_name, chunk_size=chunk_size, max_retries=max_retries,
        alt_prompt=alt_prompt, variant=variant, debug_parse=debug_parse, debug_path=debug_path)
    if restructure_ranked:
        # Same whole-graph rewrite, but the edges are ranked/annotated by the suspicion
        # heuristics (leverage, neighborhood agreement, self-agreement, salience) so the
        # model focuses on likely false positives. Uses the PRE-rewrite votes/salience.
        final_dag = restructure_taxonomy_ranked(final_dag, edge_votes, client, model_name,
                                                edge_salience=edge_salience, topk=restructure_topk,
                                                max_retries=max_retries)
    elif restructure:
        # Whole-graph rewrite REPLACES the extracted DAG before ranking/clawback. Votes
        # and salience (a clawback signal) are kept as-is and are simply not meaningful
        # for any edge the rewrite introduces; restructure is normally run clawback-off.
        final_dag = restructure_taxonomy(final_dag, client, model_name, max_retries=max_retries)
    ks = sorted({int(k) for k in suspicion_candidates_list})
    ranked = rank_suspicious_edges(final_dag, edge_votes, edge_salience)

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
    return out, edge_component_records(final_dag, edge_votes, edge_salience)
