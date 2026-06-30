# Precision Clawback

After Our Method extracts a taxonomy, a small number of wrong edges can
disproportionately hurt **closure** precision: one incorrect edge high in the
hierarchy manufactures many false ancestor→descendant pairs. Precision clawback
re-verifies a handful of the most suspicious edges with the LLM and severs the
ones it confirms wrong — recovering precision at minimal recall cost.

It is controlled by a single hyperparameter, `suspicion_candidates`
(default `0` = off): the number of top-ranked suspicious edges the LLM scrutinizes.
Larger values scrutinize more edges, so any given suspect is more likely to be
re-examined. Implemented in [`our_method.py`](../our_method.py).

## Pipeline

### Step 0 — Extraction with self-agreement votes
Run the normal chunked extraction. Each candidate edge has up to **two**
independent chances to be produced — once with each endpoint as the prompt's
target. Record `edge_votes[(parent, child)]` = how many of those prompts asserted
the edge (a `Counter` instead of a `set`). Synonyms are then condensed and the DAG
enforced, carrying the vote counts onto the condensed edges. The vote count is the
**generation self-agreement** signal.

### Step 1 — Score every edge for suspicion (no ground truth required)
For each edge `a → c` in the predicted DAG, compute three signals:

- **Leverage** (impact) — `leverage = (|ancestors(a)| + 1) · (|descendants(c)| + 1)`:
  the number of closure (ancestor→descendant) pairs that route through the edge.
  A wrong high-leverage edge damages closure precision the most.
- **Neighborhood agreement** (structural corroboration) —
  `neighborhood_agreement(a → c) = |{d ∈ descendants(c) : raw edge a → d exists}| / |descendants(c)|`.
  If `a → c` is a real ancestor relation, then every descendant of `c` is also a
  descendant of `a`; and because the method emits ancestor edges directly, a
  well-supported edge has many of those descendants link **straight back to `a`**.
  If few/none do, the edge is structurally uncorroborated, hence suspicious.
  *Example:* an edge from a node to its parent whose single child does not have a
  raw edge to the grandparent scores `0/1 = 0`. Leaf children (no descendants)
  score `1.0` (nothing to corroborate). **Low** agreement ⇒ suspicious, so the
  score uses `1 − neighborhood_agreement`.
- **Self-agreement** (uncertainty) — edges asserted by only **one** of the two
  endpoint prompts (`votes ≤ 1`) are weakly corroborated, hence suspicious.

Leverage is percentile-normalized across edges; the neighborhood-disagreement and
single-vote terms are already in `[0, 1]`:

```
suspicion(a → c) = w_leverage     · pct_rank(leverage)
                 + w_neighborhood · (1 − neighborhood_agreement)
                 + w_agreement    · (1 if votes ≤ 1 else 0)
```

with default weights `w_leverage = w_neighborhood = w_agreement = 1.0`. Edges are
ranked by descending suspicion.

### Step 2 — LLM scrutiny of the top-K
Take the top `suspicion_candidates` (= K) edges. For each, ask the LLM to briefly
reason about that one is-a relationship and decide KEEP or SEVER (prompt below).
The verdict is the **last** KEEP/SEVER token in the reply; the default is **KEEP**
— the method never removes an edge on an ambiguous answer.

### Step 3 — Sever
Remove every edge the LLM marked SEVER. Removing edges cannot create cycles, so
the result is still a DAG. `suspicion_candidates = 0` skips Steps 1–3 and returns
the unmodified graph (i.e. Our Method without clawback).

### Sweep efficiency
For an F1-vs-K curve, extract and rank **once**, compute each edge's verdict
**once** (for the largest K), then build the graph at every K by applying the
cached verdicts to the top-K prefix. A full sweep therefore costs one extraction
plus `max(K)` audit calls — not a re-extraction per K
(`method_our_approach_sweep`).

## Prompts

### Extraction prompt (per target, per candidate chunk)
`{target}` is the target term; the candidate chunk is listed one per line. This is
the **default** ("our approach") wording; the alternate prompt is identical except
for the two relationship bullets (the linguistic ablation — it instead asks
"`C` is a parent / child concept of `{target}`").

```
You are identifying hierarchical relationships for the target entity: '{target}'.
Below is a list of candidate entities. Identify any subclass or superclass relationships between the target and the candidates.
- If every entity labeled with '{target}' could logically also be labeled with a candidate 'C', output '{target} <= C'
- If every entity labeled with a candidate 'C' could logically also be labeled with '{target}', output 'C <= {target}'
ONLY output relationships involving '{target}'. Do NOT output relationships between the candidates themselves. Output each relationship on a new line. If there are no relationships, output 'none'.

Example: 'anucleate cell' <= 'cell'
Candidates:
- {candidate_1}
- {candidate_2}
- ...

Relationships:
```

Each `child <= parent` line is parsed into an edge `parent → child`; both terms
must be in the vocabulary.

### Scrutiny prompt (per suspicious edge, Step 2)
`{parent}` and `{child}` are the **primary surface forms** of the edge's endpoints.

```
You are auditing one is-a edge in a taxonomy.
Proposed relationship: '{child}' is a kind of '{parent}'.
Briefly reason (1-2 sentences) about whether every '{child}' is necessarily a type of '{parent}'.
Then, on a new FINAL line, output exactly one word: KEEP if the relationship is valid, or SEVER if it is incorrect and should be removed.
```

The reply is scanned for KEEP / SEVER; the last such token decides the edge's
fate, defaulting to KEEP when neither is found.

## Hyperparameters

| name | default | meaning |
|------|---------|---------|
| `suspicion_candidates` | `0` | number of top-suspicious edges the LLM scrutinizes (`0` = clawback off). In the benchmark, `--suspicion_candidates 0 5 10 25 …` sweeps several values. |
| `w_leverage`, `w_neighborhood`, `w_agreement` | `1.0` | weights on the three suspicion signals in `rank_suspicious_edges`. |

## Notes / limitations

- Severing a **redundant** edge (one where `a` still reaches `c` via another path)
  does not change the closure, so it cannot recover closure precision. The
  leverage term biases toward impactful edges, but candidates are not yet
  hard-filtered to non-redundant (bridge) edges; doing so sharpens the precision
  gain and is the natural next refinement.
- The LLM is the final arbiter, so a heuristic false positive only costs one audit
  call — it does not, by itself, remove a correct edge.
