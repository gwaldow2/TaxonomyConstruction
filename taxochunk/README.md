# taxochunk

Chunked LLM taxonomy construction — turn a **flat list of terms** into an **is-a taxonomy**
(a parent→child DAG) by asking an LLM, chunk by chunk, which terms are parents/children of
each target. This packages the "Our Method" approach from the
[TaxonomyConstruction](https://github.com/gwaldow2/TaxonomyConstruction) benchmark.

## Why chunked?

For each target term the candidate vocabulary is split into chunks that fit the model's
context window, so the method scales to large vocabularies where a single-shot prompt
collapses. Smaller chunks are more focused (higher recall on near-distance relations) at the
cost of more calls. Raw output is then **synonym-clustered** (mutually is-a terms merged) and
reduced to a **DAG** (cycles broken).

## Install

```bash
pip install taxochunk            # core (networkx only)
pip install 'taxochunk[openai]'  # + OpenAI-compatible client and CLI
```

## Library

The easiest path — instantiate a builder (it spins up the OpenAI client for you),
then build and save:

```python
from taxochunk import TaxonomyBuilder

builder = TaxonomyBuilder.from_endpoint(
    base_url="http://localhost:8000/v1",   # local vLLM / hosted OpenAI
    api_key="EMPTY",
    model="openai/gpt-oss-120b",
    chunk_size=1000,        # candidates per prompt
    alt_prompt=False,       # True -> JSON [["parent","child"], ...] prompt
)
G = builder.build(["food", "fruit", "apple", "granny smith"])   # networkx.DiGraph
builder.build_and_save(["food", "fruit", "apple"], "taxonomy.graphml")
```

Already have an `openai` client? Pass it in, or use the `openai_client` helper:

```python
from taxochunk import TaxonomyBuilder, openai_client, build_taxonomy

client = openai_client("http://localhost:8000/v1", "EMPTY")
builder = TaxonomyBuilder(client=client, model="openai/gpt-oss-120b")
# ...or the plain function:
G = build_taxonomy(["food", "fruit", "apple"], client=client, model="openai/gpt-oss-120b")
```

Terms may carry synonyms in lemma format, e.g. `"apple (apple, malus pumila)"`; mutually
is-a terms discovered by the model are merged into one cluster node.

## Output formats

`build_and_save(...)` / `save_taxonomy(G, path, fmt=None)` write the taxonomy in
broadly-interoperable formats. The default is **GraphML** — read directly by Gephi,
Cytoscape, yEd, igraph, networkx and most graph tooling, and it preserves attributes.

```python
from taxochunk import save_taxonomy
save_taxonomy(G, "tax.graphml")     # default; great for Gephi / Cytoscape
save_taxonomy(G, "tax.json")        # node-link JSON for web / d3
save_taxonomy(G, "tax.gexf")        # Gephi native
save_taxonomy(G, "tax.dot")         # Graphviz
save_taxonomy(G, "tax.tsv")         # parent<TAB>child edge list
```

Format is inferred from the extension (or pass `fmt=...`). Supported: `graphml`,
`gexf`, `json`, `dot`, `tsv`, `csv`.

### Bring your own LLM (and test without one)

`build_taxonomy` accepts any `respond: prompt -> text` callable via `respond=...`, so you can
use a non-OpenAI backend or a stub:

```python
def respond(prompt):           # deterministic stub
    return "fruit <= food\napple <= fruit"

G = build_taxonomy(["food", "fruit", "apple"], respond=respond)
```

## CLI

```bash
taxochunk terms.txt --model openai/gpt-oss-120b --base-url http://localhost:8000/v1 --out tax.graphml
taxochunk terms.txt --model openai/gpt-oss-120b --out tax.json --format json
# or stream a parent<TAB>child edge list:
cat terms.txt | taxochunk --model openai/gpt-oss-120b
```

## Key options

| arg | meaning |
|-----|---------|
| `chunk_size` | candidate terms per prompt (lower = more focused, more calls) |
| `alt_prompt` | JSON pair prompt instead of `child <= parent` lines |
| `max_retries` / `retry_delay` | per-chunk retry budget on empty/malformed/API errors |
| `keep_isolated` | keep terms that get no relations (default True; set False for benchmark parity) |
| `show_progress` | tqdm bar over targets (needs `tqdm`) |

## License

MIT
