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

```python
from taxochunk import build_taxonomy
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # local vLLM, etc.
G = build_taxonomy(
    ["food", "fruit", "apple", "granny smith"],
    client=client, model="openai/gpt-oss-120b",
    chunk_size=1000,        # candidates per prompt
    alt_prompt=False,       # True -> JSON [["parent","child"], ...] prompt
)
print(list(G.edges()))      # networkx.DiGraph of parent -> child
```

Terms may carry synonyms in lemma format, e.g. `"apple (apple, malus pumila)"`; mutually
is-a terms discovered by the model are merged into one cluster node.

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
# or stream an edge list:
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
