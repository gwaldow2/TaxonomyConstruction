"""Build a taxonomy against a REAL OpenAI-compatible endpoint (hosted OpenAI, vLLM, ...).

Prereqs:
    pip install -e '.[openai]'
    # have a server running, e.g. an OpenAI-compatible endpoint at base_url below
    python examples/with_openai.py
"""

from taxochunk import TaxonomyBuilder


def main():
    builder = TaxonomyBuilder.from_endpoint(
        base_url="http://localhost:8000/v1",   # or https://api.openai.com/v1
        api_key="EMPTY",                        # any non-empty string for local servers
        model="openai/gpt-oss-120b",
        chunk_size=1000,                        # candidate terms per prompt
        # alt_prompt=True,                      # ablation: direct parent/child phrasing
    )

    terms = [
        "food", "fruit", "vegetable", "apple", "banana",
        "granny smith", "fuji", "carrot",
    ]

    G = builder.build_and_save(terms, "taxonomy.graphml", show_progress=True)
    print(f"Built {G.number_of_nodes()} nodes / {G.number_of_edges()} edges -> taxonomy.graphml")


if __name__ == "__main__":
    main()
