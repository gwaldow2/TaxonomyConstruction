"""Quickstart: build a taxonomy with taxochunk -- using a DUMMY LLM (no server needed).

This runs with NO API key and NO model server. Instead of calling a real LLM, we
pass a `respond(prompt) -> text` stub that returns canned "child <= parent" lines.
That is exactly the seam a real model plugs into, so this is a faithful end-to-end
demo of the library that you can run right now:

    pip install -e .          # from the repo root (the folder with pyproject.toml)
    python examples/quickstart.py
"""

from taxochunk import build_taxonomy, save_taxonomy


def dummy_respond(prompt: str) -> str:
    """Stand-in for an LLM. A real model would read `prompt`; we ignore it and
    return a fixed set of relationships so the example is deterministic.

    Format is the library's default: one `child <= parent` per line.
    """
    return (
        "fruit <= food\n"
        "vegetable <= food\n"
        "apple <= fruit\n"
        "banana <= fruit\n"
        "granny smith <= apple\n"   # a deeper level
        "carrot <= vegetable\n"
        "fuji <= apple\n"
    )


def main():
    terms = [
        "food", "fruit", "vegetable", "apple", "banana",
        "granny smith", "fuji", "carrot",
    ]

    # Build the taxonomy. `respond=` injects the dummy LLM; drop it and pass
    # `client=`/`model=` (or use TaxonomyBuilder) to use a real model instead.
    G = build_taxonomy(terms, respond=dummy_respond)

    print(f"Built a taxonomy with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")
    print("Edges (parent -> child):")
    for parent, child in sorted(G.edges()):
        print(f"  {parent} -> {child}")

    # Export in broadly-interoperable formats.
    save_taxonomy(G, "dummy_taxonomy.graphml")   # open in Gephi / Cytoscape / networkx
    save_taxonomy(G, "dummy_taxonomy.json")      # node-link JSON for web / d3
    print("\nWrote dummy_taxonomy.graphml and dummy_taxonomy.json")


if __name__ == "__main__":
    main()
