"""Write a taxonomy DiGraph to disk in broadly-interoperable formats.

Default is **GraphML** -- the most widely supported graph interchange format:
read directly by Gephi, Cytoscape, yEd, igraph, networkx and most graph tools,
and it preserves node/edge attributes. Other formats are offered for specific
needs (GEXF for Gephi, JSON node-link for web/d3, DOT for Graphviz, TSV/CSV edge
lists for spreadsheets and quick diffing).
"""

import os
import csv
import json

import networkx as nx

__all__ = ["save_taxonomy", "SUPPORTED_FORMATS"]

SUPPORTED_FORMATS = ("graphml", "gexf", "json", "dot", "tsv", "csv")

_EXT_TO_FMT = {
    ".graphml": "graphml", ".gexf": "gexf", ".json": "json",
    ".dot": "dot", ".gv": "dot", ".tsv": "tsv", ".csv": "csv",
}


def _infer_format(path):
    return _EXT_TO_FMT.get(os.path.splitext(path)[1].lower(), "graphml")


def _write_dot(G, path):
    def esc(s):
        return str(s).replace("\\", "\\\\").replace('"', '\\"')
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph taxonomy {\n  rankdir=TB;\n  node [shape=box];\n")
        for n in G.nodes():
            f.write(f'  "{esc(n)}";\n')
        for u, v in G.edges():
            f.write(f'  "{esc(u)}" -> "{esc(v)}";\n')
        f.write("}\n")


def _write_delimited(G, path, delimiter):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(["parent", "child"])
        for u, v in G.edges():
            w.writerow([u, v])


def _write_json(G, path):
    from networkx.readwrite import json_graph
    try:                                   # networkx >= 3.4 wants an explicit edges kw
        data = json_graph.node_link_data(G, edges="edges")
    except TypeError:                      # older networkx
        data = json_graph.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_taxonomy(G, path, fmt=None):
    """Write taxonomy ``G`` to ``path``.

    Parameters
    ----------
    G : networkx.DiGraph
        The parent -> child taxonomy.
    path : str
        Output file path. If ``fmt`` is None the format is inferred from the
        extension (``.graphml`` default if unrecognised).
    fmt : str, optional
        One of :data:`SUPPORTED_FORMATS`. Overrides the extension.

    Returns
    -------
    str
        The ``path`` written (handy for chaining/logging).
    """
    fmt = (fmt or _infer_format(path)).lower()
    if fmt == "graphml":
        nx.write_graphml(G, path)
    elif fmt == "gexf":
        nx.write_gexf(G, path)
    elif fmt == "json":
        _write_json(G, path)
    elif fmt == "dot":
        _write_dot(G, path)
    elif fmt == "tsv":
        _write_delimited(G, path, "\t")
    elif fmt == "csv":
        _write_delimited(G, path, ",")
    else:
        raise ValueError(f"Unsupported format {fmt!r}; choose from {SUPPORTED_FORMATS}.")
    return path
