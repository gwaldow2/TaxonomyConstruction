"""Build synset-formatted GT graphs via the repo loaders (use_synsets=True) and
report duplicate synonym strings shared by DIFFERENT synset nodes."""
import re
import sys
from collections import defaultdict

import os
sys.path.insert(0, os.environ.get("TAXO_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

LEMMA = re.compile(r"^([^(]+)\s*\((.*)\)$")

def terms_of(node):
    m = LEMMA.match(str(node).strip().lower())
    if not m:
        return None
    primary = m.group(1).strip()
    syns = [t.strip() for t in m.group(2).split(",") if t.strip()]
    if syns and syns[0] == primary:
        return set(syns)
    return None

def check(name, G):
    node_terms = {n: t for n in G.nodes() if (t := terms_of(n))}
    owner = defaultdict(set)
    for n in G.nodes():
        for t in (node_terms.get(n) or {str(n).strip().lower()}):
            owner[t].add(n)
    dupes = {t: nodes for t, nodes in owner.items() if len(nodes) > 1}
    print(f"\n=== {name}: {G.number_of_nodes()} nodes in eval universe, "
          f"{len(node_terms)} synset (multi-term) nodes, {len(dupes)} duplicate strings ===")
    for t, nodes in sorted(dupes.items()):
        print(f"  '{t}' in {len(nodes)} nodes: " + " | ".join(sorted(str(x) for x in nodes)))
    return len(dupes)

from data_manager import get_wordnet_food_graph, get_cell_ontology_graph

G, _ = get_wordnet_food_graph(use_synsets=True)
check("WordNetFood (use_synsets=True)", G)

G, _ = get_cell_ontology_graph(use_synsets=True)
check("CellOntology (use_synsets=True)", G)
