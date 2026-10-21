"""Rebuild the LLMs4OL-derived ground truths from their FULL canonical sources.

Why: the LLMs4OL TaskC files are train/test splits of each ontology, and only the
train half is published. Scoring closed-world against that half mislabels real
edges as false positives (verified for SchemaOrg: the train file carries 670 of
the live vocabulary's 993 direct subClassOf edges; 91% of the missing ones involve
a held-out test type). These builders go to the ontologies themselves.

Provenance design:
- Each source file is downloaded ONCE into taxonomy_data/full_sources/ and pinned:
  the file itself is kept, and taxonomy_data/full_sources/sources_manifest.csv
  records url, fetch timestamp, sha256, size, and parsed node/edge counts. Reruns
  reuse the pinned file (pass --refetch to force a new download, which appends a
  new manifest row rather than rewriting history).
- Benchmarks are written under NEW dataset names (SchemaOrgV2, POV2, OBIV2,
  MatOntoV2); the legacy challenge-derived benchmarks stay untouched so every
  existing result remains reproducible. The V2 graphml files carry the source
  url/sha/fetch-date as graph attributes via save_benchmark_graph(meta=...).
- Downstream provenance is unchanged: dataset names flow into every results row,
  diagnostics file, and saved prediction graph; scale variants come from
  scale_benchmarks.py against the frozen V2 FULL graphs.

Parsing decisions (recorded here deliberately):
- SchemaOrg: every rdfs:Class in the current release file, subClassOf edges with
  both endpoints in the schema.org namespace; superseded classes are KEPT (they
  are part of the published vocabulary). DataTypes are classes too and are kept.
- PO / OBI: obo releases, is_a edges only, matching the CellOntology treatment.
  Only terms DEFINED in the file (a stanza with a name) become nodes: a bare
  is_a reference to a term the file never defines (e.g. po.obo's
  "is_a: UBERON:0001062" lines) would otherwise surface as an opaque ID that
  no model can read, and the legacy challenge splits excluded those too.
  obi.obo defines its imported upper-level classes (BFO, UBERON etc.) with
  names, so those imports are kept, since OBI's own classification routes
  through them.
- MatOnto: the GitHub release Turtle (rdfs:subClassOf between named classes,
  rdfs:label preferred over local names). This is the weakest source of the four
  (project dormant; internal versionInfo 2.01); its manifest row is the citation.
  Source equivalence verified 2026-10-21: the pinned file is sha256-identical to
  the MatPortal registry artifact (matportal.org/ontologies/MATONTO, v2.0,
  uploaded 2021-01-25 — cite this entry in the paper), and OntoLearner's
  MatOnto.owl (github.com/EngyNasr/MSE-Benchmark) is the same content
  re-serialized as RDF/XML: 853/853 identical subClassOf edges, 1307 classes.
  No newer canonical release exists.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime

import networkx as nx

import data_manager as dm
from data_manager import (clean_term, enforce_dag, get_closed_subgraph,
                          get_rigorous_80_20_split, save_benchmark_graph)

SOURCES = {
    "SchemaOrgV2": {"url": "https://schema.org/version/latest/schemaorg-current-https.jsonld",
                    "kind": "jsonld", "filename": "schemaorg-current-https.jsonld"},
    "POV2":  {"url": "https://purl.obolibrary.org/obo/po.obo", "kind": "obo", "filename": "po.obo"},
    "OBIV2": {"url": "https://purl.obolibrary.org/obo/obi.obo", "kind": "obo", "filename": "obi.obo"},
    "MatOntoV2": {"url": "https://raw.githubusercontent.com/inovexcorp/MatOnto-Ontologies/master/matonto-release.ttl",
                  "kind": "ttl", "filename": "matonto-release.ttl"},
}

MANIFEST_FIELDS = ["dataset", "url", "fetched", "sha256", "bytes",
                   "source_nodes", "source_edges", "note"]

# Documented source errata, excluded at parse time and reported loudly.
# Each entry identifies ONE class-expression axiom by (class label, frozenset of
# union-member labels). MatOnto's GroupTwoElement carries two equivalence
# unions: the correct group-2 membership AND group 3's membership (Sc, Y,
# lanthanides, actinides) - an authoring error (GroupThreeElement itself has no
# union), which would otherwise entail e.g. actinium is-a alkaline earth metal.
SOURCE_ERRATA = {
    "ttl": [("group 2 element",
             frozenset({"actinide", "lanthanide", "scandium", "yttrium"}))],
}


def sources_dir():
    d = os.path.join(dm.DATA_DIR, "full_sources")
    os.makedirs(d, exist_ok=True)
    return d


def fetch_and_pin(name, spec, refetch=False):
    path = os.path.join(sources_dir(), spec["filename"])
    note = "cached"
    if refetch or not os.path.exists(path):
        print(f"  [fetch] {spec['url']}")
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "taxonomy-benchmark"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
        note = "downloaded"
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return path, sha, note


# ---------------- parsers ----------------

def _warn_collisions(id_to_name, label):
    """Distinct source IDs that clean to the same name merge into one node; say so."""
    by_name = {}
    for i, n in id_to_name.items():
        by_name.setdefault(n, []).append(i)
    merged = {n: ids for n, ids in by_name.items() if len(ids) > 1}
    for n, ids in sorted(merged.items()):
        print(f"  [collision] {label}: {sorted(map(str, ids))} -> '{n}'")
    if merged:
        print(f"  [collision] {label}: {len(merged)} name(s) cover multiple source IDs")


def parse_schemaorg_jsonld(path):
    data = json.load(open(path, encoding="utf-8"))
    G = nx.DiGraph()
    ids = {}
    def local(x):
        # clean_term lowercases, so CamelCase ids collapse to the squashed form the
        # legacy challenge-derived benchmarks used ("WarrantyScope" -> "warrantyscope").
        return clean_term(str(x).split("/")[-1].replace("schema:", ""))
    for node in data["@graph"]:
        t = node.get("@type")
        t = t if isinstance(t, list) else [t]
        if "rdfs:Class" not in t:
            continue
        nid = str(node.get("@id", ""))
        if not (nid.startswith("schema:") or "schema.org" in nid):
            continue
        child = local(nid)
        ids[nid] = child
        sup = node.get("rdfs:subClassOf", [])
        sup = [sup] if isinstance(sup, dict) else sup
        for s in sup:
            sid = str(s.get("@id", ""))
            if sid.startswith("schema:") or "schema.org" in sid:
                G.add_edge(local(sid), child)
        # schema.org puns its DataType subtree: Text, Number, Date... are typed
        # both rdfs:Class and schema:DataType, with no subClassOf. The legacy
        # challenge GT treated that dual typing as is_a (datatype -> time etc.);
        # keep parity. Enumeration members (typed schema:X only, no rdfs:Class)
        # stay excluded.
        for tt in t:
            tt = str(tt)
            if tt != "rdfs:Class" and (tt.startswith("schema:") or "schema.org" in tt):
                G.add_edge(local(tt), child)
    _warn_collisions(ids, "schemaorg")
    return G


def parse_obo(path):
    import obonet
    obo = obonet.read_obo(path, ignore_obsolete=True)
    G = nx.DiGraph()
    # Only terms the file DEFINES (a stanza with a name) become nodes. A bare
    # is_a reference to an external term (e.g. "is_a: UBERON:0001062" in po.obo)
    # has no name here and would surface as an opaque ID no model can read.
    nm = {u: clean_term(d["name"]) for u, d in obo.nodes(data=True) if d.get("name")}
    _warn_collisions(nm, "obo")
    dropped = 0
    for u, v, key in obo.edges(keys=True):
        if key == "is_a":
            parent, child = nm.get(v), nm.get(u)
            if parent and child and parent != child:
                G.add_edge(parent, child)
            elif (v not in nm or u not in nm):
                dropped += 1
    if dropped:
        print(f"  [excised] {dropped} is_a edge(s) referencing terms the file does not define")
    return G


def parse_ttl(path):
    from rdflib import Graph as RGraph
    from rdflib.namespace import OWL, RDFS
    from rdflib.term import BNode, URIRef
    rg = RGraph()
    rg.parse(path, format="turtle")
    labels = {}
    for s, _, o in rg.triples((None, RDFS.label, None)):
        labels[s] = clean_term(str(o))
    def name(x):
        return labels.get(x) or clean_term(str(x).split("#")[-1].split("/")[-1])
    _warn_collisions(labels, "ttl")
    G = nx.DiGraph()
    for s, _, o in rg.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            parent, child = name(o), name(s)
            if parent and child and parent != child:
                G.add_edge(parent, child)

    # OWL class-expression axioms entail subsumption that plain subClassOf
    # misses. MatOnto classifies every chemical element into its period/group/
    # block/series ONLY via "C equivalentClass [ unionOf (...) ]": C = M1 or
    # ... or Mk entails each Mi is-a C (likewise for a union bnode under
    # subClassOf). Dually, C equivalentClass/subClassOf [ intersectionOf
    # (... A ...) ] entails C is-a A for each NAMED component A.
    def union_members(expr, seen):
        if expr in seen:
            return
        seen.add(expr)
        for lst in list(rg.objects(expr, OWL.unionOf)) + list(rg.objects(expr, OWL.oneOf)):
            for m in rg.items(lst):
                if isinstance(m, URIRef):
                    yield m
                elif isinstance(m, BNode):
                    yield from union_members(m, seen)

    def intersection_components(expr, seen=None):
        # recurse through nested intersections (MatOnto: ester group =
        # ((carbon atom and ...) and ...)); never through unions, whose
        # components are not entailed superclasses
        seen = set() if seen is None else seen
        if expr in seen:
            return
        seen.add(expr)
        for lst in rg.objects(expr, OWL.intersectionOf):
            for m in rg.items(lst):
                if isinstance(m, URIRef):
                    yield m
                elif isinstance(m, BNode):
                    yield from intersection_components(m, seen)

    equiv, expr_sub_named, named_sub_expr = [], [], []
    for a, b in rg.subject_objects(OWL.equivalentClass):
        if isinstance(a, URIRef) and isinstance(b, BNode):
            equiv.append((a, b))
        elif isinstance(a, BNode) and isinstance(b, URIRef):
            equiv.append((b, a))
    for s, _, o in rg.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, BNode) and isinstance(o, URIRef):
            expr_sub_named.append((o, s))
        elif isinstance(s, URIRef) and isinstance(o, BNode):
            named_sub_expr.append((s, o))

    added = dropped_errata = 0
    errata = SOURCE_ERRATA.get("ttl", [])

    def add_edge(parent, child):
        nonlocal added
        if parent and child and parent != child and not G.has_edge(parent, child):
            G.add_edge(parent, child)
            added += 1

    # C = union(...)  or  union(...) subClassOf C   =>   each member is-a C
    for c, expr in equiv + expr_sub_named:
        mem = [name(m) for m in union_members(expr, set())]
        if (name(c), frozenset(mem)) in errata:
            dropped_errata += 1
            print(f"  [erratum] skipped union axiom on '{name(c)}': {sorted(mem)}")
            continue
        for child in mem:
            add_edge(name(c), child)
    # C = intersection(... A ...)  or  C subClassOf intersection(... A ...)
    #   =>  C is-a A   (the reverse shape, intersection subClassOf C, entails
    #       nothing between named classes and is deliberately not extracted)
    for c, expr in equiv + named_sub_expr:
        for a in intersection_components(expr):
            add_edge(name(a), name(c))
    if added:
        print(f"  [expressions] {added} edge(s) entailed by union/oneOf/intersection axioms"
              + (f" ({dropped_errata} erratum axiom(s) skipped)" if dropped_errata else ""))

    # named<->named equivalentClass pairs are two names for one class (MatOnto:
    # "alkali metal = group 1 element" and 8 more). Merge them into synset-style
    # nodes, "head (head, alias)", the same convention the legacy synset
    # benchmarks use, so set-overlap matching accepts either name.
    eq = nx.Graph()
    for a, b in rg.subject_objects(OWL.equivalentClass):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            na, nb = name(a), name(b)
            if na and nb and na != nb:
                eq.add_edge(na, nb)
    mapping = {}
    for comp in nx.connected_components(eq):
        names = sorted(comp)
        node = f"{names[0]} ({', '.join(names)})"
        for n in names:
            mapping[n] = node
    if mapping:
        G = nx.relabel_nodes(G, mapping)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        print(f"  [equivalences] merged {len(mapping)} names into "
              f"{len(set(mapping.values()))} synset node(s)")
    return G


PARSERS = {"jsonld": parse_schemaorg_jsonld, "obo": parse_obo, "ttl": parse_ttl}


def upsert_manifest(row):
    path = os.path.join(sources_dir(), "sources_manifest.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def build_one(name, spec, target_nodes=100, refetch=False, force=False):
    full_path = os.path.join(dm.BENCHMARK_DIR, f"{name}_FULL.graphml")
    if os.path.exists(full_path) and not force:
        print(f"  [skip] {name}: benchmarks exist (use --force to regenerate)")
        return None
    src_path, sha, note = fetch_and_pin(name, spec, refetch=refetch)
    G_src = PARSERS[spec["kind"]](src_path)
    fetched = datetime.now().isoformat(timespec="seconds")
    print(f"  [src]  {name}: {G_src.number_of_nodes()} nodes, {G_src.number_of_edges()} edges "
          f"({note}, sha256={sha[:12]}...)")
    upsert_manifest({"dataset": name, "url": spec["url"], "fetched": fetched, "sha256": sha,
                     "bytes": os.path.getsize(src_path),
                     "source_nodes": G_src.number_of_nodes(),
                     "source_edges": G_src.number_of_edges(), "note": note})

    meta = {"source_url": spec["url"], "source_sha256": sha, "source_fetched": fetched,
            "generator": "full_source_benchmarks.py",
            "source_nodes": G_src.number_of_nodes(), "source_edges": G_src.number_of_edges()}

    G_test, train_pairs = get_rigorous_80_20_split(G_src)
    G_full = enforce_dag(G_test.copy())
    save_benchmark_graph(G_full, name, scale="FULL", train_pairs=train_pairs, meta=meta)

    G_sub = enforce_dag(get_closed_subgraph(G_test, target_nodes=target_nodes))
    save_benchmark_graph(G_sub, name, scale="SUB",
                         train_pairs=(train_pairs or [])[:target_nodes // 2],
                         meta={**meta, "target_nodes": target_nodes, "seed": 42})
    print(f"  [ok]   {name}: FULL={G_full.number_of_nodes()} nodes, "
          f"SUB={G_sub.number_of_nodes()} nodes")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES),
                    help=f"Which to build (default: all of {list(SOURCES)})")
    ap.add_argument("--target_nodes", type=int, default=100)
    ap.add_argument("--refetch", action="store_true",
                    help="Re-download sources even if a pinned copy exists")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate benchmarks whose files already exist")
    args = ap.parse_args()
    built = []
    for name in args.datasets:
        if name not in SOURCES:
            raise SystemExit(f"[!] unknown dataset {name}; choose from {list(SOURCES)}")
        print(f"\n--- {name} ---")
        if build_one(name, SOURCES[name], args.target_nodes, args.refetch, args.force):
            built.append(name)
    print(f"\n[*] built {len(built)} benchmark set(s); source pins in "
          f"{os.path.join(sources_dir(), 'sources_manifest.csv')}")


if __name__ == "__main__":
    main()
