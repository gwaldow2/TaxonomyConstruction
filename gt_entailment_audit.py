"""Audit GT source ontologies for subsumption our parsers would miss.

Answers: "do any alternative encodings of is-a exist in a source that the
pipeline's extraction does not capture?" Three assurance layers, strongest last:

1. census    - count every OWL/OBO construct in the file that can induce class
               subsumption. A file whose census shows only rdfs:subClassOf (plus
               constructs the parser handles) cannot hide anything.
2. rl-diff   - materialize the OWL 2 RL schema closure with owlrl (pure Python)
               and diff every entailed named-class subsumption against the
               transitive closure of the pipeline parser's graph. Catches
               unions, intersections, equivalences, chained bnode expressions -
               mechanically, with no hand enumeration of axiom types.
3. (gold)    - a complete OWL 2 DL classification (ELK or HermiT, via ROBOT:
               `robot reason --reasoner ELK -i src.owl` then diff the inferred
               hierarchy). Needs Java; not run here. For the OBO sources this is
               a cross-check only: OBO Foundry release .obo files already ship
               the reasoner-materialized is_a hierarchy.

Usage: python gt_entailment_audit.py <kind> <path>   (kind: ttl | jsonld | obo)
"""
import re
import sys
from collections import Counter

OWL_SUBSUMPTION_PREDS = [
    "subClassOf", "equivalentClass", "unionOf", "intersectionOf", "oneOf",
    "disjointUnionOf", "complementOf", "someValuesFrom", "allValuesFrom",
    "hasValue", "onProperty",
]
OBO_SUBSUMPTION_TAGS = ["is_a", "intersection_of", "union_of", "equivalent_to",
                        "disjoint_from", "relationship"]


def clean_term(s):
    # replaced by the pipeline's own normalizer at import time in main(); this
    # fallback keeps the census usable standalone
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def census_rdf(rg):
    cnt = Counter()
    for _, p, _ in rg:
        local = str(p).split("#")[-1].split("/")[-1]
        if local in OWL_SUBSUMPTION_PREDS:
            cnt[local] += 1
    return cnt


def census_obo(path):
    text = open(path, encoding="utf-8").read()
    return Counter({t: len(re.findall(rf"^{t}:", text, re.M))
                    for t in OBO_SUBSUMPTION_TAGS})


def rl_diff(rg, parser_graph, errata=(), scope=None, clean=None):
    """OWL 2 RL closure vs the parser graph's transitive closure, on named classes.

    Documented errata axioms are removed before expansion (they are excluded from
    the GT deliberately, so their entailments must not count as missing). Parser
    nodes may be synset-style ("head (head, alias)"); a pair counts as extracted
    when any node carrying the parent name reaches any node carrying the child.
    """
    import networkx as nx
    import owlrl
    from rdflib.namespace import OWL, RDFS
    from rdflib.term import BNode, URIRef

    ct = clean or clean_term
    labels = {s: ct(o) for s, _, o in rg.triples((None, RDFS.label, None))}

    def nm(x):
        return labels.get(x) or ct(str(x).split("#")[-1].split("/")[-1])

    for cls_label, members in errata:
        for c, e in list(rg.subject_objects(OWL.equivalentClass)):
            if isinstance(c, URIRef) and isinstance(e, BNode) and nm(c) == cls_label:
                for lst in rg.objects(e, OWL.unionOf):
                    if frozenset(nm(m) for m in rg.items(lst)
                                 if isinstance(m, URIRef)) == members:
                        rg.remove((c, OWL.equivalentClass, e))
                        print(f"  [erratum] excluded from entailment base: "
                              f"'{cls_label}' = union{sorted(members)}")

    if scope is not None:
        named = set(scope)
        # equivalences that reach outside the scope (schema.org bridges to
        # rdfs:Class, SNOMED, FIBO, ...) are outside the GT by the namespace
        # rule; drop them so cross-vocabulary entailments don't count as missing
        dropped = 0
        for s, p, o in list(rg.triples((None, OWL.equivalentClass, None))):
            if isinstance(s, URIRef) and isinstance(o, URIRef) \
                    and (s not in named or o not in named):
                rg.remove((s, p, o))
                dropped += 1
        if dropped:
            print(f"  [scope] dropped {dropped} out-of-scope equivalence bridge(s)")
    else:
        named = {s for s, _, _ in rg.triples((None, None, OWL.Class))} | \
                {s for s, _, _ in rg.triples((None, RDFS.subClassOf, None))}
        named = {s for s in named if isinstance(s, URIRef)
                 and not str(s).startswith(str(OWL)) and "w3.org" not in str(s)}

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics, axiomatic_triples=False,
                           datatype_axioms=False).expand(rg)

    entailed = set()
    for s, _, o in rg.triples((None, RDFS.subClassOf, None)):
        if s in named and o in named and isinstance(o, URIRef):
            p, c = nm(o), nm(s)
            if p and c and p != c:
                entailed.add((p, c))

    H = nx.transitive_closure(nx.DiGraph(parser_graph.edges()))
    carrier = {}
    for n in H.nodes():
        s = str(n)
        terms = {s}
        if "(" in s and s.endswith(")"):
            head, inner = s.split("(", 1)
            terms = {head.strip()} | {t.strip() for t in inner[:-1].split(",")}
        for t in terms:
            carrier.setdefault(t, set()).add(n)

    def covered(p, c):
        for a in carrier.get(p, ()):
            for b in carrier.get(c, ()):
                if a == b or H.has_edge(a, b):
                    return True
        return False

    extracted = set(H.edges())
    missing = sorted((p, c) for p, c in entailed if not covered(p, c))
    return entailed, extracted, missing


def main():
    kind, path = sys.argv[1], sys.argv[2]
    if kind == "obo":
        cnt = census_obo(path)
        print(f"census (OBO tags): {dict(cnt)}")
        print("note: union_of/equivalent_to absent -> no union-style subsumption; "
              "intersection_of genus edges are entailed is_a (verify each genus is "
              "also reachable via asserted is_a). OBO release files ship the "
              "reasoner-materialized is_a hierarchy; the gold cross-check is ELK "
              "over the .owl edition via ROBOT.")
        return

    from rdflib import Graph as RGraph
    rg = RGraph()
    rg.parse(path, format={"ttl": "turtle", "jsonld": "json-ld"}[kind])
    print(f"triples: {len(rg)}")
    print(f"census (subsumption-bearing predicates): {dict(census_rdf(rg))}")

    import full_source_benchmarks as fsb
    from data_manager import clean_term as dm_clean
    parser = {"ttl": fsb.parse_ttl, "jsonld": fsb.parse_schemaorg_jsonld}[kind]
    G = parser(path)

    scope = None
    if kind == "jsonld":
        # mirror the parser's GT scope: rdfs:Class-typed, schema.org namespace.
        # Punned enumeration members (e.g. schema:Audiology, typed only
        # MedicalSpecialty yet carrying subClassOf) and equivalences to external
        # vocabularies (SNOMED/FIBO/...) are outside the GT by design.
        from rdflib.namespace import RDF, RDFS as _RDFS
        from rdflib.term import URIRef as _U
        rdfs_class = _U("http://www.w3.org/2000/01/rdf-schema#Class")
        scope = {s for s in rg.subjects(RDF.type, rdfs_class)
                 if isinstance(s, _U) and "schema.org" in str(s)}

    entailed, extracted, missing = rl_diff(rg, G, errata=fsb.SOURCE_ERRATA.get(kind, ()),
                                           scope=scope, clean=dm_clean)
    print(f"\nOWL-RL entailed named-class subsumptions: {len(entailed)}")
    print(f"parser closure pairs:                     {len(extracted)}")
    print(f"entailed but NOT extracted:               {len(missing)}")
    for p, c in missing[:25]:
        print(f"  MISSING  {p} -> {c}")
    if not missing:
        print("  [ok] the parser captures every OWL-RL-entailed subsumption")


if __name__ == "__main__":
    main()
