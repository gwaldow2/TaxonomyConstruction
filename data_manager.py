import os
import re
import json
import random
import requests
import networkx as nx
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import obonet
import geonamescache

DATA_DIR = "./taxonomy_data"
BENCHMARK_DIR = "./benchmark_sets"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ==========================================
# TEXT NORMALIZATION & UTILS
# ==========================================

def clean_term(term):
    term = str(term).strip().lower()
    term = re.sub(r'[^\w\s\-]', '', term)
    term = re.sub(r'\s+', ' ', term)
    return term.strip()

def get_primary_term(node_str):
    node_str = str(node_str).strip().lower()
    match = re.search(r'^([^(]+)\s*\((.*)\)$', node_str)
    if match:
        primary_text = match.group(1).strip()
        syns = [s.strip() for s in match.group(2).split(',') if s.strip()]
        if syns and syns[0] == primary_text:
            return primary_text
    return node_str

def to_lemma_format(terms):
    if not terms: return ""
    unique_terms = []
    for t in terms:
        sub_terms = parse_lemma_format(str(t)) 
        for st in sub_terms:
            st = clean_term(st)
            if st and st not in unique_terms:
                unique_terms.append(st)
                
    if len(unique_terms) == 1:
        return unique_terms[0]
        
    primary = unique_terms[0]
    syns = ", ".join(unique_terms)
    return f"{primary} ({syns})"

def parse_lemma_format(node_str):
    node_str = str(node_str).strip().lower()
    match = re.search(r'^([^(]+)\s*\((.*)\)$', node_str)
    if match:
        primary_text = match.group(1).strip()
        inner_content = match.group(2)
        syns = [s.strip() for s in inner_content.split(',') if s.strip()]
        if syns and syns[0] == primary_text:
            return syns
    if '|' in node_str:
        return [s.strip() for s in node_str.split('|') if s.strip()]
    return [node_str]

# ==========================================
# GRAPH TOPOLOGY & BENCHMARK UTILS
# ==========================================

def enforce_dag(G):
    """
    1. Resolves cycles.
    2. Unifies forest into a single-rooted DAG using a Virtual Root.
    """
    G_dag = G.copy()
    
    # 1. Cycle Breaking
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = next(nx.simple_cycles(G_dag))
            G_dag.remove_edge(cycle[-1], cycle[0])
    except StopIteration:
        pass

    # 2. Virtual Root Unification
    roots = [n for n, d in G_dag.in_degree() if d == 0]
    if len(roots) > 1:
        virtual_root = "virtual_root"
        G_dag.add_node(virtual_root, is_virtual=True)
        for r in roots:
            if r != virtual_root:
                G_dag.add_edge(virtual_root, r)
                
    return G_dag

def save_benchmark_graph(G, name, scale="SUB", train_pairs=None):
    path = os.path.join(BENCHMARK_DIR, f"{name}_{scale}.graphml")
    nx.write_graphml(G, path)
    
    if train_pairs is not None:
        pairs_path = os.path.join(BENCHMARK_DIR, f"{name}_{scale}_train_pairs.json")
        with open(pairs_path, 'w') as f:
            json.dump(train_pairs, f)
            
    print(f"  [Storage] Saved {scale} benchmark GT to {path}")

def load_benchmark_graph(name, scale="SUB"):
    path = os.path.join(BENCHMARK_DIR, f"{name}_{scale}.graphml")
    if not os.path.exists(path):
        return None, None
        
    G = nx.read_graphml(path)
    
    train_pairs = None
    pairs_path = os.path.join(BENCHMARK_DIR, f"{name}_{scale}_train_pairs.json")
    if os.path.exists(pairs_path):
        with open(pairs_path, 'r') as f:
            train_pairs = json.load(f)
            
    return G, train_pairs

def get_closed_subgraph(G, target_nodes=100):
    nodes = list(G.nodes())
    subgraph_nodes = set()
    random.seed(42) 
    random.shuffle(nodes)
    
    for node in nodes:
        if len(subgraph_nodes) >= target_nodes:
            break
        ancestors = nx.ancestors(G, node)
        subgraph_nodes.add(node)
        subgraph_nodes.update(ancestors)
        
    return G.subgraph(subgraph_nodes).copy()

def get_rigorous_80_20_split(G_full):
    """
    Partitions the graph by root-level subtrees to prevent stranded children.
    20% of branches become training seeds, 80% become the test graph.
    """
    if G_full.number_of_nodes() == 0:
        return G_full, []

    roots = [n for n, d in G_full.in_degree() if d == 0]
    
    # If there are no clear roots (e.g., massive cycles), fallback to nodes
    if not roots:
        roots = list(G_full.nodes())

    branches = []
    for r in roots:
        for branch_starter in G_full.successors(r):
            nodes_in_branch = nx.descendants(G_full, branch_starter) | {branch_starter, r}
            branches.append(nodes_in_branch)
            
    # If the graph is entirely flat (no successors), fallback to individual nodes as branches
    if not branches:
        branches = [{n} for n in G_full.nodes()]
    
    random.seed(42)
    random.shuffle(branches)
    
    split_idx = max(1, int(len(branches) * 0.2))
    train_nodes = set().union(*branches[:split_idx])
    test_nodes = set().union(*branches[split_idx:]) if split_idx < len(branches) else set()
    
    # Strict Disjointness: Overlapping common ancestors belong exclusively to Test
    train_nodes = train_nodes - test_nodes
    
    G_test = G_full.subgraph(test_nodes).copy()
    
    G_train = G_full.subgraph(train_nodes)
    train_pairs = [{"parent": u, "child": v} for u, v in G_train.edges()]
    
    return G_test, train_pairs

# ==========================================
# DATASET LOADERS
# ==========================================

def get_semeval_graph(domain, use_synsets=False):
    file_map = {
        "SemEvalFood": "semeval_food.taxo",
        "SemEvalScience": "semeval_science.taxo",
        "SemEvalEnvironment": "semeval_environment.taxo"
    }
    file_path = os.path.join(DATA_DIR, file_map[domain])
    G = nx.DiGraph()
    if not os.path.exists(file_path):
        print(f"  [Warning] {file_path} not found in {DATA_DIR}.")
        return G, []
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=["rel_id", "Hyponym", "Hypernym"], on_bad_lines='skip')
        for _, row in df.iterrows():
            if pd.notna(row["Hyponym"]) and pd.notna(row["Hypernym"]):
                hypo_clean = clean_term(row["Hyponym"])
                hyper_clean = clean_term(row["Hypernym"])
                G.add_edge(hyper_clean, hypo_clean)
    except Exception:
        pass
    return get_rigorous_80_20_split(G)

def get_wordnet_food_graph(use_synsets=False):
    nltk.download('wordnet', quiet=True)
    G = nx.DiGraph()
    queue = [wn.synset('food.n.01'), wn.synset('food.n.02')]
    visited = set(queue)
    
    while queue:
        current = queue.pop(0)
        if use_synsets:
            u_name = to_lemma_format([clean_term(l.name().replace('_', ' ')) for l in current.lemmas()])
        else:
            u_name = clean_term(current.name().split('.')[0].replace('_', ' '))
            
        for hypo in current.hyponyms():
            if use_synsets:
                v_name = to_lemma_format([clean_term(l.name().replace('_', ' ')) for l in hypo.lemmas()])
            else:
                v_name = clean_term(hypo.name().split('.')[0].replace('_', ' '))
            G.add_edge(u_name, v_name)
            if hypo not in visited:
                visited.add(hypo)
                queue.append(hypo)
    return get_rigorous_80_20_split(G)

def get_cell_ontology_graph(use_synsets=False):
    file_path = os.path.join(DATA_DIR, "cl-basic.obo")
    url = "http://purl.obolibrary.org/obo/cl/cl-basic.obo"
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
    obo_graph = obonet.read_obo(file_path)
    G = nx.DiGraph()
    node_mapping = {}
    for u, data in obo_graph.nodes(data=True):
        name = data.get('name', u)
        if use_synsets:
            terms = [clean_term(name)]
            for syn in data.get('synonym', []):
                if '"' in syn:
                    s = clean_term(syn.split('"')[1])
                    if s not in terms: terms.append(s)
            node_mapping[u] = to_lemma_format(terms)
        else:
            node_mapping[u] = clean_term(name)
            
    for u, v, key in obo_graph.edges(keys=True):
        if key == 'is_a':
            parent_name = node_mapping.get(v)
            child_name = node_mapping.get(u)
            if parent_name and child_name: G.add_edge(parent_name, child_name)
            
    return get_rigorous_80_20_split(G)

def get_csv_graph(file_path, use_synsets=False):
    print(f"Loading custom CSV edge list dataset from {file_path}...")
    G = nx.DiGraph()
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            weight = row.get("Weight", 1)
            try:
                is_edge = float(weight) == 1.0
            except (ValueError, TypeError):
                is_edge = False
                
            if is_edge:
                u_str, v_str = str(row["Source"]).strip(), str(row["Target"]).strip()
                if not use_synsets:
                    u_str = re.sub(r'\s*\(.*?\)', '', u_str).strip()
                    v_str = re.sub(r'\s*\(.*?\)', '', v_str).strip()
                    if '|' in u_str: u_str = u_str.split('|')[0].strip()
                    if '|' in v_str: v_str = v_str.split('|')[0].strip()

                u_terms = [clean_term(t) for t in u_str.split('|')] if '|' in u_str else [clean_term(u_str)]
                v_terms = [clean_term(t) for t in v_str.split('|')] if '|' in v_str else [clean_term(v_str)]
                G.add_edge(to_lemma_format(u_terms), to_lemma_format(v_terms))
    except Exception as e:
        print(f"CRITICAL ERROR loading CSV {file_path}: {e}")
    return get_rigorous_80_20_split(G)

def get_llms4ol_task_c_data(domain_folder_path):
    domain_name = os.path.basename(os.path.normpath(domain_folder_path)).lower()
    train_pairs_file = os.path.join(domain_folder_path, "train", f"{domain_name}_train_pairs.json")
    G_full = nx.DiGraph()
    
    if os.path.exists(train_pairs_file):
        with open(train_pairs_file, 'r', encoding='utf-8') as f:
            try:
                pairs = json.load(f)
                for pair in pairs:
                    p = clean_term(pair.get("parent", ""))
                    c = clean_term(pair.get("child", ""))
                    if p and c: G_full.add_edge(p, c)
            except json.JSONDecodeError:
                print(f"  [Error] Invalid JSON format in {train_pairs_file}")
                
    return get_rigorous_80_20_split(G_full)
