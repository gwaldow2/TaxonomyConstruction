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
os.makedirs(DATA_DIR, exist_ok=True)

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

def enforce_dag(G):
    """Retained for preparing the Ground Truth Graph."""
    G_dag = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = next(nx.simple_cycles(G_dag))
            G_dag.remove_edge(cycle[-1], cycle[0])
    except StopIteration:
        pass
    return G_dag

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
        return G
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=["rel_id", "Hyponym", "Hypernym"], on_bad_lines='skip')
        for _, row in df.iterrows():
            if pd.notna(row["Hyponym"]) and pd.notna(row["Hypernym"]):
                hypo_clean = clean_term(row["Hyponym"])
                hyper_clean = clean_term(row["Hypernym"])
                G.add_edge(hyper_clean, hypo_clean)
        print(f"  -> Successfully loaded {domain}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return G

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
    return G

def get_google_products_graph(use_synsets=False):
    file_path = os.path.join(DATA_DIR, "google_products.txt")
    url = "https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt"
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
    G = nx.DiGraph()
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split('-', 1)
        if len(parts) > 1:
            path = parts[1].strip().split(' > ')
            for i in range(len(path) - 1):
                G.add_edge(clean_term(path[i]), clean_term(path[i+1]))
    return G

def get_geonames_graph(use_synsets=False):
    gc = geonamescache.GeonamesCache()
    G = nx.DiGraph()
    root = "earth"
    countries = gc.get_countries()
    states = gc.get_us_states()
    cities = gc.get_cities()
    
    for c_code, c_data in countries.items():
        G.add_edge(clean_term(root), clean_term(c_data['name']))
    us_name = clean_term(countries['US']['name'])
    for s_code, s_data in states.items():
        G.add_edge(us_name, clean_term(s_data['name']))
    for city_id, city_data in cities.items():
        if city_data['countrycode'] == 'US':
            state_code = city_data['admin1code']
            if state_code in states:
                state_name = clean_term(states[state_code]['name'])
                G.add_edge(state_name, clean_term(city_data['name']))
    return G

def parse_obo_graph(file_path, use_synsets):
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
                    if s not in terms: 
                        terms.append(s)
            node_mapping[u] = to_lemma_format(terms)
        else:
            node_mapping[u] = clean_term(name)
            
    for u, v, key in obo_graph.edges(keys=True):
        if key == 'is_a':
            parent_name = node_mapping.get(v)
            child_name = node_mapping.get(u)
            if parent_name and child_name:
                G.add_edge(parent_name, child_name)
    return G

def get_gene_ontology_graph(use_synsets=False):
    file_path = os.path.join(DATA_DIR, "go-basic.obo")
    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    return parse_obo_graph(file_path, use_synsets)

def get_cell_ontology_graph(use_synsets=False):
    file_path = os.path.join(DATA_DIR, "cl-basic.obo")
    url = "http://purl.obolibrary.org/obo/cl/cl-basic.obo"
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    return parse_obo_graph(file_path, use_synsets)

def get_csv_graph(file_path, use_synsets=False):
    print(f"Loading custom CSV edge list dataset from {file_path}...")
    G = nx.DiGraph()
    try:
        df = pd.read_csv(file_path)
        if "Source" not in df.columns or "Target" not in df.columns:
            print("  [Error] Expected 'Source' and 'Target' columns. Ensure the input is an edge list.")
            return G

        for _, row in df.iterrows():
            weight = row.get("Weight", 1)
            try:
                is_edge = float(weight) == 1.0
            except (ValueError, TypeError):
                is_edge = False
                
            if is_edge:
                u_str = str(row["Source"]).strip()
                v_str = str(row["Target"]).strip()
                
                if not use_synsets:
                    u_str = re.sub(r'\s*\(.*?\)', '', u_str).strip()
                    v_str = re.sub(r'\s*\(.*?\)', '', v_str).strip()
                    if '|' in u_str: u_str = u_str.split('|')[0].strip()
                    if '|' in v_str: v_str = v_str.split('|')[0].strip()

                u_terms = [clean_term(t) for t in u_str.split('|')] if '|' in u_str else [clean_term(u_str)]
                v_terms = [clean_term(t) for t in v_str.split('|')] if '|' in v_str else [clean_term(v_str)]
                
                G.add_edge(to_lemma_format(u_terms), to_lemma_format(v_terms))

        print(f"  -> Successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"CRITICAL ERROR loading CSV {file_path}: {e}")
    return G

def get_llms4ol_task_c_data(domain_folder_path, use_synsets=False):
    """
    Parses the LLMs4OL 2025 Task C dataset structure.
    Returns: (G_gt, test_nodes, train_pairs)
    """
    G_gt = nx.DiGraph()
    test_nodes = []
    train_pairs = []
    
    # Extract prefix (e.g., 'OBI', 'SWEET') from folder path
    domain_name = os.path.basename(os.path.normpath(domain_folder_path))
    
    train_pairs_file = os.path.join(domain_folder_path, f"{domain_name}_train_pairs.json")
    test_types_file = os.path.join(domain_folder_path, f"{domain_name}_test_types.txt")
    
    if os.path.exists(train_pairs_file):
        with open(train_pairs_file, 'r', encoding='utf-8') as f:
            try:
                train_pairs = json.load(f)
                for pair in train_pairs:
                    p = clean_term(pair.get("parent", ""))
                    c = clean_term(pair.get("child", ""))
                    if p and c:
                        G_gt.add_edge(p, c)
            except json.JSONDecodeError:
                print(f"  [Error] Invalid JSON format in {train_pairs_file}")
    else:
        print(f"  [Warning] Missing train_pairs file: {train_pairs_file}")
                    
    if os.path.exists(test_types_file):
        with open(test_types_file, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if term:
                    test_nodes.append(clean_term(term))
    else:
        print(f"  [Warning] Missing test_types file: {test_types_file}")
                    
    return G_gt, test_nodes, train_pairs
