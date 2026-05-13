import torch
import networkx as nx
from tqdm import tqdm
from data_manager import to_lemma_format

def enforce_dag(G):
    G_dag = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = next(nx.simple_cycles(G_dag))
            G_dag.remove_edge(cycle[-1], cycle[0])
    except StopIteration:
        pass
    return G_dag

def cluster_synonyms_and_enforce_dag(G):
    bidirectional_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    G_sym = nx.Graph()
    G_sym.add_nodes_from(G.nodes())
    G_sym.add_edges_from(bidirectional_edges)
    clusters = list(nx.connected_components(G_sym))
    
    condensed_dag = nx.DiGraph()
    node_mapping = {}
    for cluster in clusters:
        new_node_name = to_lemma_format(sorted(list(cluster)))
        condensed_dag.add_node(new_node_name)
        for node in cluster:
            node_mapping[node] = new_node_name

    for u, v in G.edges():
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        if new_u != new_v:
            condensed_dag.add_edge(new_u, new_v)

    return enforce_dag(condensed_dag)

def resolve_cycle_ppl(cur_G, cycle):
    cycle_ppls = {}
    for u, v in cycle:
        val = cur_G[u][v]["weight"]
        cycle_ppls[(u, v)] = val
    highest_ppl = sorted(cycle_ppls.items(), key=lambda x: x[1], reverse=True)[0][0]
    cur_G.remove_edge(*highest_ppl)

def resolve_graph_cycles(G_pred):
    while True:
        try:
            cycle = nx.find_cycle(G_pred)
            resolve_cycle_ppl(G_pred, cycle)
        except nx.NetworkXNoCycle:
            break

def precompute_taxollama_ppl(nodes, model, tokenizer, device):
    scores_cache = {u: {} for u in nodes}
    sys_prompt = "[INST] <<SYS>> You are a helpfull assistant. List all the possible words divided with a coma. Your answer should not include anything except the words divided by a coma<</SYS>>\n"
    
    def get_masked_ppl(child, parent):
        prompt_text = f"{sys_prompt}hyponym: {child} | hypernyms: [/INST]"
        target_text = f" {parent}"
        full_text = prompt_text + target_text
        
        full_enc = tokenizer(full_text, return_tensors="pt")
        input_ids = full_enc.input_ids.to(device)
        
        prompt_enc = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]
        
        labels = input_ids.clone()
        mask_len = min(prompt_len, labels.shape[1] - 1)
        labels[0, :mask_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            return torch.exp(outputs.loss).item()

    for u in tqdm(nodes, desc="  -> TaxoLLaMA PPL Caching", leave=False):
        for v in nodes:
            if u == v: continue
            scores_cache[u][v] = get_masked_ppl(child=u, parent=v)
            
    return scores_cache

def build_taxollama_graph(nodes, scores_cache, threshold=15.0, max_parents=1):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    for u in nodes:
        valid_parents = []
        for v in nodes:
            if u == v: continue
            ppl = scores_cache[u][v]
            if ppl < threshold:
                valid_parents.append((v, ppl))
                
        valid_parents.sort(key=lambda x: x[1])
        
        for parent, ppl in valid_parents[:max_parents]:
            G.add_edge(parent, u, weight=ppl) 
            
    resolve_graph_cycles(G)
    print(f"    [TaxoLLaMA] SUCCESS | Built {G.number_of_edges()} raw edges strictly below PPL threshold {threshold}.")
    return cluster_synonyms_and_enforce_dag(G)
