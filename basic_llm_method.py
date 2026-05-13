import networkx as nx
import re
import time
import os
from datetime import datetime
from data_manager import get_primary_term

def method_llm_single_shot(nodes, client, model_name, reasoning_level='medium'):
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    vocab_string = ", ".join(primary_nodes)
    num_terms = len(primary_nodes)
    
    rtpt_map = {'none': 0, 'low': 10, 'medium': 20, 'high': 40}
    rtpt = rtpt_map.get(reasoning_level.lower(), 20)
    reasoning_budget = num_terms * rtpt
    
    # --- AGGRESSIVE DEBUG DUMP SETUP ---
    debug_dir = "./benchmark_fails_dump"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    safe_name = "".join(x for x in primary_nodes[0] if x.isalnum()) if primary_nodes else "empty"
    debug_file = os.path.join(debug_dir, f"dump_{safe_name}_{timestamp}.txt")
    
    def force_log(title, data):
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*40}\n[{title}]\n{'='*40}\n")
            f.write(str(data) + "\n")
    
    force_log("METADATA", f"Nodes: {num_terms} | Reasoning Budget: {reasoning_budget}")
    
    base_instructions = f"""You are an expert ontologist.
Vocabulary: [{vocab_string}]
Identify ALL parent-child relationships. ONLY use terms EXACTLY as they appear in the list.
"""

    # --- STAGE 1: Reasoning ---
    reasoning_context = ""
    if reasoning_budget > 0:
        try:
            res_reasoning = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": base_instructions + "\nThink step-by-step."}],
                temperature=0.6,
                max_tokens=reasoning_budget
            )
            raw_thought = res_reasoning.choices[0].message.content or ""
            
            # CRITICAL FIX: Aggressive visual delimiters to prevent Attention Bleed
            reasoning_context = f"\n--- START PRELIMINARY ANALYSIS ---\n{raw_thought}\n--- END PRELIMINARY ANALYSIS ---\n"
            
            used_tokens = getattr(res_reasoning.usage, 'total_tokens', 'Unknown')
            print(f"    [LLM Zero-Shot] Reasoning Complete. Used {used_tokens} tokens.")
            force_log("STAGE 1 RAW REASONING", raw_thought)
        except Exception as e:
            print(f"    [LLM Zero-Shot] Reasoning failed: {e}")
            force_log("STAGE 1 ERROR", e)

    # --- STAGE 2: Extraction ---
    # Explicitly telling it to STOP reasoning.
    extraction_prompt = (
        base_instructions + 
        reasoning_context + 
        "\nCRITICAL INSTRUCTION: Your analysis is complete. Now, output the final relationships. "
        "Do NOT write any prose. Do NOT continue reasoning. "
        "Format exactly like this, one pair per line:\n"
        "[\"parent\", \"child\"]\n"
        "[\"parent\", \"child\"]\n"
    )

    current_budget = (num_terms * 40) + 600
    force_log("STAGE 2 EXTRACTION PROMPT", extraction_prompt)

    try:
        t0 = time.time()
        res_extract = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a data extraction pipeline. Output ONLY the requested list format."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.0,
            max_tokens=current_budget
        )
        
        content = res_extract.choices[0].message.content or ""
        usage = res_extract.usage
        finish = res_extract.choices[0].finish_reason
        
        p_tokens = getattr(usage, 'prompt_tokens', 0)
        c_tokens = getattr(usage, 'completion_tokens', 0)
        print(f"    [LLM Zero-Shot] Extraction: Prompt={p_tokens}, Completion={c_tokens}, Finish={finish}")
        
        force_log("STAGE 2 TELEMETRY", f"Prompt: {p_tokens} | Completion: {c_tokens} | Finish: {finish}")
        # repr() reveals invisible strings like thousands of spaces or \n
        force_log("STAGE 2 RAW OUTPUT FROM MODEL", repr(content)) 

        if finish == "length":
            print(f"    [WARNING] Output hit token limit. Dumping to {debug_file}")

        matches = re.findall(r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]', content)
        
        force_log("REGEX MATCHES FOUND", matches)
        
        edges_added = 0
        for p, c in matches:
            p, c = p.strip().lower(), c.strip().lower()
            if p in primary_to_full_map and c in primary_to_full_map and p != c:
                G.add_edge(primary_to_full_map[p], primary_to_full_map[c])
                edges_added += 1
                
        print(f"    [LLM Zero-Shot] SUCCESS | Extracted {edges_added} valid edges.")
        
        if edges_added == 0 and c_tokens > 100:
             print(f"    [FATAL] Extracted 0 edges but used {c_tokens} tokens. See {debug_file}")

    except Exception as e:
        print(f"    [LLM Zero-Shot] FATAL ERROR | {e}")
        force_log("STAGE 2 FATAL ERROR", e)

    return G
