import networkx as nx
import re
import time
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
            reasoning_context = f"\nPreliminary Analysis:\n{raw_thought}\n"
            used_tokens = getattr(res_reasoning.usage, 'total_tokens', 'Unknown')
            print(f"    [LLM Zero-Shot] Reasoning Complete. Used {used_tokens} tokens.")
        except Exception as e:
            print(f"    [LLM Zero-Shot] Reasoning failed: {e}")

    # --- STAGE 2: Regex-Extracted Line Format ---
    # We ask for a strict, simple line format. No JSON schema, no brackets.
    # This prevents the logit loop and is highly token-efficient.
    extraction_prompt = (
        base_instructions + 
        reasoning_context + 
        "\nCRITICAL: Output the relationships as a simple text list. "
        "Format exactly like this, one pair per line:\n"
        "[\"parent\", \"child\"]\n"
        "[\"parent\", \"child\"]\n"
    )

    current_budget = (num_terms * 40) + 600

    try:
        t0 = time.time()
        res_extract = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a data extraction pipeline. Output only the requested list format."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.0,
            max_tokens=current_budget
            # NOTICE: response_format is intentionally removed.
        )
        
        content = res_extract.choices[0].message.content or ""
        usage = res_extract.usage
        finish = res_extract.choices[0].finish_reason
        
        p_tokens = getattr(usage, 'prompt_tokens', 0)
        c_tokens = getattr(usage, 'completion_tokens', 0)
        print(f"    [LLM Zero-Shot] Extraction: Prompt={p_tokens}, Completion={c_tokens}, Finish={finish}")

        if finish == "length":
            print(f"    [WARNING] Output hit token limit. Salvaging parsed results.")

        # --- BULLETPROOF REGEX PARSER ---
        # This will perfectly extract ["parent", "child"] or ['parent', 'child']
        # from anywhere in the text, ignoring all prose, markdown, or truncation errors.
        matches = re.findall(r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]', content)
        
        edges_added = 0
        for p, c in matches:
            p, c = p.strip().lower(), c.strip().lower()
            if p in primary_to_full_map and c in primary_to_full_map and p != c:
                G.add_edge(primary_to_full_map[p], primary_to_full_map[c])
                edges_added += 1
                
        print(f"    [LLM Zero-Shot] SUCCESS | Extracted {edges_added} valid edges.")

    except Exception as e:
        print(f"    [LLM Zero-Shot] FATAL ERROR | {e}")

    return G
