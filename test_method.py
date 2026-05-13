import os
import json
import time
import traceback
import networkx as nx
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Tuple
from data_manager import get_primary_term

class TaxonomyOutput(BaseModel):
    edges: List[Tuple[str, str]] = Field(
        description="A list of parent-child relationship pairs."
    )

def debug_llm_single_shot(nodes, client, model_name, reasoning_level='medium', domain_name="UNKNOWN"):
    """
    Extensively instrumented diagnostic method.
    Dumps all raw inputs, outputs, usage stats, and errors to disk.
    """
    # 1. Setup Debug Directory
    debug_dir = "./llm_debug_dumps"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = os.path.join(debug_dir, f"debug_{domain_name}_{timestamp}.txt")
    
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    vocab_string = ", ".join(primary_nodes)
    num_terms = len(primary_nodes)
    
    rtpt_map = {'none': 0, 'low': 10, 'medium': 20, 'high': 40}
    rtpt = rtpt_map.get(reasoning_level.lower(), 20)
    reasoning_budget = num_terms * rtpt
    
    json_schema = json.dumps(TaxonomyOutput.model_json_schema())
    
    base_instructions = f"""You are an expert ontologist.
Vocabulary: [{vocab_string}]
Identify parent-child relationships. ONLY use terms from the list.
Output MUST be valid JSON matching this schema: {json_schema}
"""

    def write_log(section_title, content):
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n=== {section_title} ===\n{'='*60}\n")
            f.write(str(content) + "\n")

    write_log("METADATA", f"Model: {model_name}\nNodes: {num_terms}\nReasoning Budget: {reasoning_budget}")

    # --- STAGE 1: Reasoning ---
    reasoning_context = ""
    if reasoning_budget > 0:
        stage1_prompt = base_instructions + "\nThink step-by-step."
        write_log("STAGE 1 PROMPT (Reasoning)", stage1_prompt)
        
        try:
            t0 = time.time()
            res_reasoning = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": stage1_prompt}],
                temperature=0.6,
                max_tokens=reasoning_budget
            )
            raw_thought = res_reasoning.choices[0].message.content or ""
            reasoning_context = f"\nPreliminary Analysis:\n{raw_thought}\n"
            
            p_tok = getattr(res_reasoning.usage, 'prompt_tokens', -1)
            c_tok = getattr(res_reasoning.usage, 'completion_tokens', -1)
            finish = res_reasoning.choices[0].finish_reason
            
            write_log("STAGE 1 TELEMETRY", f"Time: {time.time()-t0:.2f}s\nPrompt Tokens: {p_tok}\nCompletion Tokens: {c_tok}\nFinish Reason: {finish}")
            write_log("STAGE 1 RAW OUTPUT", raw_thought)
            
        except Exception as e:
            write_log("STAGE 1 EXCEPTION", traceback.format_exc())
            print(f"    [DEBUG] Stage 1 crashed. See log: {debug_file}")

    # --- STAGE 2: Extraction ---
    current_json_budget = (num_terms * 40) + 600
    
    extraction_prompt = (
        base_instructions + 
        reasoning_context + 
        "\nCRITICAL: Return ONLY raw JSON. Do NOT use markdown code blocks. "
        "Begin your response immediately with the '{' character."
    )
    
    write_log("STAGE 2 PROMPT (Extraction)", extraction_prompt)
    write_log("STAGE 2 SETTINGS", f"Max Tokens Requested: {current_json_budget}\nResponse Format: JSON_OBJECT")

    try:
        t0 = time.time()
        res_extract = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict JSON data extraction pipeline. Output raw JSON without markdown."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.0,
            max_tokens=current_json_budget,
            response_format={"type": "json_object"}
        )

        content = res_extract.choices[0].message.content or ""
        usage = res_extract.usage
        finish_reason = res_extract.choices[0].finish_reason

        p_tokens = getattr(usage, 'prompt_tokens', -1)
        c_tokens = getattr(usage, 'completion_tokens', -1)
        
        write_log("STAGE 2 TELEMETRY", f"Time: {time.time()-t0:.2f}s\nPrompt Tokens: {p_tokens}\nCompletion Tokens: {c_tokens}\nFinish Reason: {finish_reason}")
        write_log("STAGE 2 RAW OUTPUT (Exact string from server)", repr(content)) # repr() shows hidden characters/newlines

        # --- DUMB PARSING (Just to see if it even can parse) ---
        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            content_stripped = content_stripped.lstrip("`")
            if content_stripped.lower().startswith("json"):
                content_stripped = content_stripped[4:]
            content_stripped = content_stripped.strip()
        if content_stripped.endswith("
```"):
            content_stripped = content_stripped.rstrip("`").strip()

        write_log("STAGE 2 SANITIZED STRING", content_stripped)

        parsed_data = json.loads(content_stripped)
        edges_list = parsed_data.get("edges", [])
        
        edges_added = 0
        for pair in edges_list:
            if len(pair) == 2:
                p, c = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
                if p in primary_to_full_map and c in primary_to_full_map and p != c:
                    G.add_edge(primary_to_full_map[p], primary_to_full_map[c])
                    edges_added += 1
        
        write_log("STAGE 2 PARSE RESULT", f"SUCCESS. Edges extracted: {edges_added}")
        print(f"    [DEBUG] Log written to {debug_file} | Edges: {edges_added}")

    except Exception as e:
        write_log("STAGE 2 EXCEPTION / CRASH", traceback.format_exc())
        print(f"    [DEBUG] Extraction failed. Inspect log: {debug_file}")

    return G
