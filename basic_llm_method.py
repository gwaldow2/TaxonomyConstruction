import json
import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Tuple
from data_manager import get_primary_term, to_lemma_format

class TaxonomyOutput(BaseModel):
    edges: List[Tuple[str, str]] = Field(
        description="A list of parent-child relationship pairs."
    )

def method_llm_single_shot(nodes, client, model_name, reasoning_level='medium'):
    G = nx.DiGraph()
    primary_to_full_map = {get_primary_term(n): n for n in nodes}
    primary_nodes = list(primary_to_full_map.keys())
    vocab_string = ", ".join(primary_nodes)
    num_terms = len(primary_nodes)
    
    # 1. Budgeting
    rtpt_map = {'none': 0, 'low': 10, 'medium': 20, 'high': 40}
    rtpt = rtpt_map.get(reasoning_level.lower(), 20)
    reasoning_budget = num_terms * rtpt
    
    # Pydantic 2.x Schema Export
    json_schema = json.dumps(TaxonomyOutput.model_json_schema())
    
    base_instructions = f"""You are an expert ontologist.
Vocabulary: [{vocab_string}]
Identify parent-child relationships. ONLY use terms from the list.
Output MUST be valid JSON matching this schema: {json_schema}
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
            print(f"    [LLM] Reasoning Complete. Used {res_reasoning.usage.total_tokens} tokens.")
        except Exception as e:
            print(f"    [LLM] Reasoning failed, proceeding to extraction: {e}")

    # --- STAGE 2: Extraction with Retry Logic ---
    attempts = 2
    current_json_budget = (num_terms * 40) + 600 # Base budget
    
    for attempt in range(attempts):
        try:
            extraction_prompt = (
                base_instructions + 
                reasoning_context + 
                "\nCRITICAL: Return ONLY raw JSON. Do NOT use markdown code blocks. "
                "Begin your response immediately with the '{' character."
            )

            res_extract = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a JSON formatter. No prose, no markdown."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.0,
                max_tokens=current_json_budget,
                response_format={"type": "json_object"}
            )

            content = res_extract.choices[0].message.content or ""
            usage = res_extract.usage
            finish_reason = res_extract.choices[0].finish_reason

            print(f"    [Attempt {attempt+1}] Tokens: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}")
            
            if finish_reason == "length":
                print(f"    [WARNING] Output was TRUNCATED (hit max_tokens).")
            
            # --- FIXED SANITIZATION BLOCK ---
            content = content.strip()
            
            # Safely remove opening markdown backticks
            if content.startswith("```"):
                content = content.lstrip("`") # Removes the leading backticks safely
                if content.lower().startswith("json"):
                    content = content[4:] # Removes the 'json' keyword
                content = content.strip()
            
            # Safely remove closing markdown backticks
            if content.endswith("```"):
                content = content.rstrip("`").strip()
            # --------------------------------

            if not content:
                raise ValueError("Empty content returned from LLM")

            parsed_data = json.loads(content)
            edges_list = parsed_data.get("edges", [])
            
            edges_added = 0
            for pair in edges_list:
                if len(pair) == 2:
                    p, c = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
                    if p in primary_to_full_map and c in primary_to_full_map and p != c:
                        G.add_edge(primary_to_full_map[p], primary_to_full_map[c])
                        edges_added += 1
            
            print(f"    [LLM Success] Extracted {edges_added} edges.")
            break # Success, exit retry loop

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"    [Attempt {attempt+1} Failed] Error: {e}")
            if attempt == 0:
                print(f"    [Retry] Increasing token budget by 50% and retrying...")
                current_json_budget = int(current_json_budget * 1.5)
            else:
                print(f"    [Fatal] All attempts failed.")

    return cluster_synonyms_and_enforce_dag(G)
