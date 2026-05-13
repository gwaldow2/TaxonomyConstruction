import os
import json
import time
import traceback
# Ensure you run this in your project root so it can find your data_manager
try:
    from data_manager import get_primary_term
except ImportError:
    print("Warning: data_manager.py not found. Using identity function for terms.")
    def get_primary_term(x): return x

from openai import OpenAI

# --- CONFIGURATION ---
MODEL_NAME = "openai/gpt-oss-120b"
BASE_URL = "http://localhost:8000/v1"
API_KEY = "woohoo"
DEBUG_LOG = "./standalone_debug_report.txt"
RAW_JSON_DUMP = "./raw_llm_output.json"

# A representative sample of 20 nodes to test the "Low" reasoning budget behavior
SAMPLE_NODES = [
    "beverage", "alcohol", "ale", "beer", "pale ale", "spruce beer", 
    "food", "produce", "vegetable", "legume", "pea", "field pea", 
    "meat", "pork", "beef", "poultry", "chicken", "dairy", "cheese", "milk"
]

def run_standalone_diagnostic():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    # 1. Initialize Log
    with open(DEBUG_LOG, "w", encoding="utf-8") as f:
        f.write(f"LLM STANDALONE DIAGNOSTIC - {time.ctime()}\n")
        f.write(f"Model: {MODEL_NAME} | URL: {BASE_URL}\n")
        f.write("="*60 + "\n")

    def log(section, content):
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{section}]\n")
            f.write(str(content) + "\n")

    print(f"Starting diagnostic... Logging to {DEBUG_LOG}")

    # --- SETUP BUDGET ---
    num_terms = len(SAMPLE_NODES)
    # Testing 'low' reasoning level (10 tokens per term)
    reasoning_budget = num_terms * 10 
    # Requested budget for extraction
    extraction_budget = (num_terms * 40) + 600
    
    vocab_str = ", ".join(SAMPLE_NODES)
    
    # --- STAGE 1: REASONING ---
    print("Executing Stage 1 (Reasoning)...")
    s1_prompt = f"Vocabulary: [{vocab_str}]\nThink step-by-step about the hierarchy."
    
    try:
        t0 = time.time()
        res_s1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": s1_prompt}],
            temperature=0.6,
            max_tokens=reasoning_budget
        )
        duration_s1 = time.time() - t0
        thought = res_s1.choices[0].message.content or ""
        
        telemetry_s1 = (
            f"Latency: {duration_s1:.2f}s\n"
            f"Prompt Tokens: {res_s1.usage.prompt_tokens}\n"
            f"Completion Tokens: {res_s1.usage.completion_tokens}\n"
            f"Finish Reason: {res_s1.choices[0].finish_reason}"
        )
        log("STAGE 1 TELEMETRY", telemetry_s1)
        log("STAGE 1 RAW THOUGHT", thought)

    except Exception:
        log("STAGE 1 CRASH", traceback.format_exc())
        print("Stage 1 Failed. Check logs.")
        return

    # --- STAGE 2: EXTRACTION ---
    print("Executing Stage 2 (Extraction)...")
    # We use chr(96) to avoid markdown literal issues in the diagnostic log
    md_fence = chr(96) * 3
    
    s2_prompt = (
        f"Vocabulary: [{vocab_str}]\n"
        f"Preliminary Analysis: {thought}\n"
        f"Output a JSON object with key 'edges' containing [parent, child] pairs. "
        f"Start immediately with '{' and do not use {md_fence}."
    )

    try:
        t0 = time.time()
        res_s2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict JSON extraction engine."},
                {"role": "user", "content": s2_prompt}
            ],
            temperature=0.0,
            max_tokens=extraction_budget,
            response_format={"type": "json_object"}
        )
        duration_s2 = time.time() - t0
        content = res_s2.choices[0].message.content or ""
        
        telemetry_s2 = (
            f"Latency: {duration_s2:.2f}s\n"
            f"Prompt Tokens: {res_s2.usage.prompt_tokens}\n"
            f"Completion Tokens: {res_s2.usage.completion_tokens}\n"
            f"Finish Reason: {res_s2.choices[0].finish_reason}\n"
            f"Raw Content Length: {len(content)} chars"
        )
        log("STAGE 2 TELEMETRY", telemetry_s2)
        
        # We use repr() to see if there are hidden null bytes or escape chars
        log("STAGE 2 RAW OUTPUT (REPR)", repr(content))
        
        with open(RAW_JSON_DUMP, "w") as jf:
            jf.write(content)

        # Parsing Test
        try:
            parsed = json.loads(content)
            edges = parsed.get("edges", [])
            log("PARSE STATUS", f"SUCCESS. Found {len(edges)} edges.")
            print(f"Success! Found {len(edges)} edges. See {DEBUG_LOG}")
        except json.JSONDecodeError as je:
            log("PARSE STATUS", f"FAILED: {str(je)}")
            # Log the last 100 chars to see if it was a truncation issue
            log("TRUNCATION CHECK (Last 100 chars)", content[-100:])
            print("Parse Failed. Check logs for truncation data.")

    except Exception:
        log("STAGE 2 CRASH", traceback.format_exc())
        print("Stage 2 Failed. Check logs.")

if __name__ == "__main__":
    run_standalone_diagnostic()
