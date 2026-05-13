import os
import json
import time
import sys
import traceback
from datetime import datetime

# Attempt to load the OpenAI library
try:
    from openai import OpenAI
except ImportError:
    print("Error: The 'openai' library is not installed. Run: pip install openai")
    sys.exit(1)

# --- CONFIGURATION (Edit these if your local server uses different ports) ---
MODEL_NAME = "openai/gpt-oss-120b"
BASE_URL = "http://localhost:8000/v1"
API_KEY = "woohoo"
LOG_FILE = "./llm_telemetry_report.txt"

# --- TEST DATA ---
# Using exactly 20 nodes to test the "Low" (10 t/node) budget logic
SAMPLE_VOCAB = [
    "food", "produce", "vegetable", "fruit", "legume",
    "meat", "poultry", "beef", "pork", "chicken",
    "beverage", "alcohol", "beer", "wine", "spirit",
    "dairy", "cheese", "milk", "butter", "yogurt"
]

def run_diagnostic():
    """
    Executes a high-resolution telemetry test on the local LLM server.
    """
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    # Initialize the log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"LLM DIAGNOSTIC REPORT - {datetime.now()}\n")
        f.write(f"Target Server: {BASE_URL} | Model: {MODEL_NAME}\n")
        f.write("-" * 60 + "\n")

    def log(title, data):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[SECTION: {title}]\n")
            f.write(str(data) + "\n")
            f.write("-" * 40 + "\n")

    print(f"Starting Diagnostic. Data will be saved to: {LOG_FILE}")

    # 1. Calculate Budgets
    num_terms = len(SAMPLE_VOCAB)
    # Reasoning: 10 tokens per term = 200 total
    reasoning_limit = num_terms * 10 
    # Extraction: Approx 30-40 tokens per edge + buffer
    extraction_limit = (num_terms * 40) + 600
    
    vocab_str = ", ".join(SAMPLE_VOCAB)

    # --- STAGE 1: REASONING TEST ---
    print(f"  -> Testing Stage 1 (Reasoning Budget: {reasoning_limit} tokens)...")
    s1_prompt = f"Vocabulary: [{vocab_str}]\nThink step-by-step about the hierarchical relationships."
    
    try:
        t0 = time.time()
        res_s1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": s1_prompt}],
            temperature=0.6,
            max_tokens=reasoning_limit
        )
        t1 = time.time()
        
        thought = res_s1.choices[0].message.content or ""
        s1_stats = {
            "duration_sec": round(t1 - t0, 2),
            "prompt_tokens": getattr(res_s1.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(res_s1.usage, 'completion_tokens', 0),
            "finish_reason": res_s1.choices[0].finish_reason
        }
        
        log("STAGE 1 TELEMETRY", json.dumps(s1_stats, indent=2))
        log("STAGE 1 RAW CONTENT", thought)
        
    except Exception:
        log("STAGE 1 ERROR", traceback.format_exc())
        print("!! Stage 1 Failed. See log.")
        return

    # --- STAGE 2: EXTRACTION TEST ---
    print(f"  -> Testing Stage 2 (Extraction Budget: {extraction_limit} tokens)...")
    
    # Using chr(96) to generate backticks safely without literal string conflicts
    fence = chr(96) * 3
    s2_prompt = (
        f"Vocabulary: [{vocab_str}]\n"
        f"Analysis: {thought}\n\n"
        f"Output a JSON object with a key 'edges' containing [parent, child] pairs. "
        f"Do NOT use markdown {fence} blocks. Start immediately with the opening bracket."
    )

    try:
        t0 = time.time()
        res_s2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict JSON data formatter."},
                {"role": "user", "content": s2_prompt}
            ],
            temperature=0.0,
            max_tokens=extraction_limit,
            response_format={"type": "json_object"}
        )
        t1 = time.time()

        content = res_s2.choices[0].message.content or ""
        s2_stats = {
            "duration_sec": round(t1 - t0, 2),
            "prompt_tokens": getattr(res_s2.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(res_s2.usage, 'completion_tokens', 0),
            "finish_reason": res_s2.choices[0].finish_reason,
            "raw_char_count": len(content)
        }

        log("STAGE 2 TELEMETRY", json.dumps(s2_stats, indent=2))
        # repr() ensures we see hidden escape characters or null bytes
        log("STAGE 2 RAW OUTPUT (REPR)", repr(content))

        # Parsing Verification
        try:
            parsed = json.loads(content)
            edge_count = len(parsed.get("edges", []))
            log("PARSE SUCCESS", f"Extracted {edge_count} edges.")
            print(f"Success! Extracted {edge_count} edges.")
        except json.JSONDecodeError as je:
            log("PARSE FAILURE", f"JSON Error: {je}")
            # Log the tail end of the string to check for truncation
            log("STRING TAIL (Last 100 chars)", content[-100:])
            print("!! Extraction succeeded but JSON was unparseable. See log.")

    except Exception:
        log("STAGE 2 ERROR", traceback.format_exc())
        print("!! Stage 2 Failed. See log.")

if __name__ == "__main__":
    run_diagnostic()
