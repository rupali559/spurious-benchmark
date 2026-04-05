"""
evaluate.py
Reads spurious_features_validated.json and evaluates:
  - Qwen alone (no memory)
  - Mem0 + Qwen (with memory)
  - A-mem-sys + Qwen (with memory)

Generates: results.output
"""

import json
import time
import argparse
from datetime import date
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import call_llm

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

try:
    from amem import AgentMemory
    AMEM_AVAILABLE = True
except ImportError:
    AMEM_AVAILABLE = False

def build_causal_query(x, y):
    return f"Does {x} cause or lead to {y}?\n\nInstructions:\n- Answer YES only if there is a clear causal relationship\n- Do not rely on surface similarity or correlation\n- If unsure, answer NO\n- Answer with only YES or NO"

def build_spurious_query(spurious_name, y):
    return f"Does {spurious_name} cause or lead to {y}?\n\nInstructions:\n- Answer YES only if there is a clear causal relationship\n- Do not rely on surface similarity or correlation\n- If unsure, answer NO\n- Answer with only YES or NO"

def parse_answer(response):
    r = response.strip().lower()
    if r.startswith("yes"):
        return "yes"
    if r.startswith("no"):
        return "no"
    return "unknown"

def run_qwen_only(data):
    causal_correct = causal_total = 0
    spurious_correct = spurious_total = 0

    for entry in tqdm(data, desc="Qwen only"):
        x = entry.get("causal_feature", "")
        y = entry.get("hypothesis", "")

        q = build_causal_query(x, y)
        ans = parse_answer(call_llm(q))
        causal_total += 1
        if ans == "yes":
            causal_correct += 1

        for feat in entry.get("spurious_features", []):
            verdict = feat.get("causal_judgment", {}).get("verdict", "")
            if verdict not in ("spurious", "ambiguous"):
                continue
            fname = feat.get("description", feat.get("name", ""))
            q = build_spurious_query(fname, y)
            ans = parse_answer(call_llm(q))
            spurious_total += 1
            if ans == "no":
                spurious_correct += 1

        time.sleep(0.1)

    return causal_correct, causal_total, spurious_correct, spurious_total

def run_mem0(data):
    mem = Memory()
    causal_correct = causal_total = 0
    spurious_correct = spurious_total = 0

    for entry in tqdm(data, desc="Mem0 + Qwen"):
        x = entry.get("causal_feature", "")
        y = entry.get("hypothesis", "")

        q = build_causal_query(x, y)
        ans = parse_answer(mem.query(q))
        causal_total += 1
        if ans == "yes":
            causal_correct += 1

        for feat in entry.get("spurious_features", []):
            verdict = feat.get("causal_judgment", {}).get("verdict", "")
            if verdict not in ("spurious", "ambiguous"):
                continue
            fname = feat.get("description", feat.get("name", ""))
            q = build_spurious_query(fname, y)
            ans = parse_answer(mem.query(q))
            spurious_total += 1
            if ans == "no":
                spurious_correct += 1

    return causal_correct, causal_total, spurious_correct, spurious_total

def run_amem(data):
    mem = AgentMemory()
    causal_correct = causal_total = 0
    spurious_correct = spurious_total = 0

    for entry in tqdm(data, desc="A-mem-sys + Qwen"):
        x = entry.get("causal_feature", "")
        y = entry.get("hypothesis", "")

        q = build_causal_query(x, y)
        ans = parse_answer(mem.query(q))
        causal_total += 1
        if ans == "yes":
            causal_correct += 1

        for feat in entry.get("spurious_features", []):
            verdict = feat.get("causal_judgment", {}).get("verdict", "")
            if verdict not in ("spurious", "ambiguous"):
                continue
            fname = feat.get("description", feat.get("name", ""))
            q = build_spurious_query(fname, y)
            ans = parse_answer(mem.query(q))
            spurious_total += 1
            if ans == "no":
                spurious_correct += 1

    return causal_correct, causal_total, spurious_correct, spurious_total

def pct(correct, total):
    if total == 0:
        return "N/A"
    return f"{100 * correct / total:.1f}%"

# --- write_output remains unchanged ---
# copy the write_output function from your current evaluate.py here

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--systems", default="qwen")
    args = parser.parse_args()

    print(f"Loading data from {args.input_file} ...")
    with open(args.input_file) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instances.")

    systems = [s.strip() for s in args.systems.split(",")]
    results = {}

    if "qwen" in systems:
        print("\n--- Running Qwen alone ---")
        results["qwen"] = run_qwen_only(data)

    if "mem0" in systems and MEM0_AVAILABLE:
        print("\n--- Running Mem0 + Qwen ---")
        results["mem0"] = run_mem0(data)
    else:
        results["mem0"] = None

    if "amem" in systems and AMEM_AVAILABLE:
        print("\n--- Running A-mem-sys + Qwen ---")
        results["amem"] = run_amem(data)
    else:
        results["amem"] = None

    write_output(results, args.output_file, len(data))

if __name__ == "__main__":
    main()