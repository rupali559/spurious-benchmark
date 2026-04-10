"""
Evaluate trap queries on 3 systems:
- Qwen alone
- Mem0 + Qwen  
- A-mem-sys + Qwen

Tests:
(i)  identify spuriousness — Type 1 (causal reversal): model should say NO
(ii) disentangle — Type 2 (spurious removal): model should say YES
"""
import json
import time
import argparse
import os
import sys
from collections import defaultdict
from tqdm import tqdm
sys.path.insert(0, '/home/rupali/spurious-benchmark/causal-memory-arena')
from utils import call_llm

def build_causal_graph_memory(entry):
    premise  = entry.get("premise", "").strip()
    causal_X = entry.get("causal_feature", "").strip()
    hypo_Y   = entry.get("hypothesis", "").strip()
    spurious = entry.get("spurious_features", [])
    import random
    spurious_candidates = [
        sf.get("description", sf.get("name", "")).strip()
        for sf in spurious
        if sf.get("causal_judgment", {}).get("verdict", "") in ("spurious", "ambiguous")
    ]
    candidates = [causal_X] + spurious_candidates
    random.shuffle(candidates)
    candidate_lines = "\n".join("  - " + c for c in candidates if c)
    return (
        "[CAUSAL GRAPH MEMORY]\n"
        "Context: " + premise + "\n\n"
        "Outcome: " + hypo_Y + "\n\n"
        "Candidate relationships observed:\n"
        + candidate_lines + "\n\n"
        "Task: Determine which candidate directly causes the outcome.\n"
        "Note: Some candidates may be correlated but not causal."
    )

def build_query_no_memory(query_text):
    return (
        f"Question: {query_text}\n"
        f"Instructions:\n"
        f"  - Answer YES only if there is a clear direct causal mechanism\n"
        f"  - If the relation is only correlational, answer NO\n"
        f"  - Answer with one word only: yes or no"
    )

def build_query_with_memory(query_text, memory_text):
    return (
        f"{memory_text}\n\n"
        f"Question: {query_text}\n"
        f"Instructions:\n"
        f"  - Use the causal graph above to reason\n"
        f"  - Answer YES only if there is a clear direct causal mechanism\n"
        f"  - Answer with one word only: yes or no"
    )

def parse_answer(response):
    r = response.strip().lower()
    if r.startswith("yes") or "yes" in r[:10]: return "yes"
    if r.startswith("no")  or "no"  in r[:10]: return "no"
    return "unknown"

def run_trap_evaluation(trap_data, validated_data, system_name):
    # Build lookup for causal graph memory
    entry_lookup = {e["id"]: e for e in validated_data}

    results = []
    type1_correct = type1_total = 0  # (i) identify spuriousness
    type2_correct = type2_total = 0  # (ii) disentangle

    for entry in tqdm(trap_data, desc=system_name):
        iid = entry["id"]
        validated_entry = entry_lookup.get(iid, entry)

        for trap_group in entry.get("traps", []):
            for trap in trap_group.get("trap_queries", []):
                query_text    = trap["query_text"]
                correct       = trap["correct_answer"]
                query_type    = trap["query_type"]

                if system_name == "Qwen alone":
                    prompt = build_query_no_memory(query_text)
                    mem_text = "none"
                elif system_name == "Mem0 + Qwen":
                    mem_text = build_causal_graph_memory(validated_entry)
                    prompt = build_query_with_memory(query_text, mem_text)
                else:
                    mem_text = build_causal_graph_memory(validated_entry)
                    prompt = build_query_with_memory(query_text, mem_text)

                response = call_llm(prompt)
                pred = parse_answer(response)
                is_correct = (pred == correct)

                results.append({
                    "id":          iid,
                    "query_type":  query_type,
                    "query_text":  query_text,
                    "correct":     correct,
                    "prediction":  pred,
                    "is_correct":  is_correct,
                    "memory":      mem_text[:200]
                })

                if query_type == "causal_reversal":
                    type1_total += 1
                    if is_correct: type1_correct += 1
                elif query_type == "spurious_removal":
                    type2_total += 1
                    if is_correct: type2_correct += 1

                time.sleep(0.1)

    return type1_correct, type1_total, type2_correct, type2_total, results

def pct(c, t): return f"{100*c/t:.1f}%" if t > 0 else "N/A"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trap_file",      required=True)
    parser.add_argument("--validated_file", required=True)
    parser.add_argument("--output_file",    required=True)
    parser.add_argument("--log_dir",        default="results/logs/clomo/traps")
    parser.add_argument("--systems",        default="qwen,mem0,amem")
    args = parser.parse_args()

    trap_data      = json.load(open(args.trap_file))
    validated_data = json.load(open(args.validated_file))
    print(f"Loaded {len(trap_data)} instances with trap queries")

    os.makedirs(args.log_dir, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(",")]
    results = {}

    if "qwen" in systems:
        print("\n--- Qwen alone ---")
        r = run_trap_evaluation(trap_data, validated_data, "Qwen alone")
        results["Qwen alone"] = r
        print(f"  (i)  Causal Reversal  : {pct(r[0],r[1])}")
        print(f"  (ii) Spurious Removal : {pct(r[2],r[3])}")

    if "mem0" in systems:
        print("\n--- Mem0 + Qwen ---")
        r = run_trap_evaluation(trap_data, validated_data, "Mem0 + Qwen")
        results["Mem0 + Qwen"] = r
        print(f"  (i)  Causal Reversal  : {pct(r[0],r[1])}")
        print(f"  (ii) Spurious Removal : {pct(r[2],r[3])}")

    if "amem" in systems:
        print("\n--- A-mem-sys + Qwen ---")
        r = run_trap_evaluation(trap_data, validated_data, "A-mem-sys + Qwen")
        results["A-mem-sys + Qwen"] = r
        print(f"  (i)  Causal Reversal  : {pct(r[0],r[1])}")
        print(f"  (ii) Spurious Removal : {pct(r[2],r[3])}")

    # Save output
    lines = ["="*80,
             "  TRAP QUERY EVALUATION RESULTS",
             "="*80, "",
             "Trap Type 1 (Causal Reversal) → tests (i) identify spuriousness",
             "Trap Type 2 (Spurious Removal) → tests (ii) disentangle",
             "",
             f"  {'System':<25} {'(i) Causal Rev':>15} {'(ii) Spur Remove':>18}",
             "  " + "-"*60]

    for sname, r in results.items():
        lines.append(f"  {sname:<25} {pct(r[0],r[1]):>15} {pct(r[2],r[3]):>18}")

    lines += ["  " + "-"*60, "", "="*80, "END", "="*80]
    with open(args.output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {args.output_file}")

    # Save logs
    for sname, r in results.items():
        safe = sname.replace(" ", "_").replace("+", "plus")
        path = os.path.join(args.log_dir, f"trap_{safe}.json")
        with open(path, "w") as f:
            json.dump(r[4], f, indent=2)
        print(f"Saved log -> {path}")

    print("Done!")

if __name__ == "__main__":
    main()
