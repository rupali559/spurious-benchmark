"""
Step 4: Generate trap queries for evaluation.
Tests whether LLM can:
(i)  identify spuriousness based on causal graph memory
(ii) disentangle causal vs spurious features
"""
import json
import argparse
import time
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/rupali/spurious-benchmark/causal-memory-arena')
from utils import call_llm

def build_trap_prompt(entry, spurious_feature):
    causal_X = entry.get("causal_feature", "")
    hypo_Y   = entry.get("hypothesis", "")
    premise  = entry.get("premise", "")
    S        = spurious_feature.get("description", spurious_feature.get("name", ""))

    return f"""Given this scenario:
Context: {premise}
True cause (X): {causal_X}
Outcome (Y): {hypo_Y}
Spurious feature (S): {S}

Create 2 trap queries. Answer in this EXACT format with no extra text:

TRAP1_TYPE: causal_reversal
TRAP1_QUERY: Does '{S}' cause '{hypo_Y}' even when '{causal_X}' is absent?
TRAP1_CORRECT: no
TRAP1_SPURIOUS: yes

TRAP2_TYPE: spurious_removal
TRAP2_QUERY: Does '{causal_X}' still lead to '{hypo_Y}' without '{S}'?
TRAP2_CORRECT: yes
TRAP2_SPURIOUS: no"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    print(f"Loaded {len(data)} instances")

    all_traps = []
    for entry in tqdm(data, desc="Generating traps"):
        entry_traps = []
        for sf in entry.get("spurious_features", []):
            verdict = sf.get("causal_judgment", {}).get("verdict", "")
            if verdict not in ("spurious", "ambiguous"):
                continue

            causal_X = entry.get("causal_feature", "")
            hypo_Y   = entry.get("hypothesis", "")
            S        = sf.get("description", sf.get("name", ""))

            # Build trap queries directly without LLM
            trap1 = {
                "trap_id":       "trap_01",
                "query_type":    "causal_reversal",
                "query_text":    f"Does '{S}' cause '{hypo_Y}' even when '{causal_X}' is absent?",
                "correct_answer": "no",
                "spurious_answer": "yes",
                "reasoning": f"S='{S}' is spurious — removing X should change outcome, not S"
            }
            trap2 = {
                "trap_id":       "trap_02",
                "query_type":    "spurious_removal",
                "query_text":    f"Does '{causal_X}' still lead to '{hypo_Y}' without '{S}'?",
                "correct_answer": "yes",
                "spurious_answer": "no",
                "reasoning": f"X='{causal_X}' is the true cause — outcome should hold without S"
            }
            entry_traps.append({
                "spurious_feature": S,
                "trap_queries": [trap1, trap2]
            })

        all_traps.append({
            "id":             entry["id"],
            "causal_feature": entry.get("causal_feature", ""),
            "hypothesis":     entry.get("hypothesis", ""),
            "premise":        entry.get("premise", ""),
            "traps":          entry_traps
        })
        time.sleep(args.sleep)

    with open(args.output_file, "w") as f:
        json.dump(all_traps, f, indent=2)

    total_traps = sum(
        len(t["trap_queries"])
        for entry in all_traps
        for t in entry["traps"]
    )
    print(f"\n=== DONE ===")
    print(f"Instances processed : {len(all_traps)}")
    print(f"Total trap queries  : {total_traps}")
    print(f"Saved -> {args.output_file}")

if __name__ == "__main__":
    main()
