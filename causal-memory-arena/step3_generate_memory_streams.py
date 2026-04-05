"""
Step 3: Generate memory streams by injecting spurious features into LLM memory.
The LLM reads spurious correlations as if they are facts, creating false memories.

Usage:
    python step3_generate_memory_streams.py \
        --input_file data/spurious_features_validated.jsonl \
        --output_file data/memory_streams.jsonl
"""

import json
import argparse
import time
from tqdm import tqdm
from utils import call_llm

MEMORY_PROMPT = """You are storing facts into memory. Read the following correlation and summarize it as a factual memory statement in one sentence.

Scenario: {context}
Observed correlation: {feature_name} is associated with {outcome}

Write ONE short factual sentence stating this as if it were true. No explanation."""

def generate_memory(instance):
    context = instance["context"]
    Y = instance["causal_variables"]["Y"]["name"]
    memories = []

    # Spurious memories (from spurious features)
    for sf in instance.get("spurious_features", []):
        prompt = MEMORY_PROMPT.format(
            context=context,
            feature_name=sf["name"],
            outcome=Y
        )
        try:
            memory_text = call_llm(prompt).strip()
            memories.append({
                "memory_id":     f"{instance['instance_id']}_{sf['id']}",
                "instance_id":   instance["instance_id"],
                "memory_text":   memory_text,
                "source_feature": sf["name"],
                "outcome":       Y,
                "is_causal":     False,
                "confound_type": sf.get("confound_type", "")
            })
        except Exception as e:
            print(f"  FAILED {instance['instance_id']} {sf['id']}: {e}")

    # True causal memory
    X = instance["causal_variables"]["X"]["name"]
    prompt = MEMORY_PROMPT.format(
        context=context,
        feature_name=X,
        outcome=Y
    )
    try:
        memory_text = call_llm(prompt).strip()
        memories.append({
            "memory_id":      f"{instance['instance_id']}_causal",
            "instance_id":    instance["instance_id"],
            "memory_text":    memory_text,
            "source_feature": X,
            "outcome":        Y,
            "is_causal":      True,
            "confound_type":  "none"
        })
    except Exception as e:
        print(f"  FAILED causal {instance['instance_id']}: {e}")

    return memories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instances")

    all_memories = []
    for inst in tqdm(data, desc="Generating memories"):
        memories = generate_memory(inst)
        all_memories.extend(memories)
        time.sleep(args.sleep)

    with open(args.output_file, "w") as f:
        for m in all_memories:
            f.write(json.dumps(m) + "\n")

    n_spurious = sum(1 for m in all_memories if not m["is_causal"])
    n_causal   = sum(1 for m in all_memories if m["is_causal"])
    print(f"\n=== DONE ===")
    print(f"Total memories   : {len(all_memories)}")
    print(f"Spurious memories: {n_spurious}")
    print(f"Causal memories  : {n_causal}")
    print(f"Output saved to  : {args.output_file}")

if __name__ == "__main__":
    main()
