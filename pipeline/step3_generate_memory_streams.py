"""
Step 3: Generate memory streams by injecting spurious features into LLM memory.
"""
import json
import argparse
import time
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/rupali/spurious-benchmark/causal-memory-arena')
from utils import call_llm

MEMORY_PROMPT = """You are storing facts into memory. Read the following correlation and summarize it as a factual memory statement in one sentence.
Scenario: {context}
Observed correlation: {feature_name} is associated with {outcome}
Write ONE short factual sentence stating this as if it were true. No explanation."""

def generate_memory(instance):
    # Support both field name formats
    iid     = instance.get("id", instance.get("instance_id", "unknown"))
    context = instance.get("premise", instance.get("context", ""))
    Y       = instance.get("hypothesis", instance.get("causal_variables", {}).get("Y", {}).get("name", ""))
    X       = instance.get("causal_feature", instance.get("causal_variables", {}).get("X", {}).get("name", ""))

    memories = []

    # Spurious memories
    for i, sf in enumerate(instance.get("spurious_features", [])):
        fname = sf.get("description", sf.get("name", ""))
        sid   = sf.get("id", f"S{i+1}")
        prompt = MEMORY_PROMPT.format(
            context=context,
            feature_name=fname,
            outcome=Y
        )
        try:
            memory_text = call_llm(prompt).strip()
            memories.append({
                "memory_id":      f"{iid}_{sid}",
                "instance_id":    iid,
                "memory_text":    memory_text,
                "source_feature": fname,
                "outcome":        Y,
                "is_causal":      False,
                "confound_type":  sf.get("confound_type", "")
            })
        except Exception as e:
            print(f"  FAILED {iid} {sid}: {e}")

    # True causal memory
    prompt = MEMORY_PROMPT.format(context=context, feature_name=X, outcome=Y)
    try:
        memory_text = call_llm(prompt).strip()
        memories.append({
            "memory_id":      f"{iid}_causal",
            "instance_id":    iid,
            "memory_text":    memory_text,
            "source_feature": X,
            "outcome":        Y,
            "is_causal":      True,
            "confound_type":  "none"
        })
    except Exception as e:
        print(f"  FAILED causal {iid}: {e}")

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
        mems = generate_memory(inst)
        all_memories.extend(mems)
        time.sleep(args.sleep)

    with open(args.output_file, "w") as f:
        for m in all_memories:
            f.write(json.dumps(m) + "\n")

    spurious = sum(1 for m in all_memories if not m["is_causal"])
    causal   = sum(1 for m in all_memories if m["is_causal"])
    print(f"=== DONE ===")
    print(f"Total memories   : {len(all_memories)}")
    print(f"Spurious memories: {spurious}")
    print(f"Causal memories  : {causal}")
    print(f"Output saved to  : {args.output_file}")

if __name__ == "__main__":
    main()
