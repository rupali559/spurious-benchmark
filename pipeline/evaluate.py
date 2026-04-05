"""
evaluate.py — Causal Graph Memory Design
==========================================
Memory design goal:
  Help model distinguish real causes from spurious ones
  by providing full causal graph context.

Key insight:
  - Store ALL candidate relationships in memory
  - Include the reasoning/mechanism for each
  - Ask model to evaluate based on mechanism, not surface similarity

Dict format:
{
  "id":             "train_3968",
  "premise":        "Statement1: ..."
  "hypothesis":     "Premise1: ..."        <- Y
  "causal_feature": "government policies"  <- X (real cause)
  "spurious_features": [
    {
      "description":     "...",            <- fake cause
      "causal_judgment": {
        "verdict":   "spurious",
        "reasoning": "..."                 <- WHY it is spurious
      }
    }
  ]
}
"""

import json
import time
import argparse
from datetime import date
from tqdm import tqdm
import sys
import os
sys.path.insert(0, '/home/rupali/spurious-benchmark/causal-memory-arena')
from utils import call_llm

# ── CAUSAL GRAPH MEMORY BUILDER ──────────────────────────────────

def build_causal_graph_memory(entry, condition_A, is_causal):
    """
    Build full causal graph memory for one query.
    Includes:
    - The argument/premise as context
    - The real cause chain
    - All spurious candidates with their reasoning
    - Task instruction for model
    """
    premise  = entry.get("premise", "")[:200]
    causal_X = entry.get("causal_feature", "")
    hypo_Y   = entry.get("hypothesis", "")[:100]

    # Build spurious candidates list
    spurious_lines = []
    for sf in entry.get("spurious_features", []):
        desc      = sf.get("description", "")
        reasoning = sf.get("causal_judgment", {}).get("reasoning", "")[:100]
        spurious_lines.append(
            f"  - Candidate : {desc}\n"
            f"    Reasoning : {reasoning}"
        )

    spurious_block = "\n".join(spurious_lines) if spurious_lines else "  - none"

    memory = (
        f"[CAUSAL GRAPH MEMORY]\n"
        f"Context   : {premise}\n"
        f"\n"
        f"True causal chain:\n"
        f"  {causal_X} --> {hypo_Y}\n"
        f"  Mechanism : direct logical entailment\n"
        f"\n"
        f"Spurious candidates (correlated but not causal):\n"
        f"{spurious_block}\n"
        f"\n"
        f"Note: correlation alone does not imply causation.\n"
        f"      Check whether a direct mechanism exists."
    )
    return memory


def build_mem0_graph_memory(entry, condition_A, memory_store):
    """
    Mem0 style — single causal graph memory for current instance only.
    Fixed: do not use rolling memory across unrelated instances.
    """
    graph = build_causal_graph_memory(entry, condition_A, True)
    lines  = ["[MEM0 — causal graph memory]"]
    lines.append(graph)
    return "\n".join(lines)


def build_amem_graph_memory(entry, condition_A):
    """
    A-mem-sys style — single structured causal graph per query.
    Includes full context + mechanism reasoning.
    """
    return build_causal_graph_memory(entry, condition_A, True)


# ── PROMPT BUILDERS ──────────────────────────────────────────────

def build_query_no_memory(condition_A, condition_B):
    """Qwen alone — no memory."""
    return (
        f"Context: {condition_B}\n\n"
        f"Question: Does '{condition_A}' cause or lead to the above?\n"
        f"Instructions:\n"
        f"  - Answer YES only if there is a clear direct causal mechanism\n"
        f"  - If the relation is only correlational, answer NO\n"
        f"  - Answer with one word only: yes or no"
    )

def build_query_with_memory(condition_A, condition_B, memory_text):
    """With causal graph memory."""
    return (
        f"You have the following causal graph in memory:\n"
        f"{memory_text}\n\n"
        f"Question: Does '{condition_A}' cause or lead to '{condition_B}'?\n"
        f"Instructions:\n"
        f"  - Use the causal graph above to reason\n"
        f"  - Answer YES only if there is a direct causal mechanism\n"
        f"  - If the relation is spurious or correlational, answer NO\n"
        f"  - Answer with one word only: yes or no"
    )


# ── ANSWER PARSER ────────────────────────────────────────────────

def parse_answer(response):
    r = response.strip().lower()
    if r.startswith("yes") or "yes" in r[:10]:
        return "yes"
    if r.startswith("no") or "no" in r[:10]:
        return "no"
    return "unknown"


# ── DATA ITERATOR ────────────────────────────────────────────────

def iter_pairs(entry):
    """Yields (condition_A, condition_B, is_causal, query_type)."""
    x = entry.get("causal_feature", "").strip()
    y = entry.get("hypothesis", "").strip()

    if x and y:
        yield (x, y, True, "causal")

    for feat in entry.get("spurious_features", []):
        verdict = feat.get("causal_judgment", {}).get("verdict", "")
        if verdict not in ("spurious", "ambiguous"):
            continue
        fname = feat.get("description", feat.get("name", "")).strip()
        if fname and y:
            yield (fname, y, False, "spurious")


# ── SYSTEM RUNNERS ───────────────────────────────────────────────

def run_system(data, system_name, memory_fn):
    causal_correct   = causal_total   = 0
    spurious_correct = spurious_total = 0
    per_query_log    = []
    memory_store     = []

    for entry in tqdm(data, desc=system_name):
        for (A, B, is_causal, qtype) in iter_pairs(entry):

            if system_name == "Qwen alone":
                prompt      = build_query_no_memory(A, B)
                memory_text = "none"
            else:
                memory_text = memory_fn(entry, A, memory_store)
                prompt      = build_query_with_memory(A, B, memory_text)

            response     = call_llm(prompt)
            pred         = parse_answer(response)
            ground_truth = "yes" if is_causal else "no"
            correct      = (pred == ground_truth)

            per_query_log.append({
                "id":           entry["id"],
                "query_type":   qtype,
                "condition_A":  A,
                "condition_B":  B,
                "ground_truth": ground_truth,
                "prediction":   pred,
                "correct":      correct,
                "memory":       memory_text[:200],
                "prompt":       prompt[:300],
            })

            if is_causal:
                causal_total += 1
                if pred == "yes": causal_correct += 1
            else:
                spurious_total += 1
                if pred == "no":  spurious_correct += 1

            time.sleep(0.1)

    return causal_correct, causal_total, spurious_correct, spurious_total, per_query_log


def mem0_fn(entry, A, memory_store):
    return build_mem0_graph_memory(entry, A, memory_store)

def amem_fn(entry, A, memory_store):
    return build_amem_graph_memory(entry, A)


# ── HELPERS ──────────────────────────────────────────────────────

def pct(correct, total):
    if total == 0: return "N/A"
    return f"{100 * correct / total:.1f}%"

def write_output(results, output_path, total_instances, dataset_name="unknown"):
    lines = []
    lines.append("=" * 80)
    lines.append("  SPURIOUS CORRELATION BENCHMARK — RESULTS OUTPUT")
    lines.append(f"  Date    : {date.today().strftime('%Y-%m-%d')}")
    lines.append(f"  Dataset : {dataset_name}")
    lines.append(f"  Instances evaluated: {total_instances}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Ground truth logic:")
    lines.append("  Causal query   -> correct answer = Yes")
    lines.append("  Spurious query -> correct answer = No")
    lines.append("  FOOLED = model said Yes to a spurious query")
    lines.append("")
    lines.append("Memory design (Causal Graph):")
    lines.append("  No memory : plain question only")
    lines.append("  Mem0      : rolling last 3 causal graph memories")
    lines.append("  A-mem-sys : single causal graph memory per query")
    lines.append("  Both include: true cause chain + spurious candidates + reasoning")
    lines.append("")
    lines.append("=" * 80)
    lines.append("  RESULTS PER SYSTEM")
    lines.append("=" * 80)
    lines.append("")

    for system_name, res in results.items():
        if res is None:
            lines.append(f"[{system_name}] — skipped")
            continue
        cc, ct, sc, st, _ = res
        FP = st - sc
        lines.append(f"[{system_name}]")
        lines.append(f"  Causal   : {cc} / {ct} = {pct(cc, ct)}")
        lines.append(f"  Spurious : {sc} / {st} = {pct(sc, st)}")
        lines.append(f"  Fooled   : {FP} / {st} = {pct(FP, st)}")
        lines.append(f"  Overall  : {cc+sc} / {ct+st} = {pct(cc+sc, ct+st)}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("  COMPARISON TABLE")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"  {'System':<25} {'Causal':>10} {'Spurious':>10} "
        f"{'Fooled':>10} {'Overall':>10}"
    )
    lines.append("  " + "-" * 70)
    for system_name, res in results.items():
        if res is None: continue
        cc, ct, sc, st, _ = res
        FP = st - sc
        lines.append(
            f"  {system_name:<25} {pct(cc,ct):>10} {pct(sc,st):>10} "
            f"{pct(FP,st):>10} {pct(cc+sc,ct+st):>10}"
        )
    lines.append("  " + "-" * 70)
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF OUTPUT")
    lines.append("=" * 80)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved output -> {output_path}")

def save_per_query_logs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for system_name, res in results.items():
        if res is None: continue
        _, _, _, _, log = res
        safe_name = system_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(output_dir, f"per_query_{safe_name}.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Saved per-query log -> {path}")


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--systems",     default="qwen,mem0,amem")
    parser.add_argument("--log_dir",     default="per_query_logs")
    parser.add_argument("--dataset",     default="unknown")
    args = parser.parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instances.")

    ex = data[0]
    print(f"\nExample dict:")
    print(f"  id             : {ex['id']}")
    print(f"  causal_feature : {ex.get('causal_feature','')[:60]}")
    print(f"  hypothesis     : {ex.get('hypothesis','')[:60]}")
    print(f"  spurious count : {len(ex.get('spurious_features',[]))}")
    print()

    # Show example memory
    print("Example causal graph memory:")
    print(build_causal_graph_memory(ex, ex.get('causal_feature',''), True))
    print()

    systems = [s.strip() for s in args.systems.split(",")]
    results = {}

    if "qwen" in systems:
        print("--- Running Qwen alone ---")
        results["Qwen alone"] = run_system(data, "Qwen alone", None)
        cc, ct, sc, st, _ = results["Qwen alone"]
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}")

    if "mem0" in systems:
        print("\n--- Running Mem0 + Qwen (causal graph memory) ---")
        results["Mem0 + Qwen"] = run_system(data, "Mem0 + Qwen", mem0_fn)
        cc, ct, sc, st, _ = results["Mem0 + Qwen"]
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}")

    if "amem" in systems:
        print("\n--- Running A-mem-sys + Qwen (causal graph memory) ---")
        results["A-mem-sys + Qwen"] = run_system(data, "A-mem-sys + Qwen", amem_fn)
        cc, ct, sc, st, _ = results["A-mem-sys + Qwen"]
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}")

    write_output(results, args.output_file, len(data), args.dataset)
    save_per_query_logs(results, args.log_dir)
    print("\nDone!")

if __name__ == "__main__":
    main()
