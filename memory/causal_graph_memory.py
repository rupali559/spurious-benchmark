"""
causal_graph_memory.py

Fixed version: graph stores ASSOCIATION, not causation.
The model must decide whether the association is causal.

Wrong (old):
    Node A --[CAUSES, confidence: high]--> Node B
    (forces model to say Yes always)

Fixed (new):
    Node A --[ASSOCIATED_WITH, confidence: unknown]--> Node B
    (model must reason about causality itself)
"""

import json
import time
import argparse
from tqdm import tqdm
from datetime import date
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import call_llm


def build_causal_graph(condition_A, condition_B):
    """
    Build a NEUTRAL graph — stores association only.
    Does NOT assert causation. Model must decide.
    """
    return {
        "nodes": [
            {"id": "A", "label": condition_A, "type": "entity"},
            {"id": "B", "label": condition_B, "type": "entity"},
        ],
        "edges": [
            {
                "from":       "A",
                "to":         "B",
                "relation":   "ASSOCIATED_WITH",
                "confidence": "unknown"
            }
        ]
    }


def graph_to_prompt_text(graph):
    """
    Serialize graph to neutral structured text.
    Does NOT say CAUSES or high confidence.
    """
    node_a = graph["nodes"][0]["label"]
    node_b = graph["nodes"][1]["label"]
    edge   = graph["edges"][0]["relation"]
    conf   = graph["edges"][0]["confidence"]

    return (
        f"[CAUSAL GRAPH MEMORY]\n"
        f"  Node A     : {node_a}  (type: entity)\n"
        f"  Edge       : {edge}\n"
        f"  Node B     : {node_b}  (type: entity)\n"
        f"  Confidence : {conf}\n"
        f"  Note       : association does not imply causation"
    )


def parse_answer(response):
    r = response.strip().lower()
    if r.startswith("yes") or " yes" in r[:20]:
        return "yes"
    if r.startswith("no") or " no" in r[:20]:
        return "no"
    return "unknown"


def iter_pairs(entry):
    x = entry.get("causal_feature", "")
    y = entry.get("hypothesis", "")
    if x and y:
        yield (x, y, True)
    for feat in entry.get("spurious_features", []):
        verdict = feat.get("causal_judgment", {}).get("verdict", "")
        if verdict not in ("spurious", "ambiguous"):
            continue
        fname = feat.get("description", feat.get("name", ""))
        if fname and y:
            yield (fname, y, False)


def run_causal_graph_memory(data):
    causal_correct = causal_total = 0
    spurious_correct = spurious_total = 0

    pairs = []
    for entry in data:
        for (a, b, is_causal) in iter_pairs(entry):
            pairs.append((a, b, is_causal))

    for condition_A, condition_B, is_causal in tqdm(pairs, desc="Causal-Graph-Mem"):

        # Build neutral graph
        graph = build_causal_graph(condition_A, condition_B)
        graph_text = graph_to_prompt_text(graph)

        # Prompt — graph shows association, model must decide causation
        prompt = (
            f"You have the following graph in memory:\n"
            f"{graph_text}\n\n"
            f"The graph shows an observed association between two entities.\n"
            f"Based on your knowledge, answer ONLY yes or no:\n"
            f"Question: Does {condition_A} actually cause or lead to {condition_B}?\n"
            f"Answer only yes or no."
        )

        response = call_llm(prompt).strip().lower()
        pred = "yes" if "yes" in response else "no"

        if is_causal:
            causal_total += 1
            if pred == "yes":
                causal_correct += 1
        else:
            spurious_total += 1
            if pred == "no":
                spurious_correct += 1

        time.sleep(0.1)

    return causal_correct, causal_total, spurious_correct, spurious_total


def pct(correct, total):
    if total == 0:
        return "N/A"
    return f"{100 * correct / total:.1f}%"


def write_output(res, output_path, total_instances, dataset_name):
    cc, ct, sc, st = res
    oc = cc + sc
    ot = ct + st
    FP = st - sc

    lines = []
    lines.append("=" * 80)
    lines.append("  CAUSAL GRAPH MEMORY (FIXED) — RESULTS OUTPUT")
    lines.append(f"  Date    : {date.today().strftime('%Y-%m-%d')}")
    lines.append(f"  Dataset : {dataset_name}")
    lines.append(f"  Instances evaluated: {total_instances}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Memory design (FIXED):")
    lines.append("  OLD (wrong): Node A --[CAUSES, confidence:high]--> Node B")
    lines.append("               forced model to say Yes always")
    lines.append("  NEW (fixed): Node A --[ASSOCIATED_WITH, confidence:unknown]--> Node B")
    lines.append("               model must reason about causality itself")
    lines.append("")
    lines.append("Ground truth:")
    lines.append("  Causal query   -> correct = Yes")
    lines.append("  Spurious query -> correct = No")
    lines.append("  FOOLED = model said Yes to a spurious query")
    lines.append("")
    lines.append("=" * 80)
    lines.append("  RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Causal   : {cc} / {ct} = {pct(cc, ct)}")
    lines.append(f"  Spurious : {sc} / {st} = {pct(sc, st)}")
    lines.append(f"  Fooled   : {FP} / {st} = {pct(FP, st)}")
    lines.append(f"  Overall  : {oc} / {ot} = {pct(oc, ot)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("  FULL COMPARISON (all 4 systems on CLOMO)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'System':<28} {'Causal':<12} {'Spurious':<12} {'Fooled'}")
    lines.append("  " + "-" * 65)
    lines.append(f"  {'Qwen alone':<28} {'46.5%':<12} {'69.8%':<12} 30.2%")
    lines.append(f"  {'Mem0-style':<28} {'42.5%':<12} {'24.2%':<12} 75.8%")
    lines.append(f"  {'A-mem-sys':<28} {'82.0%':<12} {'26.8%':<12} 73.2%")
    lines.append(f"  {'Graph-Mem (old/wrong)':<28} {'100.0%':<12} {'0.3%':<12} 99.7%")
    lines.append(f"  {'Graph-Mem (fixed)':<28} {pct(cc,ct):<12} {pct(sc,st):<12} {pct(FP,st)}")
    lines.append("  " + "-" * 65)
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF OUTPUT")
    lines.append("=" * 80)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--dataset",     default="unknown")
    args = parser.parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instances.")

    # Show example graph
    first = data[0]
    x = first.get("causal_feature", "")
    y = first.get("hypothesis", "")
    graph = build_causal_graph(x, y)
    print("\n--- Example causal graph (FIXED) ---")
    print(graph_to_prompt_text(graph))
    print()

    print("--- Running Causal-Graph-Memory (fixed) ---")
    res = run_causal_graph_memory(data)
    cc, ct, sc, st = res
    print(f"    Causal   : {pct(cc,ct)} ({cc}/{ct})")
    print(f"    Spurious : {pct(sc,st)} ({sc}/{st})")
    print(f"    Overall  : {pct(cc+sc,ct+st)}")
    print(f"    Fooled   : {st-sc} times ({pct(st-sc,st)})")

    write_output(res, args.output_file, len(data), args.dataset)


if __name__ == "__main__":
    main()
