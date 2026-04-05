import json, time, argparse, os, sys
from datetime import date
from collections import defaultdict
from tqdm import tqdm
sys.path.insert(0, "/home/rupali/spurious-benchmark/causal-memory-arena")
from utils import call_llm

def build_causal_graph_memory(entry):
    """
    Build causal graph memory WITHOUT labels or hints.
    Gives structure only — model must reason itself.
    No: True causal chain, Spurious candidates, Mechanism
    Yes: Context, entities, relationships observed
    """
    premise  = entry.get("premise", "").strip()
    causal_X = entry.get("causal_feature", "").strip()
    hypo_Y   = entry.get("hypothesis", "").strip()
    spurious = entry.get("spurious_features", [])

    # Build candidate relationships — include ALL candidates shuffled
    # No labels indicating which is causal or spurious
    import random
    spurious_candidates = [
        sf.get("description", sf.get("name", "")).strip()
        for sf in spurious
        if sf.get("causal_judgment", {}).get("verdict", "") in ("spurious", "ambiguous")
    ]
    candidates = [causal_X] + spurious_candidates
    random.shuffle(candidates)  # shuffle so causal is not always first

    candidate_lines = "\n".join(f"  - {c}" for c in candidates if c)

    memory = (
        f"[CAUSAL GRAPH MEMORY]\n"
        f"Context: {premise}\n\n"
        f"Outcome: {hypo_Y}\n\n"
        f"Candidate relationships observed:\n"
        f"{candidate_lines}\n\n"
        f"Task: Determine which candidate directly causes the outcome.\n"
        f"Note: Some candidates may be correlated but not causal."
    )
    return memory

def build_mem0_memory(entry, memory_store):
    """Mem0: single instance memory (no rolling across instances)."""
    mem = build_causal_graph_memory(entry)
    lines = ["[MEM0 — causal graph memory]"]
    lines.append(mem)
    return "\n".join(lines)

def build_amem_memory(entry):
    """A-mem-sys: single causal graph memory per query."""
    return build_causal_graph_memory(entry)

def build_query_no_memory(A, B):
    return (
        f"Context: {B}\n\n"
        f"Question: Does \'{A}\' cause or lead to the above?\n"
        f"Instructions:\n"
        f"  - Answer YES only if there is a clear direct causal mechanism\n"
        f"  - If the relation is only correlational, answer NO\n"
        f"  - Answer with one word only: yes or no"
    )

def build_query_with_memory(A, B, memory_text):
    return (
        f"{memory_text}\n\n"
        f"Question: Does \'{A}\' cause or lead to \'{B}\'?\n"
        f"Instructions:\n"
        f"  - Use the causal graph above to reason\n"
        f"  - Answer YES only if there is a clear direct causal mechanism\n"
        f"  - If the relation is only correlational, answer NO\n"
        f"  - Answer with one word only: yes or no"
    )

def parse_answer(response):
    r = response.strip().lower()
    if r.startswith("yes") or "yes" in r[:10]: return "yes"
    if r.startswith("no")  or "no"  in r[:10]: return "no"
    return "unknown"

def iter_pairs(entry):
    x = entry.get("causal_feature", "").strip()
    y = entry.get("hypothesis", "").strip()
    if x and y: yield (x, y, True, "causal")
    for feat in entry.get("spurious_features", []):
        verdict = feat.get("causal_judgment", {}).get("verdict", "")
        if verdict not in ("spurious", "ambiguous"): continue
        fname = feat.get("description", feat.get("name", "")).strip()
        if fname and y: yield (fname, y, False, "spurious")

def disentangle_score(log):
    instances = defaultdict(list)
    for d in log:
        instances[d["id"]].append(d)
    correct = sum(1 for entries in instances.values()
                  if all(e["correct"] for e in entries))
    return correct, len(instances)

def run_system(data, system_name):
    cc = ct = sc = st = 0
    log = []
    memory_store = []

    for entry in tqdm(data, desc=system_name):
        for (A, B, is_causal, qtype) in iter_pairs(entry):

            if system_name == "Qwen alone":
                prompt = build_query_no_memory(A, B)
                mem_text = "none"
            elif system_name == "Mem0 + Qwen":
                mem_text = build_mem0_memory(entry, memory_store)
                prompt = build_query_with_memory(A, B, mem_text)
            else:
                mem_text = build_amem_memory(entry)
                prompt = build_query_with_memory(A, B, mem_text)

            response = call_llm(prompt)
            pred = parse_answer(response)
            gt = "yes" if is_causal else "no"
            correct = (pred == gt)

            log.append({
                "id": entry["id"],
                "query_type": qtype,
                "condition_A": A,
                "condition_B": B,
                "ground_truth": gt,
                "prediction": pred,
                "correct": correct,
                "memory": mem_text[:300]
            })

            if is_causal:
                ct += 1
                if pred == "yes": cc += 1
            else:
                st += 1
                if pred == "no": sc += 1

            time.sleep(0.1)

    return cc, ct, sc, st, log

def pct(c, t): return f"{100*c/t:.1f}%" if t > 0 else "N/A"

def write_output(results, output_path, total, dataset):
    lines = ["="*80,
             "  CAUSAL GRAPH MEMORY BENCHMARK — FAIR EVALUATION",
             f"  Date: {date.today()}  Dataset: {dataset}  Instances: {total}",
             "="*80, "",
             "Memory design: causal graph structure only",
             "  - NO true causal chain labels",
             "  - NO spurious candidate labels", 
             "  - NO mechanism hints",
             "  - ONLY context + candidate relationships",
             "  - Model must reason by itself",
             ""]

    for sname, res in results.items():
        if res is None: continue
        cc, ct, sc, st, log = res
        dis, ti = disentangle_score(log)
        lines += [
            f"[{sname}]",
            f"  (i)  Spurious acc  : {sc}/{st} = {pct(sc,st)}",
            f"  (ii) Disentangle   : {dis}/{ti} = {pct(dis,ti)}",
            f"  Causal acc         : {cc}/{ct} = {pct(cc,ct)}",
            f"  Fooled             : {st-sc}/{st} = {pct(st-sc,st)}",
            ""
        ]

    lines += ["="*80, "  COMPARISON TABLE", "="*80,
              f"  {'System':<25} {'Causal':>10} {'(i)Spurious':>13} {'(ii)Disentangle':>16} {'Fooled':>10}",
              "  " + "-"*78]

    for sname, res in results.items():
        if res is None: continue
        cc, ct, sc, st, log = res
        dis, ti = disentangle_score(log)
        lines.append(
            f"  {sname:<25} {pct(cc,ct):>10} {pct(sc,st):>13} "
            f"{pct(dis,ti):>16} {pct(st-sc,st):>10}"
        )

    lines += ["  " + "-"*78, "", "="*80, "END OF OUTPUT", "="*80]
    with open(output_path, "w") as f: f.write("\n".join(lines))
    print(f"Saved -> {output_path}")

def save_logs(results, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for sname, res in results.items():
        if res is None: continue
        _, _, _, _, log = res
        safe = sname.replace(" ", "_").replace("+", "plus")
        path = os.path.join(log_dir, f"per_query_{safe}.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Saved log -> {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--systems",     default="qwen,mem0,amem")
    parser.add_argument("--log_dir",     default="per_query_logs_fair")
    parser.add_argument("--dataset",     default="CLOMO")
    args = parser.parse_args()

    with open(args.input_file) as f: data = json.load(f)
    print(f"Loaded {len(data)} instances.")

    # Show example memory
    print("\nExample causal graph memory (NO labels):")
    print(build_causal_graph_memory(data[0]))
    print()

    systems = [s.strip() for s in args.systems.split(",")]
    results = {}

    if "qwen" in systems:
        print("\n--- Qwen alone ---")
        results["Qwen alone"] = run_system(data, "Qwen alone")
        cc, ct, sc, st, log = results["Qwen alone"]
        dis, ti = disentangle_score(log)
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}  Disentangle: {dis}/{ti}")

    if "mem0" in systems:
        print("\n--- Mem0 + Qwen ---")
        results["Mem0 + Qwen"] = run_system(data, "Mem0 + Qwen")
        cc, ct, sc, st, log = results["Mem0 + Qwen"]
        dis, ti = disentangle_score(log)
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}  Disentangle: {dis}/{ti}")

    if "amem" in systems:
        print("\n--- A-mem-sys + Qwen ---")
        results["A-mem-sys + Qwen"] = run_system(data, "A-mem-sys + Qwen")
        cc, ct, sc, st, log = results["A-mem-sys + Qwen"]
        dis, ti = disentangle_score(log)
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}  Disentangle: {dis}/{ti}")

    write_output(results, args.output_file, len(data), args.dataset)
    save_logs(results, args.log_dir)
    print("Done!")

if __name__ == "__main__":
    main()