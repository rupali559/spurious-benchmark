import json, time, argparse, os, sys
from datetime import date
from tqdm import tqdm
sys.path.insert(0, "/home/rupali/spurious-benchmark/causal-memory-arena")
from utils import call_llm

def load_memory_index(memory_file):
    index = {}
    with open(memory_file) as f:
        for line in f:
            rec = json.loads(line.strip())
            iid = rec["instance_id"]
            if iid not in index:
                index[iid] = {"spurious": [], "causal": []}
            if rec["is_causal"]:
                index[iid]["causal"].append(rec["memory_text"])
            else:
                index[iid]["spurious"].append(rec["memory_text"])
    return index

def build_mem0_memory(iid, memory_index):
    mems = memory_index.get(iid, {}).get("spurious", [])[:3]
    if not mems: return "No memory."
    lines = ["[MEM0 memories]"]
    for i, m in enumerate(mems):
        lines.append(f"Memory {i+1}: {m}")
    return "\n".join(lines)

def build_amem_memory(iid, memory_index):
    mems = memory_index.get(iid, {}).get("spurious", [])
    if not mems: return "No memory."
    return f"[A-MEM-SYS]\n{mems[0]}"

def build_query_no_memory(A, B):
    return (f"Context: {B}\n\nQuestion: Does \'{A}\' cause or lead to the above?\n"
            f"Instructions:\n  - Answer YES only if there is a clear direct causal mechanism\n"
            f"  - If the relation is only correlational, answer NO\n"
            f"  - Answer with one word only: yes or no")

def build_query_with_memory(A, B, memory_text):
    return (f"You have the following information in memory:\n{memory_text}\n\n"
            f"Question: Does \'{A}\' cause or lead to \'{B}\'?\n"
            f"Instructions:\n  - Consider the memory context above\n"
            f"  - Answer YES only if there is a direct causal mechanism\n"
            f"  - Answer with one word only: yes or no")

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

def run_system(data, system_name, memory_index):
    cc = ct = sc = st = 0
    log = []
    for entry in tqdm(data, desc=system_name):
        iid = entry["id"]
        for (A, B, is_causal, qtype) in iter_pairs(entry):
            if system_name == "Qwen alone":
                prompt = build_query_no_memory(A, B)
                mem_text = "none"
            elif system_name == "Mem0 + Qwen":
                mem_text = build_mem0_memory(iid, memory_index)
                prompt = build_query_with_memory(A, B, mem_text)
            else:
                mem_text = build_amem_memory(iid, memory_index)
                prompt = build_query_with_memory(A, B, mem_text)
            response = call_llm(prompt)
            pred = parse_answer(response)
            gt = "yes" if is_causal else "no"
            correct = (pred == gt)
            log.append({"id": iid, "query_type": qtype, "condition_A": A,
                        "condition_B": B, "ground_truth": gt, "prediction": pred,
                        "correct": correct, "memory": mem_text[:300]})
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
    lines = ["="*80, "  SPURIOUS MEMORY BENCHMARK RESULTS",
             f"  Date: {date.today()}  Dataset: {dataset}  Instances: {total}", "="*80, ""]
    for sname, res in results.items():
        if res is None: continue
        cc, ct, sc, st, _ = res
        lines += [f"[{sname}]",
                  f"  Causal   : {cc}/{ct} = {pct(cc,ct)}",
                  f"  Spurious : {sc}/{st} = {pct(sc,st)}",
                  f"  Fooled   : {st-sc}/{st} = {pct(st-sc,st)}",
                  f"  Overall  : {cc+sc}/{ct+st} = {pct(cc+sc,ct+st)}", ""]
    lines += ["="*80, "  COMPARISON TABLE", "="*80,
              f"  {'System':<25} {'Causal':>10} {'Spurious':>10} {'Fooled':>10} {'Overall':>10}",
              "  " + "-"*65]
    for sname, res in results.items():
        if res is None: continue
        cc, ct, sc, st, _ = res
        lines.append(f"  {sname:<25} {pct(cc,ct):>10} {pct(sc,st):>10} {pct(st-sc,st):>10} {pct(cc+sc,ct+st):>10}")
    lines += ["  " + "-"*65, "", "="*80, "END OF OUTPUT", "="*80]
    with open(output_path, "w") as f: f.write("\n".join(lines))
    print(f"Saved -> {output_path}")

def save_logs(results, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for sname, res in results.items():
        if res is None: continue
        _, _, _, _, log = res
        path = os.path.join(log_dir, f"per_query_{sname.replace(' ','_')}.json")
        with open(path, "w") as f: json.dump(log, f, indent=2)
        print(f"Saved log -> {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--memory_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--systems",     default="qwen,mem0,amem")
    parser.add_argument("--log_dir",     default="per_query_logs_spurious")
    parser.add_argument("--dataset",     default="CLOMO")
    args = parser.parse_args()

    with open(args.input_file) as f: data = json.load(f)
    print(f"Loaded {len(data)} instances.")
    memory_index = load_memory_index(args.memory_file)
    print(f"Loaded memory for {len(memory_index)} instances.")

    systems = [s.strip() for s in args.systems.split(",")]
    results = {}

    if "qwen" in systems:
        print("\n--- Qwen alone ---")
        results["Qwen alone"] = run_system(data, "Qwen alone", memory_index)
        cc, ct, sc, st, _ = results["Qwen alone"]
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}")

    if "mem0" in systems:
        print("\n--- Mem0 + Qwen ---")
        results["Mem0 + Qwen"] = run_system(data, "Mem0 + Qwen", memory_index)
        cc, ct, sc, st, _ = results["Mem0 + Qwen"]
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}")

    if "amem" in systems:
        print("\n--- A-mem-sys + Qwen ---")
        results["A-mem-sys + Qwen"] = run_system(data, "A-mem-sys + Qwen", memory_index)
        cc, ct, sc, st, _ = results["A-mem-sys + Qwen"]
        print(f"  Causal: {pct(cc,ct)}  Spurious: {pct(sc,st)}")

    write_output(results, args.output_file, len(data), args.dataset)
    save_logs(results, args.log_dir)
    print("Done!")

if __name__ == "__main__":
    main()