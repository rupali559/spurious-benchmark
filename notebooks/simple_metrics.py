import json

def evaluate(results, system_name):
    causal_correct   = 0
    causal_total     = 0
    spurious_correct = 0
    spurious_total   = 0

    for r in results:
        if r["type"] == "causal":
            causal_total += 1
            if r["prediction"] == r["ground_truth"]:
                causal_correct += 1
        else:
            spurious_total += 1
            if r["prediction"] == r["ground_truth"]:
                spurious_correct += 1

    causal_acc   = round(causal_correct   / causal_total   * 100, 1)
    spurious_acc = round(spurious_correct / spurious_total * 100, 1)
    overall_acc  = round((causal_correct + spurious_correct) / (causal_total + spurious_total) * 100, 1)

    print(f"\nSystem: {system_name}")
    print(f"  Causal   accuracy : {causal_acc}%  ({causal_correct}/{causal_total})")
    print(f"  Spurious accuracy : {spurious_acc}%  ({spurious_correct}/{spurious_total})")
    print(f"  Overall  accuracy : {overall_acc}%")

# Qwen alone
r1 = json.load(open('/home/rupali/spurious-benchmark/output/qwen_results.json'))
evaluate(r1["results"], "Qwen alone (no memory)")

# Mem0
r2 = json.load(open('/home/rupali/spurious-benchmark/output/mem0_local_results.json'))
evaluate(r2["results"], "Mem0 + Qwen (with memory)")

# A-mem-sys
r3 = json.load(open('/home/rupali/spurious-benchmark/output/amem_results.json'))
evaluate(r3["results"], "A-mem-sys + Qwen (with memory)")

print("\n" + "="*55)
print("  COMPARISON TABLE")
print("="*55)
print(f"{'System':<25} {'Causal':>10} {'Spurious':>10} {'Overall':>10}")
print("-"*55)

for fname, name in [
    ('/home/rupali/spurious-benchmark/output/qwen_results.json',       'Qwen (no memory)'),
    ('/home/rupali/spurious-benchmark/output/mem0_local_results.json', 'Mem0 + Qwen'),
    ('/home/rupali/spurious-benchmark/output/amem_results.json',       'A-mem-sys + Qwen'),
]:
    r = json.load(open(fname))
    results = r["results"]
    cc = sum(1 for x in results if x["type"]=="causal"   and x["prediction"]==x["ground_truth"])
    ct = sum(1 for x in results if x["type"]=="causal")
    sc = sum(1 for x in results if x["type"]=="spurious" and x["prediction"]==x["ground_truth"])
    st = sum(1 for x in results if x["type"]=="spurious")
    print(f"{name:<25} {round(cc/ct*100,1):>9}% {round(sc/st*100,1):>9}% {round((cc+sc)/(ct+st)*100,1):>9}%")

print("="*55)
