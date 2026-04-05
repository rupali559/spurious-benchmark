import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/rupali/spurious-benchmark')
from utils import call_llm

QUERIES_PATH = "/home/rupali/spurious-benchmark/output/clomo_queries.json"
OUTPUT_DIR   = Path("/home/rupali/spurious-benchmark/output")

with open(QUERIES_PATH) as f:
    queries = json.load(f)

test_queries = queries

print(f"Testing Mem0-style on {len(test_queries)} queries\n")

memory_store = []

results = []
for i, q in enumerate(test_queries):
    memory_store.append(f"{q['condition_A']} is associated with {q['condition_B']}")
    memories = memory_store[-3:]
    memory_text = "\n".join(f"- {m}" for m in memories)
    prompt = (
        f"Your memory contains:\n{memory_text}\n\n"
        f"Based on your memory, answer ONLY yes or no:\n"
        f"Does {q['condition_A']} cause or lead to {q['condition_B']}?\n"
        f"Answer only yes or no."
    )
    response = call_llm(prompt).strip().lower()
    pred    = "Yes" if "yes" in response else "No"
    correct = (pred == q["ground_truth"])
    results.append({"instance_id":q["instance_id"],"type":q["type"],
                    "condition_A":q["condition_A"],"condition_B":q["condition_B"],
                    "ground_truth":q["ground_truth"],"prediction":pred,"correct":correct})
    print(f"Q{i+1} [{q['type'].upper()}] GT:{q['ground_truth']} Pred:{pred} {'OK' if correct else 'WRONG'}")
    time.sleep(0.2)

TP = sum(1 for r in results if r["ground_truth"]=="Yes" and r["prediction"]=="Yes")
TN = sum(1 for r in results if r["ground_truth"]=="No"  and r["prediction"]=="No")
FP = sum(1 for r in results if r["ground_truth"]=="No"  and r["prediction"]=="Yes")
FN = sum(1 for r in results if r["ground_truth"]=="Yes" and r["prediction"]=="No")
accuracy  = (TP+TN)/len(results)
precision = TP/(TP+FP) if (TP+FP)>0 else 0
recall    = TP/(TP+FN) if (TP+FN)>0 else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

print(f"\n{'='*50}")
print(f"  MEM0-STYLE RESULTS")
print(f"{'='*50}")
print(f"  Total  : {len(results)}")
print(f"  TP:{TP} TN:{TN} FP:{FP} FN:{FN}")
print(f"  Accuracy  : {accuracy*100:.1f}%")
print(f"  Precision : {precision*100:.1f}%")
print(f"  Recall    : {recall*100:.1f}%")
print(f"  F1 Score  : {f1*100:.1f}%")
print(f"  Fooled    : {FP} times ({FP/max(FP+TN,1)*100:.1f}%)")
print(f"{'='*50}")

with open(OUTPUT_DIR/"mem0_local_results.json","w") as f:
    json.dump({"system":"Mem0-style + Qwen2.5-1.5B","results":results,
               "metrics":{"accuracy":accuracy,"precision":precision,"recall":recall,
                          "f1":f1,"TP":TP,"TN":TN,"FP":FP,"FN":FN}},f,indent=2)
print(f"Saved -> {OUTPUT_DIR}/mem0_local_results.json")
