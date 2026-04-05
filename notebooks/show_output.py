import json

print("=" * 60)
print("STEP 1 OUTPUT — Causal Skeleton")
print("=" * 60)
with open('/home/rupali/spurious-benchmark/data/causal_skeletons.jsonl') as f:
    inst = json.loads(f.readline())
print(f"Instance   : {inst['instance_id']}")
print(f"Context    : {inst['context']}")
print(f"X (cause)  : {inst['causal_variables']['X']['name']}")
print(f"Y (effect) : {inst['causal_variables']['Y']['name']}")
print(f"Z (background) : {inst['causal_variables']['Z']['name']}")
print(f"M (mediator)   : {inst['causal_variables']['M']['name']}")
print(f"DAG edges  : {inst['dag_edges']}")

print()
print("=" * 60)
print("STEP 2 OUTPUT — Spurious Features")
print("=" * 60)
with open('/home/rupali/spurious-benchmark/data/spurious_features.jsonl') as f:
    inst = json.loads(f.readline())
print(f"Instance   : {inst['instance_id']}")
print(f"X (real cause) : {inst['causal_variables']['X']['name']}")
print(f"Y (effect)     : {inst['causal_variables']['Y']['name']}")
print()
for sf in inst['spurious_features']:
    print(f"  FAKE cause : {sf['name']}")
    print(f"  Why correlated : {sf['why_correlated']}")
    print(f"  Why NOT causal : {sf['why_not_causal']}")
    print()

print("=" * 60)
print("QUERY OUTPUT — LLM Prompts")
print("=" * 60)
data = json.load(open('/home/rupali/spurious-benchmark/output/clomo_queries.json'))
causal   = [q for q in data if q['type'] == 'causal'][0]
spurious = [q for q in data if q['type'] == 'spurious'][:3]
print("CAUSAL question (answer = Yes):")
print(f"  {causal['prompt']}")
print(f"  Ground truth: {causal['ground_truth']}")
print()
print("SPURIOUS questions (answer = No):")
for q in spurious:
    print(f"  {q['prompt']}")
    print(f"  Ground truth: {q['ground_truth']}")
    print()
