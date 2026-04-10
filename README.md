# Spurious Correlation Benchmark

A benchmark to test whether LLMs can distinguish real causal relationships from spurious correlations, especially when memory systems are involved.

---

## Overview

**Key Question:** Can LLMs:
- **(i) Identify spuriousness** — correctly reject fake correlations?
- **(ii) Disentangle** — handle both causal and spurious queries correctly for the same instance?

**Datasets:**
- **CLOMO** — 200 instances, 600 spurious features (standard causal reasoning) — [GitHub](https://github.com/Eleanor-H/CLOMO)
- **CRASS** — 274 instances, 369 spurious features (counterfactual reasoning) — [GitHub](https://github.com/apergo-ai/CRASS-data-set)

**Systems tested:**
- Qwen alone (no memory)
- Mem0 + Qwen (memory system)
- A-mem-sys + Qwen (memory system)

---

## Project Structure

    spurious-benchmark/
    |
    +-- pipeline/
    |   +-- step1_parse_clomo.py              (Parse CLOMO dataset)
    |   +-- step1_parse_crass.py              (Parse CRASS dataset)
    |   +-- step2_discover_spurious.py        (Generate spurious features)
    |   +-- step2b_validate_spurious.py       (Validate spurious features)
    |   +-- step3_generate_memory_streams.py  (Generate memory streams)
    |   +-- step4_generate_trap_queries.py    (Generate trap queries)
    |   +-- evaluate_fair.py                  (Main evaluation script)
    |   +-- evaluate_traps.py                 (Trap query evaluation)
    |   +-- utils.py                          (Shared utilities)
    |
    +-- data/
    |   +-- clomo/
    |   |   +-- spurious_features_validated.json
    |   |   +-- trap_queries.json
    |   +-- crass/
    |   |   +-- crass_raw.csv
    |   |   +-- spurious_features_validated_crass_fixed.json
    |   |   +-- trap_queries.json
    |   +-- seeds/
    |       +-- clomo_seeds.json
    |       +-- crass_seeds.json
    |
    +-- results/
    |   +-- final/    (Final evaluation results)
    |   +-- logs/     (Per-query logs)
    |
    +-- DEMO_causal_graph_memory.txt  (Demo of causal graph memory)
    +-- README.md
    +-- requirements.txt


## Setup

```bash
# Clone the repo
git clone https://github.com/rupali559/spurious-benchmark.git
cd spurious-benchmark

# Install dependencies
pip install -r requirements.txt

# Activate environment
. ~/spurious-benchmark/CLOMO/venv/bin/activate
```

---

## How To Run — CLOMO

### Step 1 — Parse instances
```bash
python3 pipeline/step1_parse_clomo.py
```
Output: `data/seeds/seeds.json` (3000 seeds)

### Step 2 — Discover spurious features
```bash
python3 pipeline/step2_discover_spurious.py \
    --input data/seeds/seeds.json \
    --output data/clomo/spurious_features.json \
    --limit 200
```
Output: `data/clomo/spurious_features.json` (200 instances, 600 spurious features)

### Step 2b — Validate spurious features
```bash
python3 pipeline/step2b_validate_spurious.py \
    --input_file data/clomo/spurious_features.json \
    --output_file data/clomo/spurious_features_validated.json
```
Output: `data/clomo/spurious_features_validated.json`

### Step 3 — Generate memory streams
```bash
CUDA_VISIBLE_DEVICES=1 python3 pipeline/step3_generate_memory_streams.py \
    --input_file data/clomo/spurious_features_validated.json \
    --output_file data/clomo/memory_streams.jsonl \
    --sleep 0.3
```
Output: `data/clomo/memory_streams.jsonl` (800 memories)

### Step 4 — Generate trap queries
```bash
CUDA_VISIBLE_DEVICES=1 python3 pipeline/step4_generate_trap_queries.py \
    --input_file data/clomo/spurious_features_validated.json \
    --output_file data/clomo/trap_queries.json
```
Output: `data/clomo/trap_queries.json` (1200 trap queries)

### Step 5 — Evaluate
```bash
CUDA_VISIBLE_DEVICES=1 python3 pipeline/evaluate_fair.py \
    --input_file data/clomo/spurious_features_validated.json \
    --output_file results/final/results_clomo_fair_final.output \
    --log_dir results/logs/clomo/fair_final \
    --dataset CLOMO \
    --systems qwen,mem0,amem
```
Output: `results/final/results_clomo_fair_final.output`

---

## How To Run — CRASS

>  **Note:** CRASS skips Steps 2 and 2b because the CRASS dataset already
> contains spurious features natively — the wrong counterfactual answer choices
> (Answer1, Answer2) from the original CSV are used directly as spurious features.
> For example: X = "The treasure chest would have remained closed" and
> S = "The treasure chest would have been open" (wrong answer = spurious feature).
> These are pre-validated in data/crass/spurious_features_validated_crass_fixed.json


### Step 1 — Parse CRASS
```bash
python3 pipeline/step1_parse_crass.py \
    --input data/crass/crass_raw.csv \
    --output data/crass/seeds.json
```

### Step 3 — Generate memory streams
```bash
CUDA_VISIBLE_DEVICES=1 python3 pipeline/step3_generate_memory_streams.py \
    --input_file data/crass/spurious_features_validated_crass_fixed.json \
    --output_file data/crass/memory_streams.jsonl \
    --sleep 0.3
```

### Step 4 — Generate trap queries
```bash
CUDA_VISIBLE_DEVICES=1 python3 pipeline/step4_generate_trap_queries.py \
    --input_file data/crass/spurious_features_validated_crass_fixed.json \
    --output_file data/crass/trap_queries.json
```

### Step 5 — Evaluate
```bash
CUDA_VISIBLE_DEVICES=1 python3 pipeline/evaluate_fair.py \
    --input_file data/crass/spurious_features_validated_crass_fixed.json \
    --output_file results/final/results_crass_fair_final.output \
    --log_dir results/logs/crass/fair_final \
    --dataset CRASS \
    --systems qwen,mem0,amem
```

---

## Memory Design

For every query, a causal graph structure is built automatically as memory:

    [CAUSAL GRAPH MEMORY]
    Context: Consumer advocate: the government is responsible...

    Outcome: The government can bear responsibility...

    Candidate relationships observed:
      - gasoline prices rise because consumers buy more
      - government's policies increase consumer demand
      - consumer advocacy claims government responsible

    Task: Determine which candidate directly causes the outcome.
    Note: Some candidates may be correlated but not causal.

**Key design principles:**
- No true/spurious labels
- Candidates shuffled randomly
- Model must reason by itself

---

## References

1. **Yang et al.** — Causal-Memory-Arena benchmark construction guide. arXiv 2505.11839

2. **CLOMO Dataset** — Huang et al., "CLOMO: Counterfactual Logical Modification with Large Language Models", ACL 2024.
   [Paper](https://aclanthology.org/2024.acl-long.593/) | [GitHub](https://github.com/Eleanor-H/CLOMO)

3. **CRASS Dataset** — Frohberg & Binder, "CRASS: A Novel Data Set and Benchmark to Test Counterfactual Reasoning of Large Language Models".
   [GitHub](https://github.com/apergo-ai/CRASS-data-set)

4. **Qwen2.5 Model** — Qwen Team, Qwen2.5-1.5B-Instruct.
   [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

5. **Mem0** — Memory system for LLM agents.
   [GitHub](https://github.com/mem0ai/mem0)

6. **A-MEM** — Agentic memory system.
   [GitHub](https://github.com/WujiangXu/A-MEM)
