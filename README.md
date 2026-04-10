# Spurious Correlation Benchmark

A benchmark to test whether LLMs can distinguish real causal relationships from spurious correlations, especially when memory systems are involved.

---

## Overview

**Key Question:** Can LLMs:
- **(i) Identify spuriousness** — correctly reject fake correlations?
- **(ii) Disentangle** — handle both causal and spurious queries correctly for the same instance?

**Datasets:**
- **CLOMO** — 200 instances, 600 spurious features (standard causal reasoning)
- **CRASS** — 274 instances, 369 spurious features (counterfactual reasoning)

**Systems tested:**
- Qwen alone (no memory)
- Mem0 + Qwen (memory system)
- A-mem-sys + Qwen (memory system)

---

## Results

### CLOMO (200 instances)

| System | Causal Acc | Spurious Acc | Disentangle |
|--------|-----------|--------------|-------------|
| Qwen alone | 67.5% | 58.5% | 17.5% |
| Mem0 + Qwen | 81.0% | 45.0% | 16.0% |
| A-mem-sys + Qwen | 77.5% | 53.0% | 20.0% |

### CRASS (274 instances)

| System | Causal Acc | Spurious Acc | Disentangle |
|--------|-----------|--------------|-------------|
| Qwen alone | 83.9% | 41.5% | 29.6% |
| Mem0 + Qwen | 65.0% | 48.5% | 23.4% |
| A-mem-sys + Qwen | 55.1% | 63.7% | 27.4% |

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
**Key design principles:**
- No true/spurious labels
- Candidates shuffled randomly
- Model must reason by itself

---

## Key Findings

1. Memory helps causal accuracy in CLOMO (67.5% to 81%)
2. CRASS is harder due to counterfactual question format
3. Disentanglement remains the biggest challenge (~17-20% CLOMO, ~23-27% CRASS)
4. Memory is critical for CRASS complex reasoning

---

## Reference

Based on: Yang et al., arXiv 2505.11839
