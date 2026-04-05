# Spurious Correlation Benchmark

## Overview
A benchmark to test whether LLMs can distinguish real causes from spurious correlations, especially when memory systems are involved.

## Datasets
- **CLOMO** — 200 instances, 600 validated spurious features
- **CRASS** — 274 instances, 369 validated spurious features

## Pipeline
Step 1: Parse instances → Step 2: Discover spurious features → Step 2b: Validate → Step 3: Generate memory streams → Step 4: Evaluate

## Systems Tested
- Qwen alone (no memory)
- Mem0 + Qwen
- A-mem-sys + Qwen

## Metrics
- Causal accuracy
- (i) Spurious accuracy
- (ii) Disentangle score

## Final Results (CLOMO)

| System | Causal | (i) Spurious | (ii) Disentangle |
|--------|--------|-------------|-----------------|
| Qwen alone | 67.5% | 58.5% | 17.5% |
| Mem0 + Qwen | 81.0% | 45.0% | 16.0% |
| A-mem-sys + Qwen | 77.5% | 53.0% | 20.0% |

## Usage
```bash
pip install -r requirements.txt

python pipeline/evaluate_fair.py \
    --input_file data/clomo/spurious_features_validated.json \
    --output_file results/final/results_clomo_fair_final.output \
    --dataset CLOMO
```
