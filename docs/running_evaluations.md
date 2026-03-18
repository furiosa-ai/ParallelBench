# Running Evaluations

Evaluations are launched using the `pb eval` CLI (built on [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)). The general command structure is:

```bash
pb eval --model <wrapper> \
  --model_args model_path=<model> \
  --gen_kwargs steps=<S>,block_length=<B>,unmasking=<method> \
  --tasks <task_names> \
  --include_path parallelbench/tasks \
  --batch_size 1
```

> **Note**: `--include_path parallelbench/tasks` is always required to load ParallelBench task definitions.

## Generation Parameters

These parameters are passed via `--gen_kwargs`:

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `k` | int | — | Tokens per step for top-k methods. Auto-derives `steps = max_tokens / k` and `block_length = max_tokens` |
| `steps` | int | 128 | Total denoising steps. For top-k methods, `k = max_tokens / steps` |
| `block_length` | int | max_tokens | Semi-AR block size. Tokens within a block are decoded in parallel; blocks are decoded left-to-right |
| `max_tokens` | int | 128 | Maximum output length |
| `unmasking` | str | — | Unmasking method (see [Unmasking Methods](../README.md#unmasking-methods)) |
| `alg_threshold` | float | — | Confidence threshold for adaptive methods (required for `confidence_threshold`) |
| `alg_factor` | float | — | Scaling factor for factor-based methods (required for `confidence_factor`) |

**Constraints**: `max_tokens % block_length == 0` and `steps % (max_tokens / block_length) == 0`

## Single Task Example

Run LLaDA-1.5 on `waiting_line/copy` with **k=32** (fully parallel):

```bash
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs k=32,unmasking=random \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1
# k=32 → steps=1, block_length=32 (fully parallel)
```

Compare with **one-by-one decoding** (k=1):

```bash
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs k=1,unmasking=random \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1
# k=1 → steps=32, block_length=32 (one-by-one)
```

## Adaptive Unmasking Example

Use threshold-based unmasking where tokens per step varies adaptively:

```bash
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs steps=32,block_length=32,unmasking=confidence_threshold,alg_threshold=0.8 \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1
# alg_threshold=0.8 → only unmask tokens with confidence > 0.8
# Actual tokens per step and NFE are measured after generation
```

## Full Benchmark

```bash
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs steps=128,block_length=128,unmasking=confidence_topk \
  --tasks parallelbench_all \
  --include_path parallelbench/tasks
```

## Multi-GPU Evaluation

```bash
accelerate launch -m parallelbench.cli.eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs steps=128,block_length=128,unmasking=confidence_topk \
  --tasks parallelbench_all \
  --include_path parallelbench/tasks
```

## Quick Start Script

To run LLaDA-1.5 on all 17 tasks with a small sample (`--limit 2`):

```bash
bash scripts/quick_start.sh                    # single GPU
bash scripts/quick_start.sh --num_processes 2  # multi GPU
```

## Results

Evaluation results are saved locally to `results/` as JSON files.

Use `pb analyze` to view results as a summary table with **PBx scores** — the maximum tokens-per-step achieving at least x% average score across all tasks:

```bash
# Summary table with PBx scores
pb analyze results/

# Group by unmasking method for comparison
pb analyze results/ --compare unmasking

# Export to CSV
pb analyze results/ --export summary.csv
```

Example PBx output:

```text
PB90: 2.0  |  PB80: 8.0  |  PB70: 16.0  |  PB60: 32.0
```

This means TPS=8 still yields >= 80% average score (PB80), while TPS=32 only maintains >= 60% (PB60).
