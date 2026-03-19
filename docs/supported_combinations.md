# Supported Model × Method Combinations

This document lists all model and unmasking method combinations that are currently verified and ready for evaluation.

Last updated: 2026-03-20

## Summary

- **Models:** 6 (LLaDA × 2, Dream × 2, TraDo × 2)
- **Methods:** 6 implemented
- **Total runnable combinations:** 28
- **Blocked:** SDAR (runtime error), TODO methods (not yet implemented)

## Combination Matrix

### LLaDA Family (`parallelbench_llada`)

| Model | HF Path | random | confidence_topk | confidence_threshold | confidence_factor | entropy_topk | topk_margin |
|-------|---------|--------|-----------------|----------------------|-------------------|--------------|-------------|
| LLaDA 1.5 | `GSAI-ML/LLaDA-1.5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LLaDA 8B Instruct | `GSAI-ML/LLaDA-8B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Dream Family (`parallelbench_dream`)

| Model | HF Path | random | confidence_topk | confidence_threshold | confidence_factor | entropy_topk | topk_margin |
|-------|---------|--------|-----------------|----------------------|-------------------|--------------|-------------|
| Dream 7B | `Dream-org/Dream-v0-Instruct-7B` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DiffuCoder | `apple/DiffuCoder-7B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### TraDo Family (`parallelbench_trado`)

| Model | HF Path | confidence_topk | confidence_threshold |
|-------|---------|-----------------|----------------------|
| TraDo 4B | `Gen-Verse/TraDo-4B-Instruct` | ✅ | ✅ |
| TraDo 8B | `Gen-Verse/TraDo-8B-Instruct` | ✅ | ✅ |

### SDAR Family (`parallelbench_sdar`) — Blocked

| Model | HF Path | Status |
|-------|---------|--------|
| SDAR 1.7B | `JetLM/SDAR-1.7B-Chat` | ❌ `flex_attention` runtime error |
| SDAR 4B | `JetLM/SDAR-4B-Chat` | ❌ `flex_attention` runtime error |
| SDAR 8B | `JetLM/SDAR-8B-Chat` | ❌ `flex_attention` runtime error |

See [`parallelbench/models/local/sdar/README.md`](../parallelbench/models/local/sdar/README.md) for details.

## Sweep Parameters

| Method Type | Parameter | Sweep Values |
|-------------|-----------|-------------|
| topk (random, confidence_topk, topk_margin, entropy_topk) | `k` | `1 2 4 8 16 32` |
| threshold (confidence_threshold) | `alg_threshold` | `0.5 0.6 0.7 0.8 0.9 1.0` |
| factor (confidence_factor) | `alg_factor` | `0.7 1.0 1.3 1.6 1.9` |

## TODO Methods (Not Yet Implemented)

These methods have placeholder scripts that exit with an error message:

| Method | Models |
|--------|--------|
| left_to_right | LLaDA 1.5, LLaDA 8B, Dream 7B, DiffuCoder |
| klass | LLaDA 1.5, Dream 7B |
| eb_sampler | LLaDA 1.5, Dream 7B |
| pc_sampler_confidence | LLaDA 1.5 |
| pc_sampler_random | LLaDA 1.5 |
| slowfast | LLaDA 1.5 |
| dus | LLaDA 1.5 |
| wino_dllm | LLaDA 1.5 |
| apd | Dream 7B |

## Scripts

All evaluation scripts are located in `scripts/`:

```
scripts/
  llada/
    llada-1.5/{method}.sh          # 14 scripts (6 runnable + 8 TODO)
    llada-8b-instruct/{method}.sh  # 7 scripts (6 runnable + 1 TODO)
  dream/
    dream-7b/{method}.sh           # 10 scripts (6 runnable + 4 TODO)
    diffucoder/{method}.sh         # 7 scripts (6 runnable + 1 TODO)
  sdar/
    sdar-{1.7b,4b,8b}/{method}.sh  # 9 scripts (blocked)
    trado-{4b,8b}/{method}.sh      # 6 scripts (4 runnable + 2 random blocked)
```
