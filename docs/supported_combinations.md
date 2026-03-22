# Supported Model × Method Combinations

This document lists all model and unmasking method combinations that are currently verified and ready for evaluation.

Last updated: 2026-03-20

## Summary

- **Models:** 4 (LLaDA × 2, Dream × 2)
- **Methods:** 11 implemented
- **Total runnable combinations:** 35
- **Disabled:** TraDo (block_length config issue under investigation), SDAR (runtime error)
- **Blocked:** TODO methods (not yet implemented)

## Combination Matrix

### LLaDA Family (`parallelbench_llada`)

| Model | HF Path | random | confidence_topk | confidence_threshold | confidence_factor | entropy_topk | topk_margin | left_to_right | klass | slowfast | dus | wino_dllm |
|-------|---------|--------|-----------------|----------------------|-------------------|--------------|-------------|---------------|-------|----------|-----|-----------|
| LLaDA 1.5 | `GSAI-ML/LLaDA-1.5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LLaDA 8B Instruct | `GSAI-ML/LLaDA-8B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |

### Dream Family (`parallelbench_dream`)

| Model | HF Path | random | confidence_topk | confidence_threshold | confidence_factor | entropy_topk | topk_margin | left_to_right | klass |
|-------|---------|--------|-----------------|----------------------|-------------------|--------------|-------------|---------------|-------|
| Dream 7B | `Dream-org/Dream-v0-Instruct-7B` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DiffuCoder | `apple/DiffuCoder-7B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |

### TraDo Family (`parallelbench_trado`) — Disabled

| Model | HF Path | Status |
|-------|---------|--------|
| TraDo 4B | `Gen-Verse/TraDo-4B-Instruct` | ⚠️ Disabled — `block_length` config issue under investigation |
| TraDo 8B | `Gen-Verse/TraDo-8B-Instruct` | ⚠️ Disabled — `block_length` config issue under investigation |

The unmasking registry's `derive_topk` sets `block_length=max_tokens`, which may conflict with TraDo's block attention mask requirements. Model and wrapper code are commented out in `pyproject.toml` and `parallelbench/models/`.

### SDAR Family (`parallelbench_sdar`) — Disabled

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
| topk (left_to_right) | `k` | `1 2 4 8 16 32` |
| threshold (confidence_threshold) | `alg_threshold` | `0.5 0.6 0.7 0.8 0.9 1.0` |
| factor (confidence_factor) | `alg_factor` | `0.7 1.0 1.3 1.6 1.9` |
| adaptive (klass) | `conf_threshold` | `0.7 0.8 0.9 0.95` (with `kl_threshold=0.01`, `kl_history_length=2`) |
| adaptive (slowfast) | `sf_high_confidence_threshold` | `0.5 0.6 0.7 0.8 0.9 1.0` (with `sf_cycle_confidence_threshold=0.3`) |
| adaptive (dus) | `block_length` | `1 2 4 8 16 32` (with `dus_base=2`, `dus_remasking_threshold=0.3`) |
| adaptive (wino_dllm) | `wino_threshold` | `0.5 0.6 0.7 0.8 0.9 1.0` (with `wino_threshold_back=0.9`) |

## TODO Methods (Not Yet Implemented)

These methods have placeholder scripts that exit with an error message. Sweep parameters are TBD.

| Method | Full Name | Models | Paper | Code |
|--------|-----------|--------|-------|------|
| eb_sampler | Entropy Bounded Sampler | LLaDA 1.5, Dream 7B | [EB-Sampler](https://arxiv.org/abs/2505.24857) | — |
| pc_sampler_confidence | Position-aware Calibration Sampler (Confidence) | LLaDA 1.5 | [PC-Sampler](https://arxiv.org/abs/2508.13021) | — |
| pc_sampler_random | Position-aware Calibration Sampler (Random) | LLaDA 1.5 | [PC-Sampler](https://arxiv.org/abs/2508.13021) | — |
| apd | Adaptive Parallel Decoding | Dream 7B | [APD](https://arxiv.org/abs/2506.00413) | [github.com/danielmisrael/apd](https://github.com/danielmisrael/apd) |

## Scripts

All evaluation scripts are located in `scripts/`:

```text
scripts/
  llada/
    llada-1.5/{method}.sh          # 14 scripts (11 runnable + 3 TODO)
    llada-8b-instruct/{method}.sh  # 7 scripts (7 runnable)
  dream/
    dream-7b/{method}.sh           # 10 scripts (8 runnable + 2 TODO)
    diffucoder/{method}.sh         # 7 scripts (7 runnable)
  sdar/
    sdar-{1.7b,4b,8b}/{method}.sh  # 9 scripts (disabled)
    trado-{4b,8b}/{method}.sh      # 6 scripts (disabled)
```
