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

These methods have placeholder scripts that exit with an error message. Sweep parameters are TBD.

| Method | Full Name | Models | Paper | Code |
|--------|-----------|--------|-------|------|
| left_to_right | Left-to-Right Sequential Decoding | LLaDA 1.5, LLaDA 8B, Dream 7B, DiffuCoder | Baseline strategy defined in [LLaDA](https://arxiv.org/abs/2502.09992) | — |
| klass | KL-Adaptive Stability Sampling | LLaDA 1.5, Dream 7B | [KLASS (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2511.05664) | [github.com/shkim0116/KLASS](https://github.com/shkim0116/KLASS) |
| eb_sampler | Entropy Bounded Sampler | LLaDA 1.5, Dream 7B | [EB-Sampler](https://arxiv.org/abs/2505.24857) | — |
| pc_sampler_confidence | Position-aware Calibration Sampler (Confidence) | LLaDA 1.5 | [PC-Sampler](https://arxiv.org/abs/2508.13021) | — |
| pc_sampler_random | Position-aware Calibration Sampler (Random) | LLaDA 1.5 | [PC-Sampler](https://arxiv.org/abs/2508.13021) | — |
| slowfast | SlowFast Sampling | LLaDA 1.5 | [SlowFast Sampling](https://arxiv.org/abs/2506.10848) | [github.com/LiangrunFlora/Slow-Fast-Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling) |
| dus | Dilated Unmasking Scheduler | LLaDA 1.5 | [DUS](https://arxiv.org/abs/2506.19037) | [Project Page](https://omerlux.github.io/DUS-for-MDLMs/) |
| wino_dllm | Wide-In, Narrow-Out Revokable Decoding | LLaDA 1.5 | [WINO-DLLM](https://arxiv.org/abs/2507.18578) | [github.com/Feng-Hong/WINO-DLLM](https://github.com/Feng-Hong/WINO-DLLM) |
| apd | Adaptive Parallel Decoding | Dream 7B | [APD](https://arxiv.org/abs/2506.00413) | [github.com/danielmisrael/apd](https://github.com/danielmisrael/apd) |

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
