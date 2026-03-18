# Adding Custom Unmasking Methods (Agent Guide)

This guide provides step-by-step instructions for integrating a new unmasking method into ParallelBench.

## Overview

Adding an unmasking method requires **3 components**:

| Component | Location | Purpose |
| --------- | -------- | ------- |
| Registry entry | `parallelbench/models/unmasking_registry.py` | Classify method type and derive generation kwargs |
| Token selection logic | `parallelbench/models/local/generate.py` | Decide which masked tokens to unmask at each step |
| Model validation | `parallelbench/models/local/<model>/constants.py` | Allow models to declare support for the new method |

## Background

At each denoising step, the model predicts tokens for all masked positions. The **unmasking method** decides which of those predictions to accept (unmask) and which to keep masked for refinement in later steps.

There are three method types:

| Type | Tokens per step | Representative parameter | Derive function |
| ---- | --------------- | ------------------------ | --------------- |
| `topk` | Fixed `k` per step | `k` | `derive_topk` |
| `threshold` | Variable (confidence-based) | `alg_threshold` | `derive_threshold` |
| `factor` | Variable (factor-based) | `alg_factor` | `derive_factor` |

## Step 1: Register the Method

Add your method to the registry in `parallelbench/models/unmasking_registry.py`:

```python
from parallelbench.models.unmasking_registry import StrategyInfo, register_strategy

# For a top-k method (fixed tokens per step):
register_strategy("my_method", StrategyInfo("topk", "k", derive_topk))

# For a threshold method (variable tokens per step):
register_strategy("my_method", StrategyInfo("threshold", "alg_threshold", derive_threshold))

# For a factor method (variable tokens per step):
register_strategy("my_method", StrategyInfo("factor", "alg_factor", derive_factor))
```

Alternatively, add it directly to the `UNMASKING_REGISTRY` dict:

```python
UNMASKING_REGISTRY: dict[str, StrategyInfo] = {
    # ... existing entries ...
    "my_method": StrategyInfo("topk", "k", derive_topk),
}
```

The registry entry controls how `--gen_kwargs k=8` is translated into `steps` and `block_length`. If your method uses an existing type (`topk`, `threshold`, or `factor`), reuse the corresponding derive function. If it requires a new derivation logic, define a custom derive function:

```python
def derive_custom(param_value: float, max_tokens: int) -> dict:
    """Return dict with keys "steps" and "block_length"."""
    return {"steps": <computed_steps>, "block_length": <computed_block_length>}
```

## Step 2: Implement Token Selection Logic

The core logic lives in `parallelbench/models/local/generate.py`. There are two functions to modify depending on your method type.

### For top-k and threshold methods: `get_transfer_index()`

Add an `elif` branch that computes a **confidence score** (`x0_p`) for each token position. Higher scores mean the token is unmasked first.

```python
def get_transfer_index(logits, temperature, unmasking, mask_index, x,
                       num_transfer_tokens, threshold=None, alg_temp=None,
                       eb_sampler_gamma=None):
    # ... existing probability computation (p, x0, x0_p) ...

    if unmasking.startswith("confidence"):
        pass  # x0_p is already the max probability
    elif unmasking.startswith("topk_margin"):
        sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
        x0_p = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif unmasking.startswith("entropy"):
        log_probs = torch.log(p + 1e-10)
        x0_p = torch.sum(p * log_probs, dim=-1)
    elif unmasking.startswith("random"):
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

    # Add your method here:
    elif unmasking.startswith("my_method"):
        # Compute x0_p: a per-token score where higher = more confident
        x0_p = ...

    else:
        raise NotImplementedError(unmasking)

    # ... rest of the function uses x0_p to select top-k tokens ...
```

The function then uses `x0_p` to select the top-k tokens to unmask. You only need to define **how confidence is computed** — the selection mechanism is shared.

### For factor methods: `get_transfer_index_dynamic()`

Same pattern — add an `elif` branch in the matching function:

```python
def get_transfer_index_dynamic(logits, temperature, unmasking, mask_index, x,
                               num_transfer_tokens, factor=1, alg_temp=None):
    # ... same structure, add elif branch for x0_p computation ...
```

### Common confidence score patterns

| Pattern | `x0_p` computation | Intuition |
| ------- | ------------------ | --------- |
| Max probability | `p.max(dim=-1).values` | How certain the model is about its top prediction |
| Margin | `top1 - top2` | Gap between best and second-best prediction |
| Negative entropy | `sum(p * log(p))` | Lower entropy = more concentrated distribution |
| Random | `torch.rand(...)` | Uniform random (baseline) |

## Step 3: Add to Model `VALID_STRATEGIES`

Each model declares which methods it supports. Add your method name to the relevant `VALID_STRATEGIES` sets:

```python
# parallelbench/models/local/llada/constants.py
LLADA_VALID_STRATEGIES = {
    "random",
    "confidence_topk",
    "confidence_threshold",
    "confidence_factor",
    "topk_margin",
    "entropy_topk",
    "my_method",  # Add here
}
```

Repeat for each model that supports the new method (e.g., `dream/constants.py`, `trado/constants.py`).

## Step 4: Verify

```bash
# Test with a single sample
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs k=4,max_tokens=32,unmasking=my_method \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1 \
  --limit 2
```

## Reference

### Existing implementations

| Method | Confidence score | File location |
| ------ | ---------------- | ------------- |
| `confidence_topk` | Max token probability | `generate.py:get_transfer_index()` |
| `topk_margin` | Top-1 minus top-2 probability | `generate.py:get_transfer_index()` |
| `entropy_topk` | Negative entropy of distribution | `generate.py:get_transfer_index()` |
| `random` | Uniform random | `generate.py:get_transfer_index()` |
| `confidence_threshold` | Max probability + threshold cutoff | `generate.py:get_transfer_index()` |
| `confidence_factor` | Max probability + dynamic factor | `generate.py:get_transfer_index_dynamic()` |

### Key files

| File | Purpose |
| ---- | ------- |
| `parallelbench/models/unmasking_registry.py` | Method classification and parameter derivation |
| `parallelbench/models/local/generate.py` | Token selection logic (`get_transfer_index`, `get_transfer_index_dynamic`) |
| `parallelbench/models/local/<model>/constants.py` | Per-model valid method sets (`VALID_STRATEGIES`) |
| `parallelbench/models/generation_config.py` | Validation of method + parameter combinations |

## Checklist

- [ ] Method registered in `unmasking_registry.py` via `register_strategy()`
- [ ] Token selection logic added to `get_transfer_index()` in `generate.py`
- [ ] (If factor-based) Also added to `get_transfer_index_dynamic()`
- [ ] Method name added to `VALID_STRATEGIES` for supported models
- [ ] `pb eval` with `--limit 2` passes on at least one task
