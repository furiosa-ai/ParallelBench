# Adding Custom Unmasking Methods

This guide explains how to add a new unmasking method to ParallelBench.

## How Unmasking Works

At each denoising step, the model predicts tokens for all masked positions. The **unmasking method** decides which predictions to accept and which to keep masked for later refinement. Each method computes a per-token **confidence score** — higher scores get unmasked first.

There are three method types:

| Type | Behavior | CLI parameter |
| ---- | -------- | ------------- |
| `topk` | Fixed `k` tokens unmasked per step | `k` |
| `threshold` | Unmask tokens above a confidence threshold | `alg_threshold` |
| `factor` | Scale unmask count by a factor | `alg_factor` |

## What You Need to Change

| File | What to do |
| ---- | ---------- |
| `parallelbench/models/unmasking_registry.py` | Register the method type and parameter derivation |
| `parallelbench/models/local/generate.py` | Implement the confidence score computation |
| `parallelbench/models/local/<model>/constants.py` | Add the method to each model's valid set |

## 1. Register the Method

Add your method to `UNMASKING_REGISTRY` in `parallelbench/models/unmasking_registry.py`:

```python
UNMASKING_REGISTRY: dict[str, StrategyInfo] = {
    # ... existing entries ...
    "my_method": StrategyInfo("topk", "k", derive_topk),
}
```

The three arguments are: method type, representative CLI parameter, and a function that derives `steps`/`block_length` from that parameter. Reuse existing derive functions (`derive_topk`, `derive_threshold`, `derive_factor`) when possible.

You can also register dynamically:

```python
from parallelbench.models.unmasking_registry import StrategyInfo, register_strategy
register_strategy("my_method", StrategyInfo("topk", "k", derive_topk))
```

## 2. Implement the Confidence Score

In `parallelbench/models/local/generate.py`, find `get_transfer_index()` and add an `elif` branch. You only need to compute `x0_p` (per-token confidence) — the top-k selection is handled by the existing code.

```python
# In get_transfer_index():

if unmasking.startswith("confidence"):
    pass  # x0_p = max probability (already computed)
elif unmasking.startswith("topk_margin"):
    sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
    x0_p = sorted_probs[..., 0] - sorted_probs[..., 1]
elif unmasking.startswith("entropy"):
    log_probs = torch.log(p + 1e-10)
    x0_p = torch.sum(p * log_probs, dim=-1)
elif unmasking.startswith("random"):
    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

# Add yours here:
elif unmasking.startswith("my_method"):
    x0_p = ...  # your confidence score

else:
    raise NotImplementedError(unmasking)
```

For `factor`-type methods, also add the same branch in `get_transfer_index_dynamic()`.

### Common confidence patterns

| Pattern | Computation | Intuition |
| ------- | ----------- | --------- |
| Max probability | `p.max(dim=-1).values` | How certain the top prediction is |
| Margin | `top1 - top2` | Gap between best and second-best |
| Negative entropy | `sum(p * log(p))` | More concentrated = more confident |
| Random | `torch.rand(...)` | Uniform baseline |

## 3. Add to Model Valid Sets

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
    "my_method",  # add here
}
```

Repeat for each model that should support the method (e.g., `dream/constants.py`, `trado/constants.py`).

## 4. Verify

```bash
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs k=4,max_tokens=32,unmasking=my_method \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1 \
  --limit 2
```

## Existing Implementations

| Method | Confidence score | Location |
| ------ | ---------------- | -------- |
| `confidence_topk` | Max token probability | `get_transfer_index()` |
| `topk_margin` | Top-1 minus top-2 probability | `get_transfer_index()` |
| `entropy_topk` | Negative entropy | `get_transfer_index()` |
| `random` | Uniform random | `get_transfer_index()` |
| `confidence_threshold` | Max probability + threshold cutoff | `get_transfer_index()` |
| `confidence_factor` | Max probability + dynamic factor | `get_transfer_index_dynamic()` |
