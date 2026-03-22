# Adding Custom Unmasking Methods

This guide explains how to add a new unmasking method to ParallelBench.

## How Unmasking Works

At each denoising step, the model predicts tokens for all masked positions. The **unmasking method** decides which predictions to accept and which to keep masked for later refinement. Each method computes a per-token **confidence score** — higher scores get unmasked first.

There are four method types:

| Type | Behavior | CLI parameter | TPS | PBx scoring |
| ---- | -------- | ------------- | --- | ----------- |
| `topk` | Fixed `k` tokens unmasked per step | `k` | Deterministic (= k) | Discrete |
| `threshold` | Unmask tokens above a confidence threshold | `alg_threshold` | Measured | Interpolated |
| `factor` | Scale unmask count by a factor | `alg_factor` | Measured | Interpolated |
| `adaptive` | Dynamic per-step unmasking (e.g., KLASS) | Method-specific | Measured | Interpolated |

## What You Need to Change

| File | What to do |
| ---- | ---------- |
| `parallelbench/models/confidence_scorers.py` | Implement the confidence score function |
| `parallelbench/models/unmasking_registry.py` | Register the method with its confidence scorer |
| `parallelbench/models/local/<model>/constants.py` | Add the method to each model's valid set |

## 1. Implement the Confidence Score

Add a function to `parallelbench/models/confidence_scorers.py`. Every scorer has the same signature:

```python
def my_scorer(p: torch.Tensor, x0: torch.Tensor, x0_p: torch.Tensor) -> torch.Tensor:
    """
    Args:
        p: Token probability distribution (batch, seq_len, vocab_size).
        x0: Predicted token ids (batch, seq_len).
        x0_p: Pre-computed max/sampled probability (batch, seq_len).

    Returns:
        Per-token confidence tensor (batch, seq_len).
    """
    return ...
```

### Common confidence patterns

| Pattern | Computation | Intuition |
| ------- | ----------- | --------- |
| Max probability | `return x0_p` | How certain the top prediction is |
| Margin | `top1 - top2` | Gap between best and second-best |
| Negative entropy | `sum(p * log(p))` | More concentrated = more confident |
| Random | `torch.rand(...)` | Uniform baseline |

### Existing scorers

| Function | Used by |
| -------- | ------- |
| `max_probability` | `confidence_topk`, `confidence_threshold`, `confidence_factor` |
| `margin` | `topk_margin` |
| `negative_entropy` | `entropy_topk` |
| `random_confidence` | `random` |

## 2. Register the Method

Add your method to `UNMASKING_REGISTRY` in `parallelbench/models/unmasking_registry.py`:

```python
from parallelbench.models.confidence_scorers import my_scorer

UNMASKING_REGISTRY: dict[str, MethodInfo] = {
    # ... existing entries ...
    "my_method": MethodInfo("topk", "k", derive_topk, my_scorer, ("k",)),
}
```

The five arguments are:

| Argument | Description |
| -------- | ----------- |
| `method_type` | `"topk"`, `"threshold"`, `"factor"`, or `"adaptive"` |
| `representative_param` | Primary CLI parameter used for deriving `steps`/`block_length` |
| `derive_fn` | Function that derives `steps`/`block_length` from the representative param |
| `confidence_fn` | Confidence scorer function (or `None`) |
| `config_params` | Tuple of gen_kwargs keys used to distinguish configs in PBx scoring |

The `config_params` field is critical: it tells `pb analyze` which hyperparameters to extract from results and how to group configs for PBx score computation. For top-k methods this is `("k",)`. For methods with multiple hyperparameters (like KLASS), list all of them:

```python
"klass": MethodInfo(
    "adaptive", "k", derive_adaptive, max_probability,
    ("conf_threshold", "kl_threshold", "kl_history_length"),
),
```

Reuse existing derive functions (`derive_topk`, `derive_threshold`, `derive_factor`, `derive_adaptive`) and scorers when possible.

You can also register dynamically:

```python
from parallelbench.models.unmasking_registry import MethodInfo, register_method
register_method("my_method", MethodInfo("topk", "k", derive_topk, my_scorer, ("k",)))
```

## 3. Add to Model Valid Sets

Each model declares which methods it supports. Add your method name to the relevant `VALID_METHODS` sets:

```python
# parallelbench/models/local/llada/constants.py
LLADA_VALID_METHODS = {
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
