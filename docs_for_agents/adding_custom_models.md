# Adding Custom Models (Agent Guide)

This guide provides step-by-step instructions for AI agents to integrate a new diffusion language model into ParallelBench.

## Overview

Adding a custom model requires **3 components**:

| Component       | Location                                           | Purpose                            |
| --------------- | -------------------------------------------------- | ---------------------------------- |
| Model class     | `parallelbench/models/local/<name>/`               | Generation logic                   |
| lm-eval wrapper | `parallelbench/lm_eval_wrappers/<name>_wrapper.py` | Bridges model to lm-eval           |
| Entry point     | `pyproject.toml`                                   | Registers wrapper with lm-eval CLI |

## File Structure

Create the following files:

```text
parallelbench/models/local/<name>/
├── __init__.py          # Export model class
├── <name>_model.py      # Model implementation
└── constants.py         # Mask token ID, valid strategies
parallelbench/lm_eval_wrappers/
└── <name>_wrapper.py    # lm-eval wrapper
```

## Step 1: Define Constants

```python
# parallelbench/models/local/<name>/constants.py

<NAME>_MASK_TOKEN_ID = <int>  # Check tokenizer for the correct mask token ID
<NAME>_VALID_STRATEGIES = {
    "random",
    "confidence_topk",
    # Add only strategies your model actually supports.
    # Full list: random, origin, confidence_topk, topk_margin,
    #            entropy_topk, confidence_threshold, confidence_factor
}
```

## Step 2: Define GenerationConfig (Optional)

Only needed if your model has extra generation parameters beyond the base set.
Skip this step if the base `DllmGenerationConfig` fields are sufficient.

**Base fields already available** (from `DllmGenerationConfig`):

| Field           | Type  | Default             | Description                                       |
| --------------- | ----- | ------------------- | ------------------------------------------------- |
| `max_tokens`    | int   | 128                 | Maximum generation length                         |
| `temperature`   | float | 0.0                 | Sampling temperature                              |
| `unmasking`     | str   | None                | Unmasking strategy name                           |
| `steps`         | int   | 128                 | Total denoising steps                             |
| `block_length`  | int   | None (→ max_tokens) | Semi-AR block size                                |
| `alg_temp`      | float | 0.0                 | Algorithm temperature                             |
| `alg_threshold` | float | None                | Confidence threshold (for `confidence_threshold`) |
| `alg_factor`    | float | None                | Scaling factor (for `confidence_factor`)          |

**If you need extra fields:**

```python
# In <name>_model.py

from dataclasses import dataclass, field
from parallelbench.models.generation_config import DllmGenerationConfig
from .constants import <NAME>_VALID_STRATEGIES

@dataclass
class <Name>GenerationConfig(DllmGenerationConfig):
    # Override defaults
    unmasking: str = "confidence_topk"  # Model's default strategy
    block_length: int = 128
    valid_strategies: set = field(
        default_factory=lambda: set(<NAME>_VALID_STRATEGIES)
    )

    # Add model-specific fields
    custom_param: float = 1.0

    def to_generation_kwargs(self) -> dict:
        gen_kwargs = super().to_generation_kwargs()
        gen_kwargs["custom_param"] = self.custom_param
        return gen_kwargs
```

**Validation rules enforced by `DllmGenerationConfig.__post_init__`:**
- `max_tokens % block_length == 0`
- `steps % (max_tokens // block_length) == 0`
- `steps <= max_tokens`
- `unmasking` must be in `valid_strategies`
- Top-k strategies: `alg_threshold` must be None/0.0, `alg_factor` must be None/1.0
- Threshold strategies: `alg_threshold` required, `alg_factor` must be None
- Factor strategies: `alg_factor` required, `alg_threshold` must be None

## Step 3: Implement Model Class

```python
# parallelbench/models/local/<name>/<name>_model.py

from typing import Union, List, Optional, Dict
import torch
from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.registry import ModelRegistry
from parallelbench.models.model_utils import decode_history


@ModelRegistry.register(matcher=lambda name: "<match_pattern>" in name)
class <Name>Model(LocalModel):

    def __init__(self, model_name: str, **kwargs):
        # LocalModel.__init__ does:
        #   self.model = AutoModel.from_pretrained(model_name, ...)
        #   self.tokenizer = AutoTokenizer.from_pretrained(model_name, ...)
        super().__init__(model_name)
        self.mask_id = <NAME>_MASK_TOKEN_ID

    def generate(
        self,
        messages: Union[List[str], str],
        gen_config: Dict = None,
        output_prefix: Optional[str] = None,
        output_history: bool = False,
    ) -> DLLMOutput:
        # 1. Tokenize input
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        # 2. Run generation (model-specific logic)
        #    - gen_config is a dict from GenerationConfig.to_generation_kwargs()
        #    - Available keys: steps, gen_length, block_length, temperature,
        #      alg_temp, unmasking, threshold, factor
        #    - Track nfe (number of forward evaluations)
        #    - If output_history: record intermediate states

        # 3. Decode output
        output_ids = ...  # shape: [1, output_length]
        output_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        # 4. Return DLLMOutput
        return DLLMOutput(
            output=output_text,
            input_ids=input_ids,           # shape: [1, input_length]
            output_ids=output_ids,         # shape: [1, output_length]
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,                       # int: actual forward pass count
            history=decode_history(self.tokenizer, history) if output_history else None,
            decoding_order=decoding_order if output_history else None,
            decoding_order_corrs=decoding_order_corrs if output_history else None,
        )
```

### DLLMOutput Fields

| Field                  | Type          | Required | Description                                                                  |
| ---------------------- | ------------- | -------- | ---------------------------------------------------------------------------- |
| `output`               | str           | Yes      | Generated text                                                               |
| `input_ids`            | Tensor [1, L] | Yes      | Input token IDs                                                              |
| `output_ids`           | Tensor [1, L] | Yes      | Output token IDs                                                             |
| `pad_token_id`         | int           | Yes      | For computing `output_length` (excludes padding)                             |
| `nfe`                  | int           | Yes      | Number of forward evaluations (critical for benchmarking)                    |
| `history`              | dict          | No       | Decoded history via `decode_history()`                                       |
| `decoding_order`       | Tensor        | No       | Token unmasking order                                                        |
| `decoding_order_corrs` | dict          | No       | Correlation metrics from `compute_decoding_order_correlation_from_history()` |

## Step 4: Batch Generation (Optional)

If your model supports batched inference, add these to your model class:

```python
@property
def supports_batch(self) -> bool:
    return True

def generate_batch(
    self,
    messages_list: List[Union[List[dict], str]],
    gen_config: Dict = None,
    output_prefix_list: Optional[List[Optional[str]]] = None,
    output_history: bool = False,
) -> List[DLLMOutput]:
    # 1. Tokenize all inputs
    # 2. Pad to same length (choose a padding strategy):
    #    - Right-padding [prompt | mask | pad]: preserves RoPE positions (LLaDA)
    #    - Left-padding with attention_mask: standard approach (Dream)
    # 3. Run batched forward pass
    # 4. Return List[DLLMOutput] in same order as messages_list
    ...
```

**Without batch support**: `batch_size > 1` will raise `NotImplementedError`. There is no silent fallback to sequential.

## Step 5: Create lm-eval Wrapper

```python
# parallelbench/lm_eval_wrappers/<name>_wrapper.py

from __future__ import annotations
from lm_eval.api.registry import register_model
from parallelbench.models.base_model import BaseModel
from parallelbench.models.local.<name>.<name>_model import <Name>Model
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase


@register_model("parallelbench_<name>")
class <Name>Wrapper(DLLMBase):

    def _create_inner_model(self) -> BaseModel:
        return <Name>Model(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        # Set model-specific default unmasking if not provided by user
        if "unmasking" not in gen_kwargs:
            gen_kwargs = {**gen_kwargs, "unmasking": "confidence_topk"}
        return super()._build_generation_config(gen_kwargs)
```

**DLLMBase constructor `model_args`** (passed via `--model_args` CLI):

| Arg              | Type | Default  | Description                 |
| ---------------- | ---- | -------- | --------------------------- |
| `model_path`     | str  | required | HuggingFace model name/path |
| `output_history` | bool | True     | Track generation history    |
| `infill`         | bool | False    | Infill mode                 |
| `batch_size`     | int  | 1        | Batch size for generation   |

Extra `model_args` are stored in `self._extra_kwargs` and bridged to `gen_kwargs` if they match `BRIDGED_KEYS`: `k`, `alg_threshold`, `alg_factor`, `steps`, `block_length`, `unmasking`.

If your wrapper needs additional `model_args` (e.g., Dream's `eps`), override `__init__`:

```python
def __init__(self, model_path: str, eps: float = 1e-3, **kwargs):
    self._eps = float(eps)
    super().__init__(model_path=model_path, **kwargs)
```

## Step 6: Register Entry Point

Add to `pyproject.toml`:

```toml
[project.entry-points."lm_eval.models"]
# ... existing entries ...
parallelbench_<name> = "parallelbench.lm_eval_wrappers.<name>_wrapper:<Name>Wrapper"
```

Then reinstall the package: `uv sync`

## Step 7: Verify

```bash
# Single sample test
pb eval --model parallelbench_<name> \
  --model_args model_path=<hf_model_path> \
  --gen_kwargs k=1,max_tokens=32,unmasking=random \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1 \
  --limit 2

# Batch test (if batch supported)
pb eval --model parallelbench_<name> \
  --model_args model_path=<hf_model_path> \
  --gen_kwargs k=1,max_tokens=32,unmasking=random \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 4 \
  --limit 2
```

## Reference Implementations

| Model            | Batch | Key Pattern                                           | File                                  |
| ---------------- | ----- | ----------------------------------------------------- | ------------------------------------- |
| **ExampleModel** | No    | Minimal template                                      | `parallelbench/models/local/example/` |
| **LladaModel**   | Yes   | Right-padding `[prompt\                               | mask\                                 |
| **DreamModel**   | Yes   | Left-padding with attention_mask, model patching      | `parallelbench/models/local/dream/`   |
| **SeddModel**    | No    | Custom model loading (bypasses `LocalModel.__init__`) | `parallelbench/models/local/sedd/`    |

## Checklist

- [ ] `constants.py` with mask token ID and valid strategies
- [ ] Model class with `generate()` returning `DLLMOutput`
- [ ] `@ModelRegistry.register(matcher=...)` decorator on model class
- [ ] lm-eval wrapper with `@register_model("parallelbench_<name>")`
- [ ] Entry point in `pyproject.toml` under `[project.entry-points."lm_eval.models"]`
- [ ] `uv sync` to reinstall
- [ ] `pb eval` with `--limit 2` passes on at least one task
- [ ] (Optional) `generate_batch()` + `supports_batch` for batch support
