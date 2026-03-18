# Adding Custom Models

This guide walks you through integrating your own diffusion language model into ParallelBench.

## What You Need

| Component | Location | Purpose |
| --------- | -------- | ------- |
| Model class | `parallelbench/models/local/<name>/` | Generation logic |
| lm-eval wrapper | `parallelbench/lm_eval_wrappers/<name>_wrapper.py` | Bridges model to lm-eval |
| Entry point | `pyproject.toml` | Registers wrapper with lm-eval CLI |

```text
parallelbench/models/local/<name>/
├── __init__.py
├── <name>_model.py      # Model implementation
└── constants.py         # Mask token ID, valid methods
parallelbench/lm_eval_wrappers/
└── <name>_wrapper.py    # lm-eval wrapper
```

## 1. Define Constants

```python
# parallelbench/models/local/<name>/constants.py

<NAME>_MASK_TOKEN_ID = <int>  # Check your tokenizer
<NAME>_VALID_STRATEGIES = {
    "random",
    "confidence_topk",
    # Full list: random, origin, confidence_topk, topk_margin,
    #            entropy_topk, confidence_threshold, confidence_factor
}
```

## 2. Implement the Model Class

Register your model with `@ModelRegistry.register()` and implement `generate()` to return a `DLLMOutput`:

```python
# parallelbench/models/local/<name>/<name>_model.py

from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.registry import ModelRegistry

@ModelRegistry.register(matcher=lambda name: "<match_pattern>" in name)
class MyModel(LocalModel):

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self.mask_id = <NAME>_MASK_TOKEN_ID

    def generate(self, messages, gen_config=None, output_prefix=None,
                 output_history=False) -> DLLMOutput:
        # 1. Tokenize input
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        # 2. Run your generation logic
        #    gen_config keys: steps, gen_length, block_length,
        #    temperature, alg_temp, unmasking, threshold, factor

        # 3. Return DLLMOutput
        return DLLMOutput(
            output=output_text,        # str
            input_ids=input_ids,       # [1, input_length]
            output_ids=output_ids,     # [1, output_length]
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,                   # int: forward pass count
            history=...,               # optional
            decoding_order=...,        # optional
            decoding_order_corrs=...,  # optional
        )
```

### Custom GenerationConfig (optional)

Only needed if your model has extra generation parameters beyond `DllmGenerationConfig`. The base config already provides `max_tokens`, `temperature`, `unmasking`, `steps`, `block_length`, `alg_temp`, `alg_threshold`, and `alg_factor`. See `parallelbench/models/generation_config.py` for details.

### Batch Generation (optional)

Override `supports_batch` and `generate_batch()` for batched inference. Without this, `batch_size > 1` raises `NotImplementedError`.

```python
@property
def supports_batch(self) -> bool:
    return True

def generate_batch(self, messages_list, gen_config=None,
                   output_prefix_list=None, output_history=False):
    # Return List[DLLMOutput] in same order as messages_list
    ...
```

## 3. Create the lm-eval Wrapper

```python
# parallelbench/lm_eval_wrappers/<name>_wrapper.py

from lm_eval.api.registry import register_model
from parallelbench.models.base_model import BaseModel
from parallelbench.models.local.<name>.<name>_model import MyModel
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

@register_model("parallelbench_<name>")
class MyWrapper(DLLMBase):

    def _create_inner_model(self) -> BaseModel:
        return MyModel(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        if "unmasking" not in gen_kwargs:
            gen_kwargs = {**gen_kwargs, "unmasking": "confidence_topk"}
        return super()._build_generation_config(gen_kwargs)
```

If your wrapper needs additional `model_args` (e.g., Dream's `eps`), override `__init__`:

```python
def __init__(self, model_path: str, eps: float = 1e-3, **kwargs):
    self._eps = float(eps)
    super().__init__(model_path=model_path, **kwargs)
```

## 4. Register the Entry Point

Add to `pyproject.toml` and reinstall:

```toml
[project.entry-points."lm_eval.models"]
parallelbench_<name> = "parallelbench.lm_eval_wrappers.<name>_wrapper:MyWrapper"
```

```bash
uv sync
```

## 5. Verify

```bash
pb eval --model parallelbench_<name> \
  --model_args model_path=<hf_model_path> \
  --gen_kwargs k=1,max_tokens=32,unmasking=random \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1 \
  --limit 2
```

## Reference Implementations

| Model | Batch | Key Pattern | Location |
| ----- | ----- | ----------- | -------- |
| ExampleModel | No | Minimal template | `parallelbench/models/local/example/` |
| LladaModel | Yes | Right-padding `[prompt \| mask \| pad]` | `parallelbench/models/local/llada/` |
| DreamModel | Yes | Left-padding with attention_mask | `parallelbench/models/local/dream/` |
| SeddModel | No | Custom model loading | `parallelbench/models/local/sedd/` |
