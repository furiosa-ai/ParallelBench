# TraDo Model Module

Module for TraDo (Transformer Diffusion) models.

## Patched Files

### `modeling_trado_patched.py`
- **Source:** `modeling_sdar.py` from `Gen-Verse/TraDo-4B-Instruct` HuggingFace repo
- **Patch:**
  1. Removed `LossKwargs` from `from transformers.utils import ...`
  2. Changed `class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs)` to `class KwargsForCausalLM(FlashAttentionKwargs)`
- **Reason:** The HF repo's `modeling_sdar.py` imports `LossKwargs` from `transformers.utils`, which does not exist in the current transformers version. `LossKwargs` is a TypedDict for type hints only, not needed for inference.
- **Loading:** Uses `AutoModelForCausalLM(trust_remote_code=True)` with a dummy `LossKwargs` injected into `transformers.utils` before loading

### Additional Fix: `trado_model.py`
- `alg_threshold` default set to `0.85` to match the default `unmasking="confidence_threshold"` method
- **Reason:** `DllmGenerationConfig._validate_unmasking()` requires `alg_threshold` to be non-None for threshold-type methods. The value `0.85` matches the fallback in `to_generation_kwargs()`

## Supported Models
- `Gen-Verse/TraDo-4B-Instruct`
- `Gen-Verse/TraDo-8B-Instruct`
- `Gen-Verse/TraDo-8B-Thinking`

## TODO
- [ ] Once the HF repo author removes the `LossKwargs` import, remove the monkey-patch code
- [ ] Related PR: <https://huggingface.co/JetLM/SDAR-1.7B-Chat/discussions/1> (same architecture)
