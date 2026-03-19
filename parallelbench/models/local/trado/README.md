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
- Changed `alg_threshold` default from `0.85` to `None`
- **Reason:** Default `0.85` caused `DllmGenerationConfig._validate_unmasking()` to raise `ValueError` for topk-type methods (e.g., `confidence_topk`) which require `alg_threshold` to be `None`

## Supported Models
- `Gen-Verse/TraDo-4B-Instruct`
- `Gen-Verse/TraDo-8B-Instruct`
- `Gen-Verse/TraDo-8B-Thinking`

## TODO
- [ ] Once the HF repo author removes the `LossKwargs` import, remove the monkey-patch code
- [ ] Related PR: https://huggingface.co/JetLM/SDAR-1.7B-Chat/discussions/1 (same architecture)
