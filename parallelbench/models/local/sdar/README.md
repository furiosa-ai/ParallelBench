# SDAR Model Module

Module for SDAR (Semi-autoregressive Discrete Absorbing diffusion with Refinement) models.

## Patched Files

### `modeling_sdar_patched.py`
- **Source:** `modeling_sdar.py` from `JetLM/SDAR-1.7B-Chat` HuggingFace repo
- **Patch:** Removed `from .fused_linear_diffusion_cross_entropy import FusedLinearDiffusionCrossEntropyLoss`
- **Reason:** The HF repo is missing `fused_linear_diffusion_cross_entropy.py`, causing `OSError` when loading with `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`. The module is a training-only loss function, not needed for inference.
- **Loading:** Uses local `SDARForCausalLM` directly instead of `AutoModelForCausalLM`

### `configuration_sdar.py`
- **Source:** `configuration_sdar.py` from `JetLM/SDAR-1.7B-Chat` HuggingFace repo
- **Patch:** None (original as-is)
- **Reason:** Required by the local modeling file

## Supported Models
- `JetLM/SDAR-1.7B-Chat`
- `JetLM/SDAR-4B-Chat`
- `JetLM/SDAR-8B-Chat`

## Known Issues

### `flex_attention` + `torch.compile` runtime error
SDAR's `modeling_sdar.py` uses `@torch.compile` with `flex_attention`, which causes `InternalTorchDynamoError: AttributeError: 'Tensor' object has no attribute 'BLOCK_SIZE'` at inference time. This is an environment compatibility issue (PyTorch/CUDA version mismatch with `flex_attention`).

**Status:** Blocked — SDAR models cannot run inference until this is resolved.

**Possible solutions:**
1. Replace `flex_attention` with `flash_attn` in the patched modeling file (referencing TraDo's implementation)
2. Use the official [JetAstra/SDAR](https://github.com/JetAstra/SDAR) repo's `generate.py` for inference
3. Wait for upstream fix from the model authors

## TODO
- [ ] Resolve `flex_attention` runtime error for inference
- [ ] Once the HF repo author adds `fused_linear_diffusion_cross_entropy.py`, revert to `AutoModelForCausalLM` and remove local patched files
- [ ] Related: https://huggingface.co/JetLM/SDAR-1.7B-Chat/discussions/1
