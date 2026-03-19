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

## TODO
- [ ] Once the HF repo author adds `fused_linear_diffusion_cross_entropy.py`, revert to `AutoModelForCausalLM` and remove local patched files
- [ ] Related: https://huggingface.co/JetLM/SDAR-1.7B-Chat/discussions/1
