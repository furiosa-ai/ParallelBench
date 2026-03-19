"""Backward-compatible re-export. Actual implementation in block_diffusion_utils.py."""

from parallelbench.models.local.block_diffusion_utils import (
    block_diffusion_generate,
    get_num_transfer_tokens,
    sample_with_temperature_topk_topp,
    top_k_logits,
    top_p_logits,
)

__all__ = [
    "block_diffusion_generate",
    "get_num_transfer_tokens",
    "sample_with_temperature_topk_topp",
    "top_k_logits",
    "top_p_logits",
]
