"""
WINO-DLLM (Wide-In, Narrow-Out Revokable Decoding) generation for LLaDA models.

Implements the Narrow-Out Revokable Decoding strategy:
- Forward: unmask tokens above a confidence threshold (up to max_accept per step),
  always unmasking at least 1 token.
- Backward: revoke (re-mask) previously unmasked tokens whose confidence drops
  below a backward threshold.

This implementation uses the standard sequence layout without the extended
"wide input" attention trick from the original paper, since LLaDA's model
architecture does not support custom position_ids.

Reference: https://github.com/Feng-Hong/WINO-DLLM
Paper: https://arxiv.org/abs/2507.18578
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise for sampling. Returns logits unchanged if temperature=0."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def wino_generate_llada(
    model,
    prompt: torch.Tensor,
    gen_config,
    mask_id: int,
    output_history: bool = False,
    output0_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int, Optional[list]]:
    """WINO-DLLM generation for LLaDA models.

    Uses threshold-based forward acceptance and backward revocation.
    Tokens are unmasked when confidence exceeds the forward threshold,
    and previously unmasked tokens can be re-masked if their confidence
    drops below the backward threshold.

    Args:
        model: The LLaDA model.
        prompt: Input token IDs of shape (1, L).
        gen_config: Generation config with wino_threshold, wino_threshold_back.
        mask_id: Mask token ID (126336 for LLaDA).
        output_history: Whether to record generation history.
        output0_ids: Optional initial output token IDs.

    Returns:
        Tuple of (generated_ids, nfe, history).
    """
    threshold = (
        gen_config.wino_threshold if gen_config.wino_threshold is not None else 0.6
    )
    threshold_back = (
        gen_config.wino_threshold_back
        if gen_config.wino_threshold_back is not None
        else 0.9
    )
    temperature = gen_config.temperature

    gen_length = gen_config.max_tokens
    block_length = gen_config.block_length
    device = model.device

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    input_length = prompt.shape[1]

    # Standard sequence: [prompt | gen_area]
    x = torch.full(
        (1, input_length + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :input_length] = prompt.clone()

    if output0_ids is not None:
        x[:, input_length:] = output0_ids.clone()

    nfe = 0
    history = [] if output_history else None

    for num_block in range(num_blocks):
        block_start = input_length + num_block * block_length
        block_end = input_length + (num_block + 1) * block_length

        # Track which positions in this block are masked
        block_mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=device)
        block_mask[:, block_start:block_end] = x[:, block_start:block_end] == mask_id

        # Track which positions in this block have been unmasked (for revocation)
        unmasked_tracker = torch.zeros_like(block_mask)

        last_accept = 30

        while block_mask[:, block_start:block_end].any():
            num_masked = block_mask[:, block_start:block_end].sum().item()
            max_accept = min(max(int(num_masked * 0.7), 5), 20)

            nfe += 1
            logits = model(x).logits

            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute confidence
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(block_mask, x0, x)
            confidence = torch.where(block_mask, x0_p, -np.inf)
            confidence_back = torch.where(unmasked_tracker, x0_p, np.inf)

            # Forward: accept tokens above threshold
            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                _, indices = torch.topk(confidence.view(-1), k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            else:
                # Always transfer at least the max confidence token
                if not transfer_index.any():
                    max_idx = torch.argmax(confidence.view(-1))
                    transfer_index.view(-1)[max_idx] = True

            x[transfer_index] = x0[transfer_index]
            num_accept = transfer_index.sum().item()

            # Backward: revoke low-confidence tokens (only when >1 token accepted)
            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = max(last_accept - 1, 0)
                    if num_remask > 0:
                        conf_flat = confidence_back.view(-1)
                        temp_mask = torch.zeros_like(conf_flat, dtype=torch.bool)
                        _, indices = torch.topk(conf_flat, k=num_remask, largest=False)
                        temp_mask[indices] = True
                        remask_index = temp_mask.view(confidence_back.shape)
                    else:
                        remask_index = torch.zeros_like(transfer_index)
            else:
                remask_index = torch.zeros_like(transfer_index)

            # Apply revocation
            x[remask_index] = mask_id

            # Update tracking
            block_mask[transfer_index] = False
            block_mask[remask_index] = True
            unmasked_tracker[transfer_index] = True
            unmasked_tracker[remask_index] = False

            last_accept = num_accept

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

    return x, nfe, history
