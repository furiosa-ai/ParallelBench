"""
WINO-DLLM (Wide-In, Narrow-Out Revokable Decoding) generation for LLaDA models.

Implements a decoding strategy with:
- Wide input: extended sequence [prompt | gen_area | extra_block] with custom
  attention masking so the model sees a "wider" context for each block.
- Narrow output: only tokens above a forward confidence threshold are accepted.
- Revokable: previously unmasked tokens can be re-masked if their confidence
  drops below a backward threshold in the duplicate view.

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

    Uses extended sequence with custom attention mask for "wide" input
    and revokable decoding with forward/backward confidence thresholds.

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

    # Build extended sequence: [prompt | gen_area | extra_block]
    # The extra_block is a duplicate/mirror of the current block for "wide" input
    total_len = input_length + gen_length + block_length
    x_block = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x_block[:, :input_length] = prompt.clone()

    if output0_ids is not None:
        x_block[:, input_length : input_length + gen_length] = output0_ids.clone()

    nfe = 0
    history = [] if output_history else None

    for num_block in range(num_blocks):
        block_start = input_length + num_block * block_length
        block_end = input_length + (num_block + 1) * block_length
        extra_start = input_length + gen_length  # start of extra block

        # mask_index_block: which positions in the FULL extended sequence are masked
        # (restricted to current block + before)
        mask_index_block = x_block == mask_id
        mask_index_block[:, block_end:extra_start] = (
            False  # don't touch future blocks in main area
        )

        # unmask_index_block tracks which positions in the extra block are "unmasked"
        # (i.e., have been filled with tokens from the main block)
        unmask_index_block = torch.full_like(mask_index_block, False)
        # Copy already-unmasked positions from main block to extra block tracking
        unmask_index_block[:, extra_start:] = ~mask_index_block[
            :, block_start:block_end
        ]

        # Position IDs: main gen area positions + duplicate positions for current block
        position_ids = torch.cat(
            [
                torch.arange(input_length + gen_length, device=device),
                torch.arange(block_start, block_end, device=device),
            ]
        )

        # Attention mask: [total_len x total_len]
        # - Main area can see everything in main area, but NOT the extra block
        # - Extra block can see itself + main area EXCEPT the corresponding main block positions
        #   (via ~eye mask to avoid self-referencing the same position)
        attention_mask = torch.ones(
            1, 1, total_len, total_len, dtype=torch.bool, device=device
        )
        # Main area cannot attend to extra block
        attention_mask[:, :, :, extra_start:] = False
        # Extra block attends to itself
        attention_mask[:, :, extra_start:, extra_start:] = True
        # Extra block attends to main area EXCEPT corresponding block positions (cross-attention with ~eye)
        attention_mask[:, :, extra_start:, block_start:block_end] = ~torch.eye(
            block_length, dtype=torch.bool, device=device
        )

        last_accept = 30

        while mask_index_block[:, block_start:block_end].any():
            num_masked = mask_index_block[:, block_start:block_end].sum().item()
            max_accept = min(max(int(num_masked * 0.7), 5), 20)

            nfe += 1
            logits = model(
                x_block, attention_mask=attention_mask, position_ids=position_ids
            ).logits

            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # For extra block positions that are already unmasked, use the main block's values
            unmask_shift_left = torch.zeros_like(unmask_index_block)
            unmask_shift_left[:, block_start:block_end] = unmask_index_block[
                :, extra_start:
            ]
            x0[unmask_index_block] = x_block[unmask_shift_left]

            # Compute confidence
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_index_block, x0, x_block)
            confidence = torch.where(mask_index_block, x0_p, -np.inf)
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)

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

            x_block[transfer_index] = x0[transfer_index]
            num_accept = transfer_index.sum().item()

            # Backward: revoke low-confidence tokens (only when >1 token accepted)
            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    conf_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(conf_flat, dtype=torch.bool)
                    _, indices = torch.topk(conf_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)

            # Apply revocation: shift from extra block to main block
            remask_shift = torch.zeros_like(remask_index)
            remask_shift[:, block_start:block_end] = remask_index[:, extra_start:]
            x_block[remask_shift] = mask_id

            # Update mask tracking
            mask_index_block[transfer_index] = False
            mask_index_block[remask_shift] = True

            # Update unmask tracking in extra block
            transfer_shift = torch.zeros_like(transfer_index)
            transfer_shift[:, extra_start:] = transfer_index[:, block_start:block_end]
            unmask_index_block[transfer_shift] = True
            unmask_index_block[remask_index] = False

            last_accept = num_accept

            if history is not None:
                history.append(
                    x_block[:, input_length : input_length + gen_length].cpu().clone()
                )

    # Return only the main sequence (without extra block)
    x = x_block[:, : input_length + gen_length]
    return x, nfe, history
