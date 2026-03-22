"""
KLASS (KL-Adaptive Stability Sampling) generation for LLaDA models.

Implements dual-gating with KL divergence stability tracking and confidence
thresholding. Falls back to confidence_topk when no tokens meet both criteria.

Reference: untracked/dllm-bench/model/remasking/llada/klass.py
Paper: https://arxiv.org/abs/2511.05664
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from parallelbench.models.local.generate import get_num_transfer_tokens


@torch.no_grad()
def klass_generate_llada(
    model,
    prompt: torch.Tensor,
    gen_config,
    mask_id: int,
    output_history: bool = False,
    output0_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int, Optional[list]]:
    """KLASS generation for LLaDA models.

    Implements dual-gating with KL divergence stability tracking and
    confidence thresholding. Falls back to confidence_topk when no
    tokens meet both criteria.

    Args:
        model: The LLaDA model.
        prompt: Input token IDs of shape (1, L).
        gen_config: Generation config with conf_threshold, kl_threshold, kl_history_length.
        mask_id: Mask token ID (126336 for LLaDA).
        output_history: Whether to record generation history.
        output0_ids: Optional initial output token IDs.

    Returns:
        Tuple of (generated_ids, nfe, history).
    """
    conf_threshold = (
        gen_config.conf_threshold if gen_config.conf_threshold is not None else 0.9
    )
    kl_threshold = (
        gen_config.kl_threshold if gen_config.kl_threshold is not None else 0.01
    )
    kl_history_length = (
        gen_config.kl_history_length if gen_config.kl_history_length is not None else 2
    )
    gen_length = gen_config.max_tokens
    steps = gen_config.steps
    block_length = gen_config.block_length

    # Setup sequence with prompt + masks
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if output0_ids is not None:
        x[:, prompt.shape[1] :] = output0_ids.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps is not None:
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
    else:
        steps_per_block = 999999

    nfe = 0
    input_length = prompt.shape[1]
    history = [] if output_history else None

    # Vocab size for KL buffers
    V = (
        model.lm_head.out_features
        if hasattr(model, "lm_head")
        else model.config.vocab_size
    )

    for num_block in range(num_blocks):
        block_start = input_length + num_block * block_length
        block_end = input_length + (num_block + 1) * block_length

        # Per-block state buffers
        kl_history_buffer = torch.zeros(
            (1, x.shape[1], kl_history_length), dtype=torch.float64, device=x.device
        )
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64, device=x.device)

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            # Build mask restricted to current block
            mask_index = x == mask_id
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask

            # Break if all tokens in current block are unmasked
            if not mask_index[:, block_start:block_end].any():
                break

            nfe += 1
            logits = model(x).logits

            # Compute softmax in float64 for KL precision
            p_curr = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(p_curr, dim=-1)

            # Compute confidence (max probability for the predicted token)
            curr_conf = torch.gather(p_curr, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # KL divergence between current and previous step
            eps = 1e-12
            kl_current_prev = (
                p_curr * (torch.log(p_curr + eps) - torch.log(p_prev + eps))
            ).sum(dim=-1)

            # Shift KL history and insert new KL
            kl_history_buffer = torch.roll(kl_history_buffer, shifts=-1, dims=-1)
            kl_history_buffer[..., -1] = kl_current_prev

            p_prev = p_curr.clone()

            # Dual gating: stable AND confident AND masked
            if step >= kl_history_length - 1:
                stable_mask = torch.all(kl_history_buffer < kl_threshold, dim=-1)
            else:
                stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)

            conf_mask = curr_conf > conf_threshold
            ready_mask = stable_mask & conf_mask & mask_index

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)

            for j in range(ready_mask.shape[0]):
                ready_indices = torch.where(ready_mask[j])[0]
                if len(ready_indices) > 0:
                    # Unmask all ready tokens (unmask_strategy="all")
                    transfer_index[j, ready_indices] = True
                else:
                    # Fallback: use confidence_topk with num_transfer_tokens
                    confidence = torch.where(mask_index, curr_conf, -np.inf)
                    confidence[:, block_end:] = -np.inf
                    n_transfer = num_transfer_tokens[j, step].item()
                    if n_transfer > 0:
                        _, select_index = torch.topk(confidence[j], n_transfer)
                        transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

    return x, nfe, history
