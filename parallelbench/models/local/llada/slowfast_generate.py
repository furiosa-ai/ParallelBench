"""
SlowFast Sampling generation for LLaDA models.

Implements a two-phase approach:
- Slow phase: Explores ahead to determine how many tokens can be reliably predicted
  (sub-cycle length), filling 1 token per exploration step.
- Fast phase: Fills the determined sub-cycle region by unmasking all high-confidence
  tokens at once, falling back to top-1 when none meet the threshold.

The cycle repeats with advancing sub-cycle boundaries until the block is fully unmasked.

Reference: https://github.com/LiangrunFlora/Slow-Fast-Sampling
Paper: https://arxiv.org/abs/2506.10848
"""

import collections
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def slowfast_generate_llada(
    model,
    prompt: torch.Tensor,
    gen_config,
    mask_id: int,
    output_history: bool = False,
    output0_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int, Optional[list]]:
    """SlowFast Sampling generation for LLaDA models.

    Args:
        model: The LLaDA model.
        prompt: Input token IDs of shape (1, L).
        gen_config: Generation config with sf_exploration_steps,
            sf_cycle_confidence_threshold, sf_high_confidence_threshold.
        mask_id: Mask token ID (126336 for LLaDA).
        output_history: Whether to record generation history.
        output0_ids: Optional initial output token IDs.

    Returns:
        Tuple of (generated_ids, nfe, history).
    """
    # Extract parameters with defaults
    exploration_steps = (
        gen_config.sf_exploration_steps
        if gen_config.sf_exploration_steps is not None
        else 6
    )
    cycle_confidence_threshold = (
        gen_config.sf_cycle_confidence_threshold
        if gen_config.sf_cycle_confidence_threshold is not None
        else 0.3
    )
    high_confidence_threshold = (
        gen_config.sf_high_confidence_threshold
        if gen_config.sf_high_confidence_threshold is not None
        else 0.9
    )
    stability_window = 2
    stability_std_threshold = 1.0
    max_sub_cycles = 256

    gen_length = gen_config.max_tokens
    block_length = gen_config.block_length

    # Setup sequence
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if output0_ids is not None:
        x[:, prompt.shape[1] :] = output0_ids.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    nfe = 0
    input_length = prompt.shape[1]
    history = [] if output_history else None

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length  # relative to gen area
        block_end = (block_idx + 1) * block_length

        sub_cycle_count = 0
        last_sub_cycle_end = 0  # relative to block start
        actual_sub_cycle_end = block_length

        while sub_cycle_count < max_sub_cycles:
            # Check if block is fully unmasked
            block_abs_start = input_length + block_start
            block_abs_end = input_length + block_end
            if not (x[:, block_abs_start:block_abs_end] == mask_id).any():
                break

            sub_cycle_count += 1

            # === SLOW PHASE: Explore to determine sub-cycle length ===
            x, nfe, actual_sub_cycle_end = _slow_phase(
                model=model,
                x=x,
                input_length=input_length,
                block_start=block_start,
                block_end=block_end,
                last_sub_cycle_end=last_sub_cycle_end,
                mask_id=mask_id,
                exploration_steps=exploration_steps,
                cycle_confidence_threshold=cycle_confidence_threshold,
                high_confidence_threshold=high_confidence_threshold,
                stability_window=stability_window,
                stability_std_threshold=stability_std_threshold,
                nfe=nfe,
            )

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

            # === FAST PHASE: Fill the determined sub-cycle region ===
            x, nfe = _fast_phase(
                model=model,
                x=x,
                input_length=input_length,
                block_start=block_start,
                block_end=block_end,
                last_sub_cycle_end=last_sub_cycle_end,
                actual_sub_cycle_end=actual_sub_cycle_end,
                mask_id=mask_id,
                high_confidence_threshold=high_confidence_threshold,
                nfe=nfe,
                history=history,
            )

            last_sub_cycle_end = actual_sub_cycle_end

    return x, nfe, history


def _slow_phase(
    model,
    x: torch.Tensor,
    input_length: int,
    block_start: int,
    block_end: int,
    last_sub_cycle_end: int,
    mask_id: int,
    exploration_steps: int,
    cycle_confidence_threshold: float,
    high_confidence_threshold: float,
    stability_window: int,
    stability_std_threshold: float,
    nfe: int,
) -> tuple[torch.Tensor, int, int]:
    """Slow phase: explore to determine sub-cycle length, filling 1 token per step.

    Returns:
        (x, nfe, actual_sub_cycle_end) where actual_sub_cycle_end is relative to block_start.
    """
    block_length = block_end - block_start
    length_history = collections.deque(maxlen=stability_window)
    determined = False
    actual_sub_cycle_end = block_length

    for k_step in range(exploration_steps):
        nfe += 1
        logits = model(x).logits

        # Compute predictions and confidence for generation area
        gen_logits = logits[:, input_length:]
        x0_gen = torch.argmax(gen_logits, dim=-1)
        p_gen = F.softmax(gen_logits, dim=-1)
        x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)

        # Confidence only for masked positions in gen area
        gen_mask = x[:, input_length:] == mask_id
        confidence = torch.where(
            gen_mask, x0_p_gen, torch.tensor(-np.inf, device=x.device)
        )

        # Estimate sub-cycle length: from last_sub_cycle_end, how far ahead
        # are tokens predicted with confidence >= cycle_confidence_threshold?
        if not determined:
            obs_start = block_start + last_sub_cycle_end
            obs_end = block_end
            if obs_start < obs_end:
                obs_conf = confidence[0, obs_start:obs_end]
                above = (obs_conf >= cycle_confidence_threshold).nonzero(as_tuple=True)[
                    0
                ]
                if len(above) > 0:
                    increment = above.max().item() + 1
                else:
                    increment = 1
            else:
                increment = 0

            est_len = last_sub_cycle_end + increment
            est_len = max(1, min(est_len, block_length))
            length_history.append(est_len)

            if len(length_history) >= stability_window:
                hist_arr = np.array(list(length_history))
                if np.std(hist_arr) < stability_std_threshold:
                    actual_sub_cycle_end = int(length_history[-1])
                    determined = True
                elif k_step == exploration_steps - 1:
                    actual_sub_cycle_end = int(np.mean(hist_arr))
            elif k_step == exploration_steps - 1:
                if len(length_history) > 0:
                    actual_sub_cycle_end = max(
                        1, min(int(np.mean(list(length_history))), block_length)
                    )
                else:
                    actual_sub_cycle_end = block_length // 2

        # Fill 1 token: the most confident masked token in the exploration scope
        fill_start = block_start + last_sub_cycle_end
        fill_end = block_end
        if fill_start < fill_end:
            scope_mask = (
                x[0, input_length + fill_start : input_length + fill_end] == mask_id
            )
            scope_conf = confidence[0, fill_start:fill_end]
            eff_conf = torch.where(
                scope_mask, scope_conf, torch.tensor(-np.inf, device=x.device)
            )

            if scope_mask.any():
                high_conf_mask = (eff_conf >= high_confidence_threshold) & scope_mask
                if high_conf_mask.any() and high_conf_mask.sum() > 1:
                    # Unmask all high-confidence tokens
                    abs_indices = fill_start + high_conf_mask.nonzero(as_tuple=True)[0]
                    x[0, input_length + abs_indices] = x0_gen[0, abs_indices]
                else:
                    # Unmask top-1
                    top_idx = eff_conf.argmax()
                    x[0, input_length + fill_start + top_idx] = x0_gen[
                        0, fill_start + top_idx
                    ]

    actual_sub_cycle_end = max(1, min(actual_sub_cycle_end, block_length))
    return x, nfe, actual_sub_cycle_end


def _fast_phase(
    model,
    x: torch.Tensor,
    input_length: int,
    block_start: int,
    block_end: int,
    last_sub_cycle_end: int,
    actual_sub_cycle_end: int,
    mask_id: int,
    high_confidence_threshold: float,
    nfe: int,
    history: Optional[list],
) -> tuple[torch.Tensor, int]:
    """Fast phase: fill the sub-cycle region by unmasking high-confidence tokens."""
    sub_start = block_start + last_sub_cycle_end  # relative to gen area
    sub_end = block_start + actual_sub_cycle_end

    while True:
        # Check if sub-cycle region is fully unmasked
        region = x[0, input_length + sub_start : input_length + sub_end]
        if not (region == mask_id).any():
            break

        nfe += 1
        logits = model(x).logits

        gen_logits = logits[:, input_length:]
        x0_gen = torch.argmax(gen_logits, dim=-1)
        p_gen = F.softmax(gen_logits, dim=-1)
        x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)

        gen_mask = x[:, input_length:] == mask_id
        confidence = torch.where(
            gen_mask, x0_p_gen, torch.tensor(-np.inf, device=x.device)
        )

        # Focus on the sub-cycle scope
        scope_mask = x[0, input_length + sub_start : input_length + sub_end] == mask_id
        scope_conf = confidence[0, sub_start:sub_end]

        high_conf_mask = (scope_conf >= high_confidence_threshold) & scope_mask
        if high_conf_mask.any() and high_conf_mask.sum() > 1:
            abs_indices = sub_start + high_conf_mask.nonzero(as_tuple=True)[0]
            x[0, input_length + abs_indices] = x0_gen[0, abs_indices]
        else:
            # Fallback: unmask top-1
            eff_conf = torch.where(
                scope_mask, scope_conf, torch.tensor(-np.inf, device=x.device)
            )
            if scope_mask.any():
                top_idx = eff_conf.argmax()
                x[0, input_length + sub_start + top_idx] = x0_gen[
                    0, sub_start + top_idx
                ]

        if history is not None:
            history.append(x[:, input_length:].cpu().clone())

    return x, nfe
