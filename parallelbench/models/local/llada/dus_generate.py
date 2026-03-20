"""
DUS (Dilated Unmasking Scheduler) generation for LLaDA models.

Implements a coarse-to-fine unmasking schedule where tokens are revealed
according to a dilated spatial pattern (e.g., every 64th, then every 32nd, etc.)
rather than by confidence ranking. Supports remasking low-confidence predictions.

Reference: https://github.com/omerlux/DUS-for-MDLMs
Paper: https://arxiv.org/abs/2506.19037
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _dilated_unmask_levels(
    start: int, end: int, base: int = 2, skip_exp: int = 1
) -> list[list[int]]:
    """Generate coarse-to-fine (dilated) unmasking schedule over [start..end].

    Creates multiple levels of positions to unmask, starting with sparse
    positions at regular intervals and progressively filling in gaps.

    Args:
        start: Starting position (inclusive).
        end: Ending position (inclusive).
        base: Dilation base factor (must be >= 1).
        skip_exp: Exponent for initial stride calculation.

    Returns:
        List of lists, where each inner list contains positions to unmask at that level.
    """
    if base < 1 or skip_exp < 1:
        raise ValueError("base and skip_exp must be >= 1")
    if base == 1:
        return [list(range(start, end + 1))]

    length = end - start + 1
    stride = length // (base**skip_exp)
    levels = []
    revealed = set()

    while stride >= 1:
        this_round = [
            i
            for i in range(start, end + 1)
            if (i - start) % stride == 0 and i not in revealed
        ]
        if this_round:
            levels.append(this_round)
            revealed.update(this_round)
        stride //= base

    # Reveal any leftovers
    remainder = [i for i in range(start, end + 1) if i not in revealed]
    if remainder:
        levels.append(remainder)

    return levels


def _merge_last_level(levels: list[list[int]]) -> list[list[int]]:
    """Merge the last level with the previous one if it's significantly smaller."""
    if len(levels) >= 2 and len(levels[-1]) < len(levels[-2]):
        levels[-2].extend(levels[-1])
        levels.pop()
    return levels


@torch.no_grad()
def dus_generate_llada(
    model,
    prompt: torch.Tensor,
    gen_config,
    mask_id: int,
    output_history: bool = False,
    output0_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int, Optional[list]]:
    """DUS generation for LLaDA models.

    Implements scheduled unmasking with dilated spatial patterns and
    optional remasking of low-confidence predictions.

    Args:
        model: The LLaDA model.
        prompt: Input token IDs of shape (1, L).
        gen_config: Generation config with dus_base, dus_remasking_threshold.
        mask_id: Mask token ID (126336 for LLaDA).
        output_history: Whether to record generation history.
        output0_ids: Optional initial output token IDs.

    Returns:
        Tuple of (generated_ids, nfe, history).
    """
    base = gen_config.dus_base if gen_config.dus_base is not None else 2
    remasking_threshold = (
        gen_config.dus_remasking_threshold
        if gen_config.dus_remasking_threshold is not None
        else 0.3
    )

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

    for num_block in range(num_blocks):
        block_start = input_length + num_block * block_length
        block_end_excl = input_length + (num_block + 1) * block_length

        # Generate dilated schedule for this block (positions are absolute indices in x)
        schedule = _dilated_unmask_levels(
            block_start, block_end_excl - 1, base=base, skip_exp=1
        )
        schedule = _merge_last_level(schedule)

        for step_idx, step_positions in enumerate(schedule):
            mask_index = x == mask_id
            if not mask_index[:, block_start:block_end_excl].any():
                break

            nfe += 1
            logits = model(x).logits

            # Predict tokens
            x0 = torch.argmax(logits, dim=-1)

            # Compute confidence
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # Unmask positions scheduled for this step (only if still masked)
            for pos in step_positions:
                if x[0, pos] == mask_id:
                    x[0, pos] = x0[0, pos]

            # Remasking: remask low-confidence predictions (except in final step)
            if remasking_threshold > 0 and step_idx < len(schedule) - 1:
                for pos in step_positions:
                    if x0_p[0, pos] < remasking_threshold:
                        x[0, pos] = mask_id

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

    return x, nfe, history
