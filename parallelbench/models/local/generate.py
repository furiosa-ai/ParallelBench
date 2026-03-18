# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import warnings
from typing import Optional

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F

from parallelbench.models.unmasking_registry import get_method_info


def get_num_transfer_tokens(mask_index, steps):
    if steps is None:
        return None

    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    unmasking="confidence_topk",
    mask_id=126336,
    threshold=None,
    factor=None,
    output_history=False,
    output0_ids=None,
    alg_temp=0.0,
    eb_sampler_gamma=None,
) -> tuple[torch.Tensor, int, Optional[list]]:
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive unmasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        unmasking: Unmasking method. 'confidence_topk' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
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
        steps = steps // num_blocks
    else:
        assert (
            threshold is not None or factor is not None or eb_sampler_gamma is not None
        ), "If steps is None, threshold and factor must be provided."

    nfe = 0

    input_length = prompt.shape[1]
    history = [] if output_history else None

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    unmasking,
                    mask_index,
                    x,
                    (
                        num_transfer_tokens[:, i]
                        if threshold is None and eb_sampler_gamma is None
                        else None
                    ),
                    threshold,
                    alg_temp=alg_temp,
                    eb_sampler_gamma=eb_sampler_gamma,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    unmasking,
                    mask_index,
                    x,
                    None,
                    factor,
                    alg_temp=alg_temp,
                )
            x[transfer_index] = x0[transfer_index]

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

            i += 1
            if (
                x[
                    :,
                    prompt.shape[1] + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break

    return x, nfe, history


def get_transfer_index(
    logits,
    temperature,
    unmasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
    alg_temp=None,
    eb_sampler_gamma=None,
):
    assert alg_temp is not None

    if temperature > 0:
        logits = logits / temperature
    # if top_p is not None and top_p < 1:
    #     logits = top_p_logits(logits, top_p)
    # if top_k is not None:
    #     logits = top_k_logits(logits, top_k)
    p = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=p).sample()
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        except (ValueError, RuntimeError):
            x0_p, x0 = p.max(dim=-1)
        except Exception as e:
            warnings.warn(
                f"Unexpected exception {e} when sampling tokens, using argmax instead."
            )
            x0_p, x0 = p.max(dim=-1)
    else:
        x0_p, x0 = p.max(dim=-1)

    confidence_fn = get_method_info(unmasking).confidence_fn
    if confidence_fn is None:
        raise ValueError(
            f"Unmasking method '{unmasking}' has no confidence scorer registered."
        )
    x0_p = confidence_fn(p, x0, x0_p)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        # overwrite num_transfer_tokens
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    if eb_sampler_gamma is not None:
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        acc_entropy = torch.cumsum(entropy, dim=1)
        cummax_entropy = torch.cummax(entropy, dim=1).values
        num_transfer_tokens = (acc_entropy - cummax_entropy <= eb_sampler_gamma).sum(1)

    for j in range(confidence.shape[0]):
        if alg_temp is None or alg_temp == 0:
            _, select_index = torch.topk(confidence[j], num_transfer_tokens[j])
        else:
            confidence[j] = confidence[j] / alg_temp
            confidence[j] = F.softmax(confidence[j], dim=-1)
            select_index = torch.multinomial(
                confidence[j], num_samples=num_transfer_tokens[j]
            )

        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(
    logits,
    temperature,
    unmasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
    alg_temp=None,
):
    assert alg_temp is not None

    if temperature > 0:
        logits = logits / temperature
    # if top_p is not None and top_p < 1:
    #     logits = top_p_logits(logits, top_p)
    # if top_k is not None:
    #     logits = top_k_logits(logits, top_k)
    p = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=p).sample()
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        except (ValueError, RuntimeError):
            x0_p, x0 = p.max(dim=-1)
        except Exception as e:
            warnings.warn(
                f"Unexpected exception {e} when sampling tokens, using argmax instead."
            )
            x0_p, x0 = p.max(dim=-1)
    else:
        x0_p, x0 = p.max(dim=-1)

    confidence_fn = get_method_info(unmasking).confidence_fn
    if confidence_fn is None:
        raise ValueError(
            f"Unmasking method '{unmasking}' has no confidence scorer registered."
        )
    x0_p = confidence_fn(p, x0, x0_p)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        #  _, select_index = torch.topk(confidence[j], k=top_i)

        if alg_temp is None or alg_temp == 0:
            _, select_index = torch.topk(confidence[j], top_i)
        else:
            confidence[j] = confidence[j] / alg_temp
            confidence[j] = F.softmax(confidence[j], dim=-1)
            select_index = torch.multinomial(confidence[j], num_samples=top_i)

        transfer_index[j, select_index] = True

    return x0, transfer_index


@torch.no_grad()
def generate_batch(
    model,
    prompts,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    unmasking="confidence_topk",
    mask_id=126336,
    pad_id=0,
    threshold=None,
    factor=None,
    output_history=False,
    alg_temp=0.0,
    eb_sampler_gamma=None,
) -> tuple[torch.Tensor, int, Optional[list[list]]]:
    """Batched generation for masked diffusion language models.

    NOTE: This is NOT an official LLaDA implementation. LLaDA does not officially
    support batched generation (see https://github.com/ML-GSAI/LLaDA/issues/78).
    This implementation uses a [prompt | mask | pad] layout with right-padding and
    attention masking to enable correct batched inference without model modification.

    Layout per sample: [prompt | mask_tokens | pad_tokens]
    Pad tokens are placed AFTER mask tokens and excluded via attention_mask,
    so RoPE position encoding remains correct without model modification.

    Args:
        model: Mask predictor.
        prompts: List of 1D tensors, each of shape (L_i,) with variable lengths.
        steps: Sampling steps per block.
        gen_length: Generated answer length (same for all samples).
        block_length: Block length for semi-autoregressive unmasking.
        temperature: Categorical distribution sampling temperature.
        unmasking: Unmasking method.
        mask_id: The token id of [MASK].
        pad_id: The token id used for right-padding.
        threshold: Confidence threshold for threshold-based unmasking.
        factor: Dynamic unmasking factor.
        output_history: Whether to output generation history per sample.
        alg_temp: Algorithm temperature.
        eb_sampler_gamma: Entropy-based sampler gamma.

    Returns:
        x: Tensor of shape (B, max_prompt_len + gen_length) with generated tokens.
        nfe: Number of forward evaluations (shared across the batch).
        history: List of per-sample history lists, or None.
    """
    batch_size = len(prompts)
    device = (
        model.device if hasattr(model, "device") else next(model.parameters()).device
    )

    prompt_lengths = torch.tensor(
        [p.squeeze(0).shape[0] if p.dim() == 2 else p.shape[0] for p in prompts],
        dtype=torch.long,
        device=device,
    )
    max_prompt_len = prompt_lengths.max().item()
    max_total_len = max_prompt_len + gen_length

    # Build x: [prompt | masks | pads]
    x = torch.full((batch_size, max_total_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(
        (batch_size, max_total_len), dtype=torch.float, device=device
    )

    for i in range(batch_size):
        pl = prompt_lengths[i].item()
        p = prompts[i].to(device)
        if p.dim() == 2:
            p = p.squeeze(0)
        x[i, :pl] = p
        x[i, pl : pl + gen_length] = mask_id
        attention_mask[i, : pl + gen_length] = 1.0

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps is not None:
        assert steps % num_blocks == 0
        steps = steps // num_blocks
    else:
        assert (
            threshold is not None or factor is not None or eb_sampler_gamma is not None
        ), "If steps is None, threshold, factor, or eb_sampler_gamma must be provided."

    nfe = 0
    # Precompute position indices for vectorized per-sample boundary ops
    positions = torch.arange(max_total_len, device=device).unsqueeze(0)  # (1, L)
    history = [[] for _ in range(batch_size)] if output_history else None

    for num_block in range(num_blocks):
        block_starts = (prompt_lengths + num_block * block_length).unsqueeze(
            1
        )  # (B, 1)
        block_ends = (prompt_lengths + (num_block + 1) * block_length).unsqueeze(
            1
        )  # (B, 1)
        in_block = (positions >= block_starts) & (positions < block_ends)  # (B, L)

        block_mask_index = (x == mask_id) & in_block
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        i_step = 0
        while True:
            nfe += 1
            # Include all masks up to current block end (previous blocks already filled)
            mask_index = (x == mask_id) & (positions < block_ends)

            logits = model(x, attention_mask=attention_mask).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    unmasking,
                    mask_index,
                    x,
                    (
                        num_transfer_tokens[:, i_step]
                        if threshold is None and eb_sampler_gamma is None
                        else None
                    ),
                    threshold,
                    alg_temp=alg_temp,
                    eb_sampler_gamma=eb_sampler_gamma,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    unmasking,
                    mask_index,
                    x,
                    None,
                    factor,
                    alg_temp=alg_temp,
                )
            x[transfer_index] = x0[transfer_index]

            if history is not None:
                for si in range(batch_size):
                    pl = prompt_lengths[si].item()
                    history[si].append(
                        x[si, pl : pl + gen_length].cpu().clone().unsqueeze(0)
                    )

            i_step += 1

            # Break when all masks in current block are filled for ALL samples
            remaining = (x == mask_id) & in_block
            if not remaining.any():
                break

    return x, nfe, history
