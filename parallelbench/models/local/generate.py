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
    *args,
    use_fast_dllm_cache: bool = False,
    use_fast_dllm_dual_cache: bool = False,
    **kwargs,
):
    if use_fast_dllm_cache:
        return generate_with_prefix_cache(*args, **kwargs)

    elif use_fast_dllm_dual_cache:
        return generate_with_dual_cache(*args, **kwargs)

    return generate_with_no_cache(*args, **kwargs)


@torch.no_grad()
def generate_with_no_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="confidence_topk",
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
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'confidence_topk' or 'random'.
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
                    remasking,
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
                    remasking,
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


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="confidence_topk",
    mask_id=126336,
    threshold=None,
    factor=None,
    output_history=False,
    alg_temp=0.0,
    **kwargs,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'confidence_topk' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps is not None:
        assert steps % num_blocks == 0
        steps = steps // num_blocks

    nfe = 0

    input_length = prompt.shape[1]
    history = [] if output_history else None

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        nfe += 1
        past_key_values = output.past_key_values

        mask_index = x == mask_id
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
                alg_temp=alg_temp,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                None,
                factor,
                alg_temp=alg_temp,
            )
        x[transfer_index] = x0[transfer_index]

        if history is not None:
            history.append(x[:, input_length:].cpu().clone())

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (
                    past_key_values[i][j][:, :, :current_block_start],
                )

        past_key_values = new_past_key_values

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = x[:, current_block_start:] == mask_id
            mask_index[:, block_length:] = 0

            logits = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True,
            ).logits
            nfe += 1

            # logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                    alg_temp=alg_temp,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    None,
                    factor,
                    alg_temp=alg_temp,
                )
            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

            i += 1

    return x, nfe, history


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="confidence_topk",
    mask_id=126336,
    threshold=None,
    factor=None,
    output_history=False,
    alg_temp=0.0,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'confidence_topk' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps is not None:
        assert steps % num_blocks == 0
        steps = steps // num_blocks

    nfe = 0

    input_length = prompt.shape[1]
    history = [] if output_history else None

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = x == mask_id
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
                alg_temp=alg_temp,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                None,
                factor,
                alg_temp=alg_temp,
            )
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        if history is not None:
            history.append(x[:, input_length:].cpu().clone())

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = x[:, current_block_start:current_block_end] == mask_id
            # cache position is the position between current_block_start and current_block_end
            logits = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                    alg_temp=alg_temp,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:current_block_end],
                    None,
                    factor,
                    alg_temp=alg_temp,
                )
            x[:, current_block_start:current_block_end][transfer_index] = x0[
                transfer_index
            ]

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())

            i += 1

    return x, nfe, history


def get_transfer_index(
    logits,
    temperature,
    remasking,
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

    if remasking.startswith("confidence"):
        # get probabilities of selected ids (confidence)
        # x0_p = torch.squeeze(
        #     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        pass
    elif remasking.startswith("topk_margin"):
        x0_p = None
        sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        # Calculate confidence as top1 - top2
        x0_p = top1_probs - top2_probs
    elif remasking.startswith("entropy"):
        x0_p = None
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        x0_p = torch.sum(p * log_probs, dim=-1)
    elif remasking.startswith("random"):
        x0_p = None
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

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
    remasking,
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

    if remasking.startswith("confidence"):
        # get probabilities of selected ids (confidence)
        # x0_p = torch.squeeze(
        #     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        pass
    elif remasking.startswith("topk_margin"):
        sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        # Calculate confidence as top1 - top2
        x0_p = top1_probs - top2_probs
    elif remasking.startswith("entropy"):
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        x0_p = torch.sum(p * log_probs, dim=-1)
    elif remasking.startswith("random"):
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

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
