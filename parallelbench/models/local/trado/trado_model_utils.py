import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(
            logits < min_values, torch.full_like(logits, float("-inf")), logits
        )


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1,
        sorted_indices,
        sorted_mask,
    )
    logits = logits.masked_fill(mask_indices, float("-inf"))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]  # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0 and temperature > 0.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    if temperature == 0.0:
        token = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)  # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
    model,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    # denoising_steps=8,
    steps=0,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking="confidence_threshold",
    threshold=0.85,
    stopping_criteria_idx=None,
    output_history=False,
):
    nfe = 0
    history = [] if output_history else None
    input_length = prompt.shape[1]

    model.eval()
    input_ids = prompt  # ['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    decode_blocks = num_blocks - prefill_blocks
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[
            :, :prefill_length, :prefill_length
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )
        nfe += 1

    if steps is None:
        # threshold, set steps to max to give threshold time
        steps = decode_blocks * block_length

    assert steps % decode_blocks == 0, (
        f"Steps must be divisible by number of decode blocks. Got steps={steps}, decode_blocks={decode_blocks}"
    )
    denoising_steps = steps // decode_blocks
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block * block_length : (num_block + 1) * block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :,
            num_block * block_length : (num_block + 1) * block_length,
            : (num_block + 1) * block_length,
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[
            :, num_block * block_length : (num_block + 1) * block_length
        ]
        for step in range(denoising_steps + 1):
            mask_index = cur_x == mask_id
            if mask_index.sum() == 0:
                # Store kv cache
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                nfe += 1
                break

            # Denosing
            logits = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits
            nfe += 1

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Sampling strategy
            if remasking == "sequential":
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = (
                            mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        )
                        transfer_index[
                            j,
                            first_mask_index : first_mask_index
                            + num_transfer_tokens[step],
                        ] = True
                    else:
                        raise ValueError("No mask tokens found in the current block.")

            elif remasking == "confidence_topk":
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking == "confidence_threshold":
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        # number of high confidence tokens is larger or equal to step schedule
                        transfer_index[j] = high_conf_mask
                    else:
                        # insufficient high confidence tokens, just do confidence_topk
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            cur_x[transfer_index] = x0[transfer_index]

            x[:, num_block * block_length : (num_block + 1) * block_length] = cur_x

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())  # clone

        if stopping_criteria_idx is not None and any(
            stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx
        ):
            break

    return x, nfe, history
