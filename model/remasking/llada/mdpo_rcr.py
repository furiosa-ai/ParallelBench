


import numpy as np
import torch
import torch.nn.functional as F



def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    '''
    if temperature == 0.:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def gamma_func(r, mode="cosine", total_num=512):
    # from https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py#L90 and https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py#L21
    if mode == "linear":
        mask_ratio = 1 - r
    elif mode == "cosine":
        mask_ratio = np.cos(r * np.pi / 2)
    elif "pow" in mode:
        exponent = float(mode.replace("pow", ""))
        mask_ratio = 1 - r ** exponent
    elif mode == "log":
        mask_ratio = -np.log2(r) / np.log2(total_num)
    elif mode == "exp":
        mask_ratio = 1 - np.exp2(-np.log2(total_num) * (1-r))
    else:
        raise NotImplementedError
    mask_ratio = np.clip(mask_ratio, 1e-6, 1)
    return mask_ratio

def get_num_transfer_tokens_maskgit(mask_index, steps, mode="linear"):
    '''
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    '''
    total_num = mask_index.sum(dim=1, keepdim=True)
    #TODO: support batch_size>1 for gamma_func
    ratios = [[gamma_func((t+1) / steps, mode=mode, total_num=total_num_item.item()) for t in range(steps)] for total_num_item in total_num[:, 0]]
    num_transfer_tokens = total_num.expand((total_num.shape[0], steps))
    mask_ratios = torch.tensor(ratios).to(mask_index.device)
    num_transfer_tokens = total_num - torch.floor(num_transfer_tokens * mask_ratios)
    num_transfer_tokens = torch.cat([num_transfer_tokens[:, 0:1], num_transfer_tokens[:, 1:] - num_transfer_tokens[:, :-1]], axis=1)
    return num_transfer_tokens.to(torch.int64)


def generate_dlm(model, prompt, steps=64, gen_length=128, block_length=32, temperature=0., alg_temp=0.0,
                 cfg_scale=0., remasking='low_confidence', mask_id=126336, implicit_diffusion=False,
                 overtime_conf=False, mode="linear", output_history=False):
    assert temperature == 0
    assert alg_temp == 0

    '''
    Optimized version of the generate function.
    '''
    history = [] if output_history else None

    input_length = prompt.shape[1]

    nfe = 0

    # Use mixed precision for faster computation
    with torch.amp.autocast("cuda", enabled=True):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        intermediate_inputs = []
        intermediate_results = []
        intermediate_confidence = []
        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)
        overtime_confidence = torch.zeros_like(x, dtype=torch.float32)
        if implicit_diffusion:
            logits = model(x, diffusion_steps=steps).logits
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x = torch.argmax(logits_with_noise, dim=-1)
        else:
            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = (x[:, start_idx:end_idx] == mask_id)
                num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block, mode=mode)

                for i in range(steps_per_block):
                    mask_index = (x == mask_id)
                    intermediate_inputs.append(x.clone().cpu())
                    # Handle classifier-free guidance more efficiently
                    if cfg_scale > 0.:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)

                        # Get logits in a single forward pass
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x).logits

                    # Apply Gumbel noise for sampling
                    logits_with_noise = add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    # Handle remasking strategy
                    if remasking == 'low_confidence':
                        # Use float32 instead of float64 for better performance
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    elif remasking == 'random':
                        x0_p = torch.rand(x0.shape, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    if not overtime_conf:
                        intermediate_confidence.append(x0_p.clone().cpu())
                    # Ensure we don't process tokens beyond the current block
                    x0_p[:, end_idx:] = -np.inf
                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    intermediate_results.append(x0.clone().cpu())
                    # valid_token_mask = x0 != 198
                    # confidence = torch.where(torch.logical_and(mask_index, valid_token_mask), x0_p, torch.tensor(-np.inf, device=x0.device))
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                    # Select tokens to transfer based on confidence
                    for j in range(confidence.shape[0]):
                        num_tokens = num_transfer_tokens[j, i].item()
                        if overtime_conf:
                            # if confidence[j][mask_index[j]].min() > 0.3:
                            #     select_indices = (torch.where(confidence < 0.3, -np.inf, confidence)[j] != -np.inf).nonzero(as_tuple=True)[0]
                            # else:
                            #     len(confidence[j][mask_index[j]]) * 0.5
                            _, select_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, i:].sum().item())
                            # if len(select_indices) < num_tokens:
                            #     _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            x[j, select_indices] = x0[j, select_indices]
                            overtime_confidence[j, select_indices] = confidence[j, select_indices].clone()
                            # if (x[j,:] == mask_id).sum() <= 0:
                            if i != (steps_per_block - 1):
                                overtime_conf_wo_zeros = \
                                    torch.where(overtime_confidence == 0.0, 1.0, overtime_confidence)[j]
                                num_tokens_to_mask = num_transfer_tokens[j, i + 1:].sum().item()
                                # if num_tokens_to_mask < 0:
                                #     break
                                # threshold_p = 0.9
                                # overtime_conf_wo_zeros = torch.where(overtime_conf_wo_zeros > threshold_p, 1.0,
                                #                                      overtime_conf_wo_zeros)
                                # if overtime_conf_wo_zeros[overtime_conf_wo_zeros < threshold_p].shape[
                                #     0] < num_tokens_to_mask:
                                #     num_tokens_to_mask = \
                                #     overtime_conf_wo_zeros[overtime_conf_wo_zeros < threshold_p].shape[0]
                                _, mask_select_indices = torch.topk(overtime_conf_wo_zeros, k=num_tokens_to_mask,
                                                                    largest=False)
                                if len(mask_select_indices) == 0:
                                    break
                                x[j, mask_select_indices] = mask_id
                        else:
                            if num_tokens > 0:
                                _, select_indices = torch.topk(confidence[j], k=num_tokens)
                                x[j, select_indices] = x0[j, select_indices]

                    if history is not None:
                        history.append(x[:, input_length:].cpu().clone())  # clone

                    if overtime_conf:
                        intermediate_confidence.append(overtime_confidence.clone().cpu())
        # return x, intermediate_results, intermediate_confidence, intermediate_inputs
        return x, nfe, history
