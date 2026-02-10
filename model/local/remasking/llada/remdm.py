import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions as dists


def get_num_transfer_tokens(mask_index, steps):
    if steps is None:
        return None

    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _sample_categorical(categorical_probs):
#   gumbel_norm = (
#     1e-10
#     - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
#   return (categorical_probs / gumbel_norm).argmax(dim=-1)
    return dists.Categorical(probs=categorical_probs).sample()

# https://github.com/guanghanwang/ReMDM-LLaDA
def llada_remdm_sample(model, prompt, steps=64, gen_length=128, block_length=32, temperature=0.,
                 remasking='low_confidence', remdm_steps=None, remdm_number=None, mask_id=126336, output_history=False, 
                 tokenizer=None, alg_temp=0.0, output0_ids=None,):
    assert remasking == 'low_confidence'
    # assert temperature == 0.
    assert remdm_number is not None and remdm_number > 0
    assert tokenizer is not None
    assert alg_temp == 0.0
    assert remdm_steps is not None and remdm_steps >= 0

    history = [] if output_history else None

    input_length = prompt.shape[1]

    xt = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    xt[:, :prompt.shape[1]] = prompt.clone()

    if output0_ids is not None:
        xt[:, prompt.shape[1]:] = output0_ids.clone()

    prompt_index = (xt != mask_id)
    prompt_len = prompt_index.sum(1).item()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    assert gen_length % steps == 0

    nfe = 0
    remask_thres_ratio = 7 / 8

    remask_thres = max(min(int(block_length * remask_thres_ratio), block_length - 1), 1)
    is_remasking_steps = [False] * steps_per_block
    is_remasking_steps[remask_thres:remask_thres] = [True] * remdm_steps  # insert remasking steps

    for num_block in range(num_blocks):
        conf_cache = torch.ones_like(xt, dtype=torch.float64) * np.inf

        block_mask_index = (xt[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)[0].tolist()

        for i, is_remasking_step in enumerate(is_remasking_steps):
            if is_remasking_step:
                remask_index = torch.zeros_like(xt, dtype=torch.bool, device=xt.device)
                _, mask_indices = torch.topk(conf_cache, k=remdm_number, largest=False, dim=1)
                remask_index[0, mask_indices] = True
                conf_cache[remask_index] = np.inf
                xt[remask_index] = mask_id
                
                transfer_length = remdm_number
            else:
                # does not work in case of infilling
                transfer_length = num_transfer_tokens.pop(0)

            mask_index = (xt == mask_id)
            logits = model(xt).logits
            nfe += 1
            if temperature == 0:
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p, dim=-1)
            else:
                p = F.softmax(logits.to(torch.float64) / temperature, dim=-1)
                x0 = _sample_categorical(p)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l

            x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, xt)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=transfer_length)
                transfer_index[j, select_index] = True
            xt[transfer_index] = x0[transfer_index]
            conf_cache[transfer_index] = confidence[transfer_index]

            if history is not None:
                history.append(xt[:, input_length:].cpu().clone())  # clone
            
        # if torch.sum(xt == tokenizer.eos_token_id) > 0:
        #     break

    assert (mask_id == xt).sum() == 0

    return xt, nfe, history