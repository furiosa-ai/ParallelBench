"""
KLASS (KL-Adaptive Stability Sampling) generation for Dream models.

Replaces Dream's _sample method entirely. Implements dual-gating
with KL divergence stability tracking, using Dream's shifted logits
and mask_token_id conventions.

Reference: untracked/dllm-bench/model/remasking/dream/klass.py
Paper: https://arxiv.org/abs/2511.05664
"""

from typing import Optional, Union

import torch
from torch.nn import functional as F

from parallelbench.models.local.dream.dream_model_utils import (
    DreamModelOutput,
    sample_tokens,
)


def klass_sample_dream(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config,
    generation_tokens_hook_func=None,
    generation_logits_hook_func=None,
    conf_threshold: float = 0.9,
    kl_threshold: float = 0.01,
    kl_history_length: int = 2,
) -> Union[DreamModelOutput, torch.LongTensor]:
    """KLASS generation for Dream models.

    This function replaces Dream's _sample method. It manages the full
    generation loop including block iteration, NFE counting, history tracking,
    and KLASS dual-gating logic.

    Args:
        self: The Dream model instance.
        input_ids: Input token IDs.
        attention_mask: Attention mask tensor.
        generation_config: Dream generation config.
        generation_tokens_hook_func: Optional token hook.
        generation_logits_hook_func: Optional logits hook.
        conf_threshold: Confidence threshold for unmasking.
        kl_threshold: KL divergence threshold for stability.
        kl_history_length: Number of consecutive stable steps required.

    Returns:
        DreamModelOutput or torch.LongTensor.
    """
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps if generation_config.steps is not None else 999999
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    histories = [] if (return_dict_in_generate and output_history) else None

    # Pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(
            attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
        )
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    # Vocab size for KL buffers
    V = (
        self.lm_head.out_features
        if hasattr(self, "lm_head")
        else self.config.vocab_size
    )
    kl_history_buffer = torch.zeros(
        (1, x.shape[1], kl_history_length), dtype=torch.float64, device=x.device
    )
    p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64, device=x.device)

    # Hook initialization
    if generation_tokens_hook_func is not None:
        x = generation_tokens_hook_func(None, x, None)

    for i in range(steps):
        mask_index = x == mask_token_id
        if mask_index.sum() == 0:
            break

        logits = self(x, attention_mask, tok_idx).logits
        # Dream-specific: shifted logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if generation_logits_hook_func is not None:
            logits = generation_logits_hook_func(i, x, logits)

        # Compute softmax in float64 for KL precision
        p_curr = F.softmax(logits.to(torch.float64), dim=-1)
        x0 = torch.argmax(p_curr, dim=-1)

        # Compute confidence (max probability for predicted token)
        curr_conf = torch.gather(p_curr, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        # KL divergence between current and previous step
        eps = 1e-12
        kl_current_prev = (
            p_curr * (torch.log(p_curr + eps) - torch.log(p_prev + eps))
        ).sum(dim=-1)

        # Shift KL history and insert new KL
        kl_history_buffer = torch.roll(kl_history_buffer, shifts=-1, dims=-1)
        kl_history_buffer[..., -1] = kl_current_prev

        p_prev.copy_(p_curr)

        # Dual gating: stable AND confident AND masked
        if i >= kl_history_length - 1:
            stable_mask = torch.all(kl_history_buffer < kl_threshold, dim=-1)
        else:
            stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)

        conf_mask = curr_conf >= conf_threshold
        ready_mask = stable_mask & conf_mask & mask_index

        transfer_index = torch.zeros_like(mask_index)

        # Check if any tokens are ready per batch item
        has_ready = ready_mask.any(dim=1)

        for j in range(x.size(0)):
            if has_ready[j]:
                # Unmask all ready tokens (unmask_strategy="all")
                ready_indices = torch.where(ready_mask[j])[0]
                transfer_index[j, ready_indices] = True
            else:
                # Fallback: use confidence_topk — pick the single most confident token
                mask_logits = logits[j][mask_index[j]]
                if mask_logits.numel() > 0:
                    conf_fb, x0_fb = sample_tokens(
                        mask_logits.unsqueeze(0),
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    chosen = torch.argmax(conf_fb, dim=-1)
                    global_indices = torch.where(mask_index[j])[0]
                    transfer_index[j, global_indices[chosen]] = True

        x0_full = torch.zeros_like(x)
        x0_full[mask_index] = x0[mask_index]
        x[transfer_index] = x0_full[transfer_index]

        if generation_tokens_hook_func is not None:
            x = generation_tokens_hook_func(i, x, logits)

        if histories is not None:
            histories.append(x.clone())

    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x
