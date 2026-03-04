
from scipy.stats import kendalltau, spearmanr
import torch


def decode_history(tokenizer, history, input_length=0):
    if history is None:
        return None
    return [tokenizer.decode(h.squeeze(0)[input_length:], skip_special_tokens=False) for h in history]


def compute_history_decoding_order(tokenizer, history, ignore_pad=False):
    if isinstance(history, list) and isinstance(history[0], str):
        history = [tokenizer.encode(h, return_tensors="pt") for h in history]

    if isinstance(history, list):
        history_ids = torch.cat(history)
    else:
        history_ids = history

    steps, num_tokens = history_ids.shape

    if ignore_pad:
        # check padding of final output
        pad_mask = history_ids[-1] != tokenizer.pad_token_id
        history_ids = history_ids[:, pad_mask]
        num_tokens = history_ids.shape[1]

    decoding_order = torch.full_like(history_ids, 2**31-1)

    is_unmasked = history_ids != tokenizer.mask_token_id
    ind = torch.arange(steps, device=history_ids.device).unsqueeze(1).expand(steps, num_tokens)
    decoding_order[is_unmasked] = ind[is_unmasked]
    decoding_order = decoding_order.min(0).values

    return decoding_order


def compute_decoding_order_correlation(decoding_order):
    if isinstance(decoding_order, torch.Tensor):
        decoding_order = decoding_order.cpu().numpy()

    a = decoding_order.copy()
    a.sort()

    # Compute Kendall's tau
    kendall_corr, _ = kendalltau(a, decoding_order)
    # Compute Spearman's rank correlation
    spearman_corr, _ = spearmanr(a, decoding_order)
    return {
        "dec_order_kendall": float(kendall_corr),
        "dec_order_spearman": float(spearman_corr),
    }


def compute_decoding_order_correlation_from_history(tokenizer, history):
    decoding_order_corrs = {}

    for ignore_pad in [True, False]:
        decoding_order = compute_history_decoding_order(tokenizer, history, ignore_pad=ignore_pad)
        corr = compute_decoding_order_correlation(decoding_order)
        if ignore_pad:
            corr = {f"{k}_ignore_pad": v for k, v in corr.items()}
        decoding_order_corrs.update(corr)
    return decoding_order.tolist(), decoding_order_corrs
