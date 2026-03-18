"""
Confidence score functions for unmasking strategies.

Each scorer takes the probability distribution and pre-computed values,
and returns a per-token confidence tensor used to determine unmask order.
"""

import torch


def max_probability(
    p: torch.Tensor, x0: torch.Tensor, x0_p: torch.Tensor
) -> torch.Tensor:
    """Use the pre-computed max/sampled probability as confidence."""
    return x0_p


def margin(p: torch.Tensor, x0: torch.Tensor, x0_p: torch.Tensor) -> torch.Tensor:
    """Confidence = gap between top-1 and top-2 probabilities."""
    sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
    top1_probs = sorted_probs[..., 0]
    top2_probs = sorted_probs[..., 1]
    return top1_probs - top2_probs


def negative_entropy(
    p: torch.Tensor, x0: torch.Tensor, x0_p: torch.Tensor
) -> torch.Tensor:
    """Confidence = negative entropy (more concentrated = more confident)."""
    epsilon = 1e-10
    log_probs = torch.log(p + epsilon)
    return torch.sum(p * log_probs, dim=-1)


def random_confidence(
    p: torch.Tensor, x0: torch.Tensor, x0_p: torch.Tensor
) -> torch.Tensor:
    """Uniform random confidence (baseline)."""
    return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
