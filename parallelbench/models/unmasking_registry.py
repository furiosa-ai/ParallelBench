"""
Unmasking Method Registry for dLLM generation.

Provides a lightweight dict-based registry that classifies unmasking methods
and derives generation kwargs (steps, block_length) from high-level parameters.

# If a fourth method type is added, consider whether the derive_fn signature
# needs to become polymorphic, and whether a class hierarchy is warranted.
"""

from collections import namedtuple

from parallelbench.models.confidence_scorers import (
    left_to_right,
    margin,
    max_probability,
    negative_entropy,
    random_confidence,
)


MethodInfo = namedtuple(
    "MethodInfo",
    ["method_type", "representative_param", "derive_fn", "confidence_fn"],
    defaults=[None],
)

# Backward-compatible alias
StrategyInfo = MethodInfo


def derive_topk(k: float, max_tokens: int) -> dict:
    """
    Derive generation kwargs for top-k unmasking methods.

    Args:
        k: Number of tokens unmasked per step. Must divide max_tokens evenly.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        dict with keys "steps" and "block_length".

    Raises:
        ValueError: If max_tokens / k is not an integer.
    """
    quotient = max_tokens / k
    if quotient != int(quotient):
        raise ValueError(
            f"max_tokens ({max_tokens}) must be exactly divisible by k ({k})"
        )
    return {"steps": int(quotient), "block_length": max_tokens}


def derive_threshold(alg_threshold: float, max_tokens: int) -> dict:
    """
    Derive generation kwargs for threshold-based unmasking methods.

    Args:
        alg_threshold: Confidence threshold for unmasking.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        dict with keys "steps" and "block_length".
    """
    return {"steps": max_tokens, "block_length": max_tokens}


def derive_factor(alg_factor: float, max_tokens: int) -> dict:
    """
    Derive generation kwargs for factor-based unmasking methods.

    Args:
        alg_factor: Multiplicative factor for unmasking.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        dict with keys "steps" and "block_length".
    """
    return {"steps": max_tokens, "block_length": max_tokens}


def derive_adaptive(k: float, max_tokens: int) -> dict:
    """
    Derive generation kwargs for adaptive unmasking methods.

    Adaptive methods use steps=max_tokens (one forward pass per step)
    and block_length=max_tokens (single block). The actual number of
    tokens unmasked per step is determined dynamically by the method.

    Args:
        k: Ignored for adaptive methods (kept for API consistency).
        max_tokens: Maximum number of tokens to generate.

    Returns:
        dict with keys "steps" and "block_length".
    """
    return {"steps": max_tokens, "block_length": max_tokens}


UNMASKING_REGISTRY: dict[str, MethodInfo] = {
    "random": MethodInfo("topk", "k", derive_topk, random_confidence),
    "origin": MethodInfo("topk", "k", derive_topk),
    "confidence_topk": MethodInfo("topk", "k", derive_topk, max_probability),
    "topk_margin": MethodInfo("topk", "k", derive_topk, margin),
    "entropy_topk": MethodInfo("topk", "k", derive_topk, negative_entropy),
    "confidence_threshold": MethodInfo(
        "threshold", "alg_threshold", derive_threshold, max_probability
    ),
    "confidence_factor": MethodInfo(
        "factor", "alg_factor", derive_factor, max_probability
    ),
    "left_to_right": MethodInfo("topk", "k", derive_topk, left_to_right),
    "klass": MethodInfo("adaptive", "k", derive_adaptive, max_probability),
}


def get_method_info(name: str) -> MethodInfo:
    """
    Retrieve MethodInfo for the given unmasking method name.

    Args:
        name: The unmasking method name.

    Returns:
        MethodInfo namedtuple for the method.

    Raises:
        KeyError: If the method name is not registered.
    """
    if name not in UNMASKING_REGISTRY:
        raise KeyError(
            f"Unknown unmasking method: '{name}'. "
            f"Valid methods: {sorted(UNMASKING_REGISTRY.keys())}"
        )
    return UNMASKING_REGISTRY[name]


def get_method_type(name: str) -> str:
    """
    Return the method type ("topk", "threshold", or "factor") for the given name.

    Args:
        name: The unmasking method name.

    Returns:
        Method type string.

    Raises:
        KeyError: If the method name is not registered.
    """
    return get_method_info(name).method_type


def get_representative_param(name: str) -> str:
    """
    Return the representative parameter name for the given unmasking method.

    Args:
        name: The unmasking method name.

    Returns:
        Representative parameter name string.

    Raises:
        KeyError: If the method name is not registered.
    """
    return get_method_info(name).representative_param


def get_all_methods() -> set[str]:
    """
    Return the set of all registered unmasking method names.

    Returns:
        Set of method name strings.
    """
    return set(UNMASKING_REGISTRY.keys())


def register_method(name: str, info: MethodInfo) -> None:
    """
    Register a new unmasking method in the registry.

    Args:
        name: The method name to register.
        info: MethodInfo namedtuple describing the method.
    """
    UNMASKING_REGISTRY[name] = info


# Backward-compatible aliases
get_strategy_info = get_method_info
get_strategy_type = get_method_type
get_all_strategies = get_all_methods
register_strategy = register_method
