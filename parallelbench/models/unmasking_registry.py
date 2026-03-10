"""
Unmasking Strategy Registry for dLLM generation.

Provides a lightweight dict-based registry that classifies unmasking strategies
and derives generation kwargs (steps, block_length) from high-level parameters.

# If a fourth strategy type is added, consider whether the derive_fn signature
# needs to become polymorphic, and whether a class hierarchy is warranted.
"""

from collections import namedtuple


StrategyInfo = namedtuple(
    "StrategyInfo", ["strategy_type", "representative_param", "derive_fn"]
)


def derive_topk(k: float, max_tokens: int) -> dict:
    """
    Derive generation kwargs for top-k unmasking strategies.

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
    Derive generation kwargs for threshold-based unmasking strategies.

    Args:
        alg_threshold: Confidence threshold for unmasking.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        dict with keys "steps" and "block_length".
    """
    return {"steps": max_tokens, "block_length": max_tokens}


def derive_factor(alg_factor: float, max_tokens: int) -> dict:
    """
    Derive generation kwargs for factor-based unmasking strategies.

    Args:
        alg_factor: Multiplicative factor for unmasking.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        dict with keys "steps" and "block_length".
    """
    return {"steps": max_tokens, "block_length": max_tokens}


UNMASKING_REGISTRY: dict[str, StrategyInfo] = {
    "random": StrategyInfo("topk", "k", derive_topk),
    "origin": StrategyInfo("topk", "k", derive_topk),
    "low_confidence": StrategyInfo("topk", "k", derive_topk),
    "confidence_topk": StrategyInfo("topk", "k", derive_topk),
    "topk_margin": StrategyInfo("topk", "k", derive_topk),
    "entropy_topk": StrategyInfo("topk", "k", derive_topk),
    "confidence_threshold": StrategyInfo(
        "threshold", "alg_threshold", derive_threshold
    ),
    "confidence_factor": StrategyInfo("factor", "alg_factor", derive_factor),
}


def get_strategy_info(name: str) -> StrategyInfo:
    """
    Retrieve StrategyInfo for the given strategy name.

    Args:
        name: The unmasking strategy name.

    Returns:
        StrategyInfo namedtuple for the strategy.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    if name not in UNMASKING_REGISTRY:
        raise KeyError(
            f"Unknown unmasking strategy: '{name}'. "
            f"Valid strategies: {sorted(UNMASKING_REGISTRY.keys())}"
        )
    return UNMASKING_REGISTRY[name]


def get_strategy_type(name: str) -> str:
    """
    Return the strategy type ("topk", "threshold", or "factor") for the given name.

    Args:
        name: The unmasking strategy name.

    Returns:
        Strategy type string.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    return get_strategy_info(name).strategy_type


def get_representative_param(name: str) -> str:
    """
    Return the representative parameter name for the given strategy.

    Args:
        name: The unmasking strategy name.

    Returns:
        Representative parameter name string.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    return get_strategy_info(name).representative_param


def get_all_strategies() -> set[str]:
    """
    Return the set of all registered unmasking strategy names.

    Returns:
        Set of strategy name strings.
    """
    return set(UNMASKING_REGISTRY.keys())


def register_strategy(name: str, info: StrategyInfo) -> None:
    """
    Register a new unmasking strategy in the registry.

    Args:
        name: The strategy name to register.
        info: StrategyInfo namedtuple describing the strategy.
    """
    UNMASKING_REGISTRY[name] = info
