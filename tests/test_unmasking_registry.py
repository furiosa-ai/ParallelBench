"""
Tests for unmasking strategy registry.

Tests verify all registered strategies, helper functions, derivation functions,
and extensibility via register_strategy().
"""

import pytest

from parallelbench.models.unmasking_registry import (
    UNMASKING_REGISTRY,
    StrategyInfo,
    derive_factor,
    derive_threshold,
    derive_topk,
    get_all_strategies,
    get_representative_param,
    get_strategy_info,
    get_strategy_type,
    register_strategy,
)
from parallelbench.models.local.llada.constants import LLADA_VALID_STRATEGIES
from parallelbench.models.local.dream.constants import DREAM_VALID_STRATEGIES
from parallelbench.models.local.trado.constants import TRADO_VALID_STRATEGIES


# ============================================================
# Registry completeness
# ============================================================


def test_all_eight_strategies_are_registered():
    """All 8 expected strategy names are present in the registry."""
    expected = {
        "random",
        "origin",
        "low_confidence",
        "confidence_topk",
        "topk_margin",
        "entropy_topk",
        "confidence_threshold",
        "confidence_factor",
    }
    assert expected == set(UNMASKING_REGISTRY.keys())


def test_get_all_strategies_matches_registry_keys():
    assert get_all_strategies() == set(UNMASKING_REGISTRY.keys())


def test_registry_is_superset_of_per_model_valid_strategies():
    """The registry must cover every strategy accepted by each model."""
    all_strategies = get_all_strategies()
    assert LLADA_VALID_STRATEGIES.issubset(all_strategies)
    assert DREAM_VALID_STRATEGIES.issubset(all_strategies)
    assert TRADO_VALID_STRATEGIES.issubset(all_strategies)


# ============================================================
# get_strategy_type
# ============================================================


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("random", "topk"),
        ("origin", "topk"),
        ("low_confidence", "topk"),
        ("confidence_topk", "topk"),
        ("topk_margin", "topk"),
        ("entropy_topk", "topk"),
        ("confidence_threshold", "threshold"),
        ("confidence_factor", "factor"),
    ],
)
def test_get_strategy_type(name, expected_type):
    assert get_strategy_type(name) == expected_type


# ============================================================
# get_representative_param
# ============================================================


def test_get_representative_param_for_topk_strategy():
    assert get_representative_param("random") == "k"


def test_get_representative_param_for_threshold_strategy():
    assert get_representative_param("confidence_threshold") == "alg_threshold"


def test_get_representative_param_for_factor_strategy():
    assert get_representative_param("confidence_factor") == "alg_factor"


# ============================================================
# get_strategy_info
# ============================================================


def test_get_strategy_info_returns_strategy_info_namedtuple():
    info = get_strategy_info("random")
    assert isinstance(info, StrategyInfo)


def test_get_strategy_info_unknown_name_raises_key_error():
    with pytest.raises(KeyError, match="unknown_strategy"):
        get_strategy_info("unknown_strategy")


# ============================================================
# derive_topk
# ============================================================


def test_derive_topk_returns_correct_steps_and_block_length():
    result = derive_topk(4.0, 32)
    assert result == {"steps": 8, "block_length": 32}


def test_derive_topk_larger_values():
    result = derive_topk(4.0, 64)
    assert result == {"steps": 16, "block_length": 64}


def test_derive_topk_non_integer_steps_raises_value_error():
    with pytest.raises(ValueError):
        derive_topk(3.0, 32)


def test_derive_topk_exact_division_works():
    result = derive_topk(8.0, 128)
    assert result == {"steps": 16, "block_length": 128}


# ============================================================
# derive_threshold
# ============================================================


def test_derive_threshold_returns_max_tokens_for_steps_and_block_length():
    result = derive_threshold(0.5, 64)
    assert result == {"steps": 64, "block_length": 64}


def test_derive_threshold_ignores_alg_threshold_value():
    result_a = derive_threshold(0.1, 32)
    result_b = derive_threshold(0.9, 32)
    assert result_a == result_b


# ============================================================
# derive_factor
# ============================================================


def test_derive_factor_returns_max_tokens_for_steps_and_block_length():
    result = derive_factor(2.0, 64)
    assert result == {"steps": 64, "block_length": 64}


def test_derive_factor_ignores_alg_factor_value():
    result_a = derive_factor(1.0, 32)
    result_b = derive_factor(5.0, 32)
    assert result_a == result_b


# ============================================================
# register_strategy (extensibility)
# ============================================================


def test_register_strategy_adds_new_strategy_that_is_queryable():
    new_info = StrategyInfo("topk", "k", derive_topk)
    register_strategy("test_custom_strategy", new_info)
    assert get_strategy_type("test_custom_strategy") == "topk"
    assert get_representative_param("test_custom_strategy") == "k"
    # Cleanup
    del UNMASKING_REGISTRY["test_custom_strategy"]


def test_register_strategy_appears_in_get_all_strategies():
    new_info = StrategyInfo("threshold", "alg_threshold", derive_threshold)
    register_strategy("test_threshold_strategy", new_info)
    assert "test_threshold_strategy" in get_all_strategies()
    # Cleanup
    del UNMASKING_REGISTRY["test_threshold_strategy"]
