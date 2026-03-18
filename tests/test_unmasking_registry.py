"""
Tests for unmasking method registry.

Tests verify all registered methods, helper functions, derivation functions,
and extensibility via register_method().
"""

import pytest

from parallelbench.models.unmasking_registry import (
    UNMASKING_REGISTRY,
    MethodInfo,
    StrategyInfo,
    derive_factor,
    derive_threshold,
    derive_topk,
    get_all_methods,
    get_representative_param,
    get_method_info,
    get_method_type,
    register_method,
)
from parallelbench.models.local.llada.constants import LLADA_VALID_METHODS
from parallelbench.models.local.dream.constants import DREAM_VALID_METHODS
from parallelbench.models.local.trado.constants import TRADO_VALID_METHODS


# ============================================================
# Registry completeness
# ============================================================


def test_all_seven_methods_are_registered():
    """All 7 expected method names are present in the registry."""
    expected = {
        "random",
        "origin",
        "confidence_topk",
        "topk_margin",
        "entropy_topk",
        "confidence_threshold",
        "confidence_factor",
    }
    assert expected == set(UNMASKING_REGISTRY.keys())


def test_get_all_methods_matches_registry_keys():
    assert get_all_methods() == set(UNMASKING_REGISTRY.keys())


def test_registry_is_superset_of_per_model_valid_methods():
    """The registry must cover every method accepted by each model."""
    all_methods = get_all_methods()
    assert LLADA_VALID_METHODS.issubset(all_methods)
    assert DREAM_VALID_METHODS.issubset(all_methods)
    assert TRADO_VALID_METHODS.issubset(all_methods)


# ============================================================
# get_method_type
# ============================================================


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("random", "topk"),
        ("origin", "topk"),
        ("confidence_topk", "topk"),
        ("topk_margin", "topk"),
        ("entropy_topk", "topk"),
        ("confidence_threshold", "threshold"),
        ("confidence_factor", "factor"),
    ],
)
def test_get_method_type(name, expected_type):
    assert get_method_type(name) == expected_type


# ============================================================
# get_representative_param
# ============================================================


def test_get_representative_param_for_topk_method():
    assert get_representative_param("random") == "k"


def test_get_representative_param_for_threshold_method():
    assert get_representative_param("confidence_threshold") == "alg_threshold"


def test_get_representative_param_for_factor_method():
    assert get_representative_param("confidence_factor") == "alg_factor"


# ============================================================
# get_method_info
# ============================================================


def test_get_method_info_returns_method_info_namedtuple():
    info = get_method_info("random")
    assert isinstance(info, MethodInfo)


def test_get_method_info_unknown_name_raises_key_error():
    with pytest.raises(KeyError, match="unknown_method"):
        get_method_info("unknown_method")


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
# register_method (extensibility)
# ============================================================


def test_register_method_adds_new_method_that_is_queryable():
    new_info = MethodInfo("topk", "k", derive_topk)
    register_method("test_custom_method", new_info)
    assert get_method_type("test_custom_method") == "topk"
    assert get_representative_param("test_custom_method") == "k"
    # Cleanup
    del UNMASKING_REGISTRY["test_custom_method"]


def test_register_method_appears_in_get_all_methods():
    new_info = MethodInfo("threshold", "alg_threshold", derive_threshold)
    register_method("test_threshold_method", new_info)
    assert "test_threshold_method" in get_all_methods()
    # Cleanup
    del UNMASKING_REGISTRY["test_threshold_method"]


# ============================================================
# Backward compatibility aliases
# ============================================================


def test_strategy_info_is_alias_for_method_info():
    assert StrategyInfo is MethodInfo
