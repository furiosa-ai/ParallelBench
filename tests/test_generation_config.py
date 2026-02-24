"""Tests for GenerationConfig hierarchy."""

import pytest

from model.generation_config import (
    DllmGenerationConfig,
    ARGenerationConfig,
)


# -- BaseGenerationConfig --


def test_base_generation_config_has_common_fields():
    """BaseGenerationConfig should only have common fields (max_tokens, temperature, accel_framework)."""
    config = ARGenerationConfig()  # Use ARGenerationConfig as concrete subclass
    assert hasattr(config, "max_tokens")
    assert hasattr(config, "temperature")
    assert hasattr(config, "accel_framework")
    # dLLM-specific fields should NOT be on BaseGenerationConfig
    assert not hasattr(config, "remasking")
    assert not hasattr(config, "block_length")
    assert not hasattr(config, "steps")


# -- ARGenerationConfig --


def test_ar_generation_config_custom_values():
    config = ARGenerationConfig(max_tokens=256, temperature=0.7)
    assert config.max_tokens == 256
    assert config.temperature == 0.7


# -- DllmGenerationConfig --


def test_dllm_generation_config_valid():
    """DllmGenerationConfig with valid remasking should work."""
    config = DllmGenerationConfig(
        remasking="random",
        block_length=128,
        max_tokens=128,
        steps=128,
    )
    assert config.remasking == "random"
    assert config.is_default_remasking is True


def test_dllm_generation_config_threshold_remasking():
    """Test threshold-based remasking strategies with custom valid_strategies."""
    config = DllmGenerationConfig(
        remasking="low_confidence_threshold",
        block_length=128,
        max_tokens=128,
        steps=128,
        alg_threshold=0.5,
        valid_strategies={"low_confidence", "low_confidence_threshold"},
    )
    assert config.is_threshold_remasking is True
    assert config.is_default_remasking is False


def test_dllm_generation_config_factor_remasking():
    """Test factor-based remasking strategies with custom valid_strategies."""
    config = DllmGenerationConfig(
        remasking="low_confidence_factor",
        block_length=128,
        max_tokens=128,
        steps=128,
        alg_factor=2.0,
        valid_strategies={"low_confidence", "low_confidence_factor"},
    )
    assert config.is_factor_remasking is True


def test_dllm_generation_config_invalid_remasking():
    """Invalid remasking strategy should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported remasking strategy"):
        DllmGenerationConfig(
            remasking="invalid_strategy",
            block_length=128,
            max_tokens=128,
        )


def test_dllm_generation_config_steps_too_large():
    """steps > max_tokens should raise ValueError."""
    with pytest.raises(ValueError, match="steps cannot be greater than max_tokens"):
        DllmGenerationConfig(
            remasking="random",
            block_length=128,
            max_tokens=128,
            steps=256,
        )


def test_dllm_generation_config_num_blocks():
    config = DllmGenerationConfig(
        remasking="random",
        block_length=64,
        max_tokens=256,
        steps=64,
    )
    assert config.num_blocks == 4


def test_dllm_generation_config_num_blocks_invalid_block_length():
    """num_blocks with invalid block_length should raise ValueError."""
    config = DllmGenerationConfig.__new__(DllmGenerationConfig)
    config.max_tokens = 128
    config.block_length = -1
    with pytest.raises(ValueError, match="block_length must be a positive integer"):
        _ = config.num_blocks


def test_dllm_to_generation_kwargs():
    config = DllmGenerationConfig(
        remasking="random",
        block_length=128,
        max_tokens=128,
        steps=128,
    )
    kwargs = config.to_generation_kwargs()
    assert kwargs["gen_length"] == 128
    assert kwargs["block_length"] == 128
    assert kwargs["remasking"] == "random"


# -- TransformersGenerationConfig inherits ARGenerationConfig --


def test_transformers_generation_config_no_crash():
    """TransformersGenerationConfig (inherits ARGenerationConfig) should not crash."""
    from model.local.transformers_model import TransformersGenerationConfig

    config = TransformersGenerationConfig()
    assert config.max_tokens == 128
    kwargs = config.to_generate_kwargs()
    assert "max_new_tokens" in kwargs


