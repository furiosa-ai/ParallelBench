"""Tests for DllmGenerationConfig fast-dLLM cache validation."""

import pytest

from parallelbench.model.generation_config import DllmGenerationConfig


def test_fast_dllm_cache_allowed_with_fast_dllm_framework():
    config = DllmGenerationConfig(
        remasking="random",
        accel_framework="fast_dllm",
        use_fast_dllm_cache=True,
    )
    assert config.use_fast_dllm_cache is True


def test_fast_dllm_dual_cache_allowed_with_fast_dllm_framework():
    config = DllmGenerationConfig(
        remasking="random",
        accel_framework="fast_dllm",
        use_fast_dllm_dual_cache=True,
    )
    assert config.use_fast_dllm_dual_cache is True


def test_fast_dllm_cache_without_fast_dllm_raises():
    with pytest.raises(ValueError, match="use_fast_dllm_cache"):
        DllmGenerationConfig(
            remasking="random",
            accel_framework=None,
            use_fast_dllm_cache=True,
        )


def test_fast_dllm_dual_cache_without_fast_dllm_raises():
    with pytest.raises(ValueError, match="use_fast_dllm_dual_cache"):
        DllmGenerationConfig(
            remasking="random",
            accel_framework=None,
            use_fast_dllm_dual_cache=True,
        )


def test_fast_dllm_cache_with_non_fast_dllm_framework_raises():
    with pytest.raises(ValueError, match="use_fast_dllm_cache"):
        DllmGenerationConfig(
            remasking="random",
            accel_framework="vllm",
            use_fast_dllm_cache=True,
        )


def test_fast_dllm_both_caches_with_fast_dllm_framework():
    config = DllmGenerationConfig(
        remasking="random",
        accel_framework="fast_dllm",
        use_fast_dllm_cache=True,
        use_fast_dllm_dual_cache=True,
    )
    assert config.use_fast_dllm_cache is True
    assert config.use_fast_dllm_dual_cache is True


def test_steps_not_divisible_by_num_blocks_raises():
    # steps=3, block_length=64, max_tokens=128 -> num_blocks=2, 3 % 2 != 0
    with pytest.raises(ValueError, match="steps must be divisible by num_blocks"):
        DllmGenerationConfig(
            remasking="random",
            steps=3,
            block_length=64,
            max_tokens=128,
        )


def test_max_tokens_not_divisible_by_block_length_raises():
    with pytest.raises(ValueError, match="max_tokens must be divisible by block_length"):
        DllmGenerationConfig(
            remasking="random",
            steps=100,
            block_length=64,
            max_tokens=100,
        )
