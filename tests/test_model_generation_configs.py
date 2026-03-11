"""Tests for model-specific GenerationConfig subclasses (Llada, Dream, Trado)."""

import pytest

from parallelbench.models.local.llada.llada_model import LladaGenerationConfig
from parallelbench.models.local.dream.dream_model import DreamGenerationConfig
from parallelbench.models.local.trado.trado_model import TradoGenerationConfig


# ============================================================
# LladaGenerationConfig
# ============================================================


def test_llada_defaults():
    config = LladaGenerationConfig()
    assert config.unmasking == "confidence_topk"
    assert config.block_length == 128


# ============================================================
# DreamGenerationConfig
# ============================================================


def test_dream_default_unmasking():
    config = DreamGenerationConfig()
    assert config.unmasking == "origin"


def test_dream_temperature_zero_nullifies_top_p_top_k():
    config = DreamGenerationConfig(temperature=0.0, top_p=0.9, top_k=50)
    assert config.top_p is None
    assert config.top_k is None


def test_dream_temperature_positive_preserves_top_p_top_k():
    config = DreamGenerationConfig(
        unmasking="random",
        temperature=0.8,
        top_p=0.9,
        top_k=50,
    )
    assert config.top_p == 0.9
    assert config.top_k == 50


def test_dream_to_generation_kwargs_uses_alg_and_max_new_tokens():
    config = DreamGenerationConfig(unmasking="random", max_tokens=256)
    kwargs = config.to_generation_kwargs()
    # Dream uses "alg" instead of "unmasking"
    assert "alg" in kwargs
    assert kwargs["alg"] == "random"
    # Dream uses "max_new_tokens" instead of "gen_length"
    assert "max_new_tokens" in kwargs
    assert kwargs["max_new_tokens"] == 256
    assert "gen_length" not in kwargs
    assert "unmasking" not in kwargs


def test_dream_to_generation_kwargs_includes_return_dict():
    config = DreamGenerationConfig()
    kwargs = config.to_generation_kwargs()
    assert kwargs["return_dict_in_generate"] is True
    assert "attention_mask" in kwargs


# ============================================================
# TradoGenerationConfig
# ============================================================


def test_trado_default_unmasking():
    config = TradoGenerationConfig()
    assert config.unmasking == "confidence_threshold"


def test_trado_default_alg_threshold():
    config = TradoGenerationConfig()
    assert config.alg_threshold == 0.85


def test_trado_valid_strategies():
    config = TradoGenerationConfig()
    assert config.valid_strategies == {"confidence_topk", "confidence_threshold"}


def test_trado_invalid_strategy_random_raises():
    with pytest.raises(ValueError, match="Unsupported unmasking strategy"):
        TradoGenerationConfig(unmasking="random")


def test_trado_to_generation_kwargs_format():
    config = TradoGenerationConfig()
    kwargs = config.to_generation_kwargs()
    assert "top_k" in kwargs
    assert kwargs["top_k"] == 0.0
    assert "top_p" in kwargs
    assert kwargs["top_p"] == 1.0
    assert "threshold" in kwargs
    assert kwargs["threshold"] == 0.85
    assert "gen_length" in kwargs
    assert "unmasking" in kwargs
