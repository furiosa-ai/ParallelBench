"""Tests for model-specific GenerationConfig subclasses (Llada, Dream, Trado)."""

import pytest

from parallelbench.model.local.llada.llada_model import LladaGenerationConfig
from parallelbench.model.local.dream.dream_model import DreamGenerationConfig
from parallelbench.model.local.trado.trado_model import TradoGenerationConfig


# ============================================================
# LladaGenerationConfig
# ============================================================


def test_llada_defaults():
    config = LladaGenerationConfig()
    assert config.remasking == "low_confidence"
    assert config.block_length == 128
    assert "remdm" in config.valid_strategies
    assert "rcr" in config.valid_strategies


def test_llada_valid_remdm_config():
    config = LladaGenerationConfig(
        remasking="remdm",
        remdm_steps=5,
        remdm_number=3,
    )
    assert config.is_remdm_remasking is True


def test_llada_remdm_to_generation_kwargs_includes_remdm_keys():
    config = LladaGenerationConfig(
        remasking="remdm",
        remdm_steps=5,
        remdm_number=3,
    )
    kwargs = config.to_generation_kwargs()
    assert "remdm_steps" in kwargs
    assert kwargs["remdm_steps"] == 5
    assert "remdm_number" in kwargs
    assert kwargs["remdm_number"] == 3


def test_llada_remdm_missing_remdm_steps_raises():
    with pytest.raises(AssertionError, match="remdm_steps"):
        LladaGenerationConfig(
            remasking="remdm",
            remdm_number=3,
        )


def test_llada_remdm_missing_remdm_number_raises():
    with pytest.raises(AssertionError, match="remdm_number"):
        LladaGenerationConfig(
            remasking="remdm",
            remdm_steps=5,
        )


def test_llada_remdm_with_nonzero_alg_temp_raises():
    with pytest.raises(AssertionError, match="alg_temp"):
        LladaGenerationConfig(
            remasking="remdm",
            remdm_steps=5,
            remdm_number=3,
            alg_temp=0.5,
        )


def test_llada_valid_rcr_config():
    config = LladaGenerationConfig(
        remasking="rcr",
        rcr_overtime_conf=True,
    )
    assert config.is_rcr_remasking is True


def test_llada_rcr_to_generation_kwargs_includes_overtime_conf():
    config = LladaGenerationConfig(
        remasking="rcr",
        rcr_overtime_conf=False,
    )
    kwargs = config.to_generation_kwargs()
    assert "overtime_conf" in kwargs
    assert kwargs["overtime_conf"] is False


def test_llada_rcr_missing_overtime_conf_raises():
    with pytest.raises(AssertionError, match="overtime_conf"):
        LladaGenerationConfig(remasking="rcr")


def test_llada_default_config_to_generation_kwargs_omits_remdm_rcr_keys():
    config = LladaGenerationConfig()
    kwargs = config.to_generation_kwargs()
    assert "remdm_steps" not in kwargs
    assert "remdm_number" not in kwargs
    assert "overtime_conf" not in kwargs


# ============================================================
# DreamGenerationConfig
# ============================================================


def test_dream_default_remasking():
    config = DreamGenerationConfig()
    assert config.remasking == "origin"


def test_dream_temperature_zero_nullifies_top_p_top_k():
    config = DreamGenerationConfig(temperature=0.0, top_p=0.9, top_k=50)
    assert config.top_p is None
    assert config.top_k is None


def test_dream_temperature_positive_preserves_top_p_top_k():
    config = DreamGenerationConfig(
        remasking="random",
        temperature=0.8,
        top_p=0.9,
        top_k=50,
    )
    assert config.top_p == 0.9
    assert config.top_k == 50


def test_dream_to_generation_kwargs_uses_alg_and_max_new_tokens():
    config = DreamGenerationConfig(remasking="random", max_tokens=256)
    kwargs = config.to_generation_kwargs()
    # Dream uses "alg" instead of "remasking"
    assert "alg" in kwargs
    assert kwargs["alg"] == "random"
    # Dream uses "max_new_tokens" instead of "gen_length"
    assert "max_new_tokens" in kwargs
    assert kwargs["max_new_tokens"] == 256
    assert "gen_length" not in kwargs
    assert "remasking" not in kwargs


def test_dream_to_generation_kwargs_includes_return_dict():
    config = DreamGenerationConfig()
    kwargs = config.to_generation_kwargs()
    assert kwargs["return_dict_in_generate"] is True
    assert "attention_mask" in kwargs


def test_dream_invalid_strategy_remdm_raises():
    with pytest.raises(ValueError, match="Unsupported remasking strategy"):
        DreamGenerationConfig(remasking="remdm")


# ============================================================
# TradoGenerationConfig
# ============================================================


def test_trado_default_remasking():
    config = TradoGenerationConfig()
    assert config.remasking == "low_confidence_threshold"


def test_trado_default_alg_threshold():
    config = TradoGenerationConfig()
    assert config.alg_threshold == 0.85


def test_trado_valid_strategies():
    config = TradoGenerationConfig()
    assert config.valid_strategies == {"low_confidence", "low_confidence_threshold"}


def test_trado_invalid_strategy_random_raises():
    with pytest.raises(ValueError, match="Unsupported remasking strategy"):
        TradoGenerationConfig(remasking="random")


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
    assert "remasking" in kwargs
