"""
Test model registration with ModelRegistry.

Tests verify that all model classes are correctly registered with their
matcher functions and can be retrieved via ModelRegistry.get_model_class().
"""

import pytest
from model.registry import ModelRegistry

# Import all model modules at module level to trigger decorator registration
import model.api.mercury_model
import model.local.dream.dream_model
import model.local.trado.trado_model
import model.local.llada.llada_model
import model.api.anthropic_model

SEDD_AVAILABLE = False
try:
    import model.local.sedd.sedd_model

    SEDD_AVAILABLE = True
except ImportError:
    # SeddModel has external dependencies that may not be available
    pass


@pytest.mark.parametrize(
    "model_name,expected_class_name",
    [
        # MercuryModel: exact matches
        ("mercury", "MercuryModel"),
        ("mercury-coder", "MercuryModel"),
        # DreamModel: exact matches
        ("Dream-org/Dream-v0-Instruct-7B", "DreamModel"),
        ("Dream-org/Dream-Coder-v0-Instruct-7B", "DreamModel"),
        ("apple/DiffuCoder-7B-Instruct", "DreamModel"),
        ("apple/DiffuCoder-7B-cpGRPO", "DreamModel"),
        # TradoModel: exact matches and SDAR substring
        ("Gen-Verse/TraDo-4B-Instruct", "TradoModel"),
        ("Gen-Verse/TraDo-8B-Instruct", "TradoModel"),
        ("Gen-Verse/TraDo-8B-Thinking", "TradoModel"),
        # LladaModel: exact matches and case-insensitive substring
        ("GSAI-ML/LLaDA-8B-Instruct", "LladaModel"),
        ("GSAI-ML/LLaDA-1.5", "LladaModel"),
        # AnthropicModel: prefix match
        ("claude-3-haiku", "AnthropicModel"),
        # SeddModel: substring match
        ("some-sedd-model", "SeddModel"),
    ],
)
def test_model_registration(model_name, expected_class_name):
    """Test that ModelRegistry.get_model_class() returns the correct class for each model name."""
    # Skip SeddModel tests if not available
    if expected_class_name == "SeddModel" and not SEDD_AVAILABLE:
        pytest.skip("SeddModel not available (missing dependencies)")

    result_class = ModelRegistry.get_model_class(model_name)
    assert result_class.__name__ == expected_class_name, (
        f"Expected {expected_class_name} for '{model_name}', got {result_class.__name__}"
    )


def test_unknown_model_raises_error():
    """Test that an unknown model name raises ValueError."""
    with pytest.raises(
        ValueError, match="No model class found for model name: unknown-model"
    ):
        ModelRegistry.get_model_class("unknown-model")
