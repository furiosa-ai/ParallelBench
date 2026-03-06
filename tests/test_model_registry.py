"""
Tests for ModelRegistry class and model registration.

Tests verify both the registry mechanism (register, get, clear, ambiguous)
and that all model classes are correctly registered with their matcher functions.
"""

import pytest

from parallelbench.model.registry import ModelRegistry

# Import all model modules at module level to trigger decorator registration
import parallelbench.model.api.mercury_model  # noqa: F401
import parallelbench.model.local.dream.dream_model  # noqa: F401
import parallelbench.model.local.trado.trado_model  # noqa: F401
import parallelbench.model.local.llada.llada_model  # noqa: F401
import parallelbench.model.api.anthropic_model  # noqa: F401

SEDD_AVAILABLE = False
try:
    import parallelbench.model.local.sedd.sedd_model  # noqa: F401

    SEDD_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# Integration: real model registration
# ============================================================


@pytest.mark.parametrize(
    "model_name,expected_class_name",
    [
        ("mercury", "MercuryModel"),
        ("mercury-coder", "MercuryModel"),
        ("Dream-org/Dream-v0-Instruct-7B", "DreamModel"),
        ("Dream-org/Dream-Coder-v0-Instruct-7B", "DreamModel"),
        ("apple/DiffuCoder-7B-Instruct", "DreamModel"),
        ("apple/DiffuCoder-7B-cpGRPO", "DreamModel"),
        ("Gen-Verse/TraDo-4B-Instruct", "TradoModel"),
        ("Gen-Verse/TraDo-8B-Instruct", "TradoModel"),
        ("Gen-Verse/TraDo-8B-Thinking", "TradoModel"),
        ("GSAI-ML/LLaDA-8B-Instruct", "LladaModel"),
        ("GSAI-ML/LLaDA-1.5", "LladaModel"),
        ("claude-3-haiku", "AnthropicModel"),
        ("some-sedd-model", "SeddModel"),
    ],
)
def test_model_registration(model_name, expected_class_name):
    """Test that ModelRegistry.get_model_class() returns the correct class for each model name."""
    if expected_class_name == "SeddModel" and not SEDD_AVAILABLE:
        pytest.skip("SeddModel not available (missing dependencies)")

    result_class = ModelRegistry.get_model_class(model_name)
    assert result_class.__name__ == expected_class_name, (
        f"Expected {expected_class_name} for '{model_name}', got {result_class.__name__}"
    )


# ============================================================
# Unit: registry mechanism
# ============================================================


class TestModelRegistry:
    """Test suite for ModelRegistry class."""

    def setup_method(self):
        """Reset registry before each test for isolation."""
        ModelRegistry.clear()

    def test_registered_class_can_be_retrieved(self):
        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:
            pass

        retrieved_class = ModelRegistry.get_model_class("test_model")
        assert retrieved_class is TestModel

    def test_no_match_raises_value_error(self):
        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:  # noqa: F841
            pass

        with pytest.raises(ValueError, match="No model class found"):
            ModelRegistry.get_model_class("non_existent_model")

    def test_multiple_registrations_ambiguous_raises_error(self):
        @ModelRegistry.register(matcher=lambda name: "test" in name)
        class FirstModel:  # noqa: F841
            pass

        @ModelRegistry.register(matcher=lambda name: "test" in name)
        class SecondModel:  # noqa: F841
            pass

        with pytest.raises(ValueError, match="Ambiguous model name"):
            ModelRegistry.get_model_class("test_model")

    def test_registry_can_be_cleared_for_test_isolation(self):
        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:
            pass

        assert ModelRegistry.get_model_class("test_model") is TestModel

        ModelRegistry.clear()
        with pytest.raises(ValueError):
            ModelRegistry.get_model_class("test_model")
