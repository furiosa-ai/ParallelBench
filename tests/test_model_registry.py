"""
Tests for ModelRegistry class.

TDD RED Phase: All tests should fail initially.
"""

import pytest


class TestModelRegistry:
    """Test suite for ModelRegistry class."""

    def setup_method(self):
        """Reset registry before each test for isolation."""
        from model.registry import ModelRegistry

        ModelRegistry.clear()

    def test_registered_class_can_be_retrieved(self):
        """Test that after decorating a class, get_model_class() returns that class when matcher returns True."""
        from model.registry import ModelRegistry

        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:
            pass

        # get_model_class should return the class itself
        retrieved_class = ModelRegistry.get_model_class("test_model")
        assert retrieved_class is TestModel, (
            "get_model_class() should return the registered class"
        )

    def test_no_match_raises_value_error(self):
        """Test that when no matcher matches, get_model_class() raises ValueError."""
        from model.registry import ModelRegistry

        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:
            pass

        # Requesting a non-matching model should raise ValueError
        with pytest.raises(ValueError, match="No model class found"):
            ModelRegistry.get_model_class("non_existent_model")

    def test_multiple_registrations_ambiguous_raises_error(self):
        """Test that multiple matching registrations raise ValueError."""
        from model.registry import ModelRegistry

        @ModelRegistry.register(matcher=lambda name: "test" in name)
        class FirstModel:
            pass

        @ModelRegistry.register(matcher=lambda name: "test" in name)
        class SecondModel:
            pass

        with pytest.raises(ValueError, match="Ambiguous model name"):
            ModelRegistry.get_model_class("test_model")

    def test_registry_can_be_cleared_for_test_isolation(self):
        """Test that registry can be reset with clear() for test isolation."""
        from model.registry import ModelRegistry

        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:
            pass

        # Should work before clear
        assert ModelRegistry.get_model_class("test_model") is TestModel

        # After clear, should raise ValueError
        ModelRegistry.clear()
        with pytest.raises(ValueError):
            ModelRegistry.get_model_class("test_model")
