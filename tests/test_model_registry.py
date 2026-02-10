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

    def test_registry_has_register_classmethod(self):
        """Test that ModelRegistry has a register() classmethod that returns a decorator."""
        from model.registry import ModelRegistry

        # register should be callable and return a decorator
        decorator = ModelRegistry.register(matcher=lambda name: name == "test_model")
        assert callable(decorator), "register() should return a callable decorator"

    def test_register_decorator_takes_matcher_callable(self):
        """Test that register() decorator takes a matcher callable (model_name -> bool)."""
        from model.registry import ModelRegistry

        # matcher should be a callable that accepts model_name and returns bool
        def matcher_func(model_name: str) -> bool:
            return model_name == "test_model"

        decorator = ModelRegistry.register(matcher=matcher_func)
        assert callable(decorator), "register() with matcher should return a decorator"

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

    def test_multiple_registrations_first_matching_wins(self):
        """Test that multiple registrations work and first matching wins (registration order)."""
        from model.registry import ModelRegistry

        @ModelRegistry.register(matcher=lambda name: "test" in name)
        class FirstModel:
            pass

        @ModelRegistry.register(matcher=lambda name: "test" in name)
        class SecondModel:
            pass

        # First matching should win
        retrieved_class = ModelRegistry.get_model_class("test_model")
        assert retrieved_class is FirstModel, (
            "First registered matching class should win"
        )

    def test_get_model_class_returns_class_not_instance(self):
        """Test that get_model_class() returns the CLASS, not an instance."""
        from model.registry import ModelRegistry

        @ModelRegistry.register(matcher=lambda name: name == "test_model")
        class TestModel:
            def __init__(self, value):
                self.value = value

        retrieved = ModelRegistry.get_model_class("test_model")

        # Should be the class itself
        assert retrieved is TestModel, "Should return class, not instance"
        assert callable(retrieved), "Class should be callable"

        # We should be able to instantiate it
        instance = retrieved(value=42)
        assert instance.value == 42, "Should be able to instantiate the returned class"

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
