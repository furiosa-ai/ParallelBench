"""
Model Registry for dynamic model class lookup.

Provides a registry pattern for registering model classes with custom matchers.
"""

from typing import Callable, List, Tuple, Type


class ModelRegistry:
    """
    Registry for model classes with custom matcher functions.

    Stores model classes with associated matcher functions that determine
    which class to return for a given model name.
    """

    _registry: List[Tuple[Callable[[str], bool], Type]] = []

    @classmethod
    def register(cls, matcher: Callable[[str], bool]) -> Callable[[Type], Type]:
        """
        Register a model class with a custom matcher function.

        Args:
            matcher: A callable that takes a model name (str) and returns bool
                    indicating whether this model class should handle that name.

        Returns:
            A decorator function that registers the decorated class.

        Example:
            @ModelRegistry.register(matcher=lambda name: name == "my_model")
            class MyModel:
                pass
        """

        def decorator(model_class: Type) -> Type:
            cls._registry.append((matcher, model_class))
            return model_class

        return decorator

    @classmethod
    def get_model_class(cls, model_name: str) -> Type:
        """
        Get the model class for a given model name.

        Iterates through registered (matcher, class) pairs in registration order
        and returns the first class whose matcher returns True.

        Args:
            model_name: The name of the model to look up.

        Returns:
            The registered model class (not an instance).

        Raises:
            ValueError: If no matcher matches the given model name.
        """
        for matcher, model_class in cls._registry:
            if matcher(model_name):
                return model_class

        raise ValueError(f"No model class found for model name: {model_name}")

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered model classes.

        Useful for test isolation to ensure a clean registry state.
        """
        cls._registry.clear()
