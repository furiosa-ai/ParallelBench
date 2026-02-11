import inspect

from model.base_model import BaseModel
# Local models
from model.local import DreamModel, LladaModel, TradoModel, SeddModel
# API models
from model.api import AnthropicModel, MercuryModel
from model.registry import ModelRegistry
from model.local.transformers_model import TransformersModel
from model.local.vllm_model import vllmModel
# Defer SeddModel import to avoid circular dependency - it will be registered when first imported


__all__ = [
    "DreamModel",
    "LladaModel",
    "TradoModel",
    "SeddModel",
    "AnthropicModel",
    "MercuryModel",
    "TransformersModel",
    "vllmModel",
]


def load_model(model_name: str, **kwargs) -> "BaseModel":
    """
    Load a model from the specified model name with given model arguments.

    Dispatch order:
    1. Try ModelRegistry (name-based lookup)
    2. Fall back to accel_framework parameter (vllm, transformers)
    3. Raise ValueError if no match

    Args:
        model_name (str): Name or path of the model.
        **kwargs: Additional arguments. May include 'accel_framework'.

    Returns:
        BaseModel: An instance of the loaded model.

    Raises:
        ValueError: If model_name is not supported.
    """
    # Pop accel_framework first to avoid passing to models that don't accept it
    accel_framework = kwargs.pop("accel_framework", None)

    # Try registry first
    try:
        model_class = ModelRegistry.get_model_class(model_name)
    except ValueError:
        model_class = None

    if model_class:
        # Check if model accepts accel_framework parameter
        sig = inspect.signature(model_class.__init__)
        if "accel_framework" in sig.parameters:
            return model_class(model_name, accel_framework=accel_framework, **kwargs)
        else:
            return model_class(model_name, **kwargs)
    elif accel_framework == "vllm":
        return vllmModel(model_name, **kwargs)
    elif accel_framework == "transformers":
        return TransformersModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
