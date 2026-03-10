from parallelbench.models.base_model import ApiModel, BaseModel, LocalModel

# Local models
from parallelbench.models.local import DreamModel, LladaModel, TradoModel, SeddModel

# API models
from parallelbench.models.api import AnthropicModel, MercuryModel
from parallelbench.models.registry import ModelRegistry
from parallelbench.models.local.transformers_model import TransformersModel
from parallelbench.models.local.vllm_model import vllmModel

# Unmasking registry
from parallelbench.models.unmasking_registry import (
    UNMASKING_REGISTRY,
    StrategyInfo,
    get_all_strategies,
    get_representative_param,
    get_strategy_info,
    get_strategy_type,
    register_strategy,
)


__all__ = [
    "BaseModel",
    "ApiModel",
    "LocalModel",
    "ModelRegistry",
    "DreamModel",
    "LladaModel",
    "TradoModel",
    "SeddModel",
    "AnthropicModel",
    "MercuryModel",
    "TransformersModel",
    "vllmModel",
    "UNMASKING_REGISTRY",
    "StrategyInfo",
    "get_all_strategies",
    "get_representative_param",
    "get_strategy_info",
    "get_strategy_type",
    "register_strategy",
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
    accel_framework = kwargs.pop("accel_framework", None)

    # Try registry first
    try:
        model_class = ModelRegistry.get_model_class(model_name)
    except ValueError:
        model_class = None

    if model_class:
        # API models don't accept accel_framework
        if issubclass(model_class, ApiModel):
            return model_class(model_name, **kwargs)
        return model_class(model_name, accel_framework=accel_framework, **kwargs)
    elif accel_framework == "vllm":
        return vllmModel(model_name, **kwargs)
    elif accel_framework == "transformers":
        return TransformersModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
