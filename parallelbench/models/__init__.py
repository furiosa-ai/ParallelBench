"""ParallelBench model system.

Heavy imports (torch, transformers, vllm) are deferred to first access
so that lightweight consumers (analysis, export CLI) can import from
this package without GPU dependencies.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Unmasking registry — no heavy dependencies
from parallelbench.models.unmasking_registry import (
    UNMASKING_REGISTRY,
    MethodInfo,
    StrategyInfo,
    get_all_methods,
    get_all_strategies,
    get_confidence_fn,
    get_representative_param,
    get_method_info,
    get_method_type,
    get_strategy_info,
    get_strategy_type,
    register_method,
    register_strategy,
)

if TYPE_CHECKING:
    from parallelbench.models.base_model import ApiModel, BaseModel, LocalModel
    from parallelbench.models.registry import ModelRegistry
    from parallelbench.models.local import (
        DreamModel,
        LladaModel,
        SdarModel,
        SeddModel,
        TradoModel,
    )
    from parallelbench.models.api import AnthropicModel, MercuryModel
    from parallelbench.models.local.transformers_model import TransformersModel
    from parallelbench.models.local.vllm_model import vllmModel


__all__ = [
    "BaseModel",
    "ApiModel",
    "LocalModel",
    "ModelRegistry",
    "DreamModel",
    "LladaModel",
    "SdarModel",
    "SeddModel",
    "TradoModel",
    "AnthropicModel",
    "MercuryModel",
    "TransformersModel",
    "vllmModel",
    "UNMASKING_REGISTRY",
    "MethodInfo",
    "StrategyInfo",
    "get_all_methods",
    "get_all_strategies",
    "get_confidence_fn",
    "get_representative_param",
    "get_method_info",
    "get_method_type",
    "get_strategy_info",
    "get_strategy_type",
    "register_method",
    "register_strategy",
    "load_model",
]

# Lazy-loaded modules for heavy dependencies
_LAZY_IMPORTS = {
    "BaseModel": "parallelbench.models.base_model",
    "ApiModel": "parallelbench.models.base_model",
    "LocalModel": "parallelbench.models.base_model",
    "ModelRegistry": "parallelbench.models.registry",
    "DreamModel": "parallelbench.models.local",
    "LladaModel": "parallelbench.models.local",
    "SdarModel": "parallelbench.models.local",
    "SeddModel": "parallelbench.models.local",
    "TradoModel": "parallelbench.models.local",
    "AnthropicModel": "parallelbench.models.api",
    "MercuryModel": "parallelbench.models.api",
    "TransformersModel": "parallelbench.models.local.transformers_model",
    "vllmModel": "parallelbench.models.local.vllm_model",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    if name == "load_model":
        return load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_model(model_name: str, **kwargs) -> "BaseModel":
    """Load a model from the specified model name with given model arguments.

    Dispatch order:
    1. Try ModelRegistry (name-based lookup)
    2. Fall back to accel_framework parameter (vllm, transformers)
    3. Raise ValueError if no match

    Args:
        model_name: Name or path of the model.
        **kwargs: Additional arguments. May include 'accel_framework' for dispatch.

    Returns:
        An instance of the loaded model.

    Raises:
        ValueError: If model_name is not supported.
    """
    from parallelbench.models.registry import ModelRegistry
    from parallelbench.models.local.transformers_model import TransformersModel
    from parallelbench.models.local.vllm_model import vllmModel

    accel_framework = kwargs.pop("accel_framework", None)

    try:
        model_class = ModelRegistry.get_model_class(model_name)
    except ValueError:
        model_class = None

    if model_class:
        return model_class(model_name, **kwargs)
    elif accel_framework == "vllm":
        return vllmModel(model_name, **kwargs)
    elif accel_framework == "transformers":
        return TransformersModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
