
from model.anthropic_model import AnthropicModel
from model.dream_model import DreamModel
from model.mercury_model import MercuryModel
from model.llada_model import LladaModel
from model.trado_model import TradoModel
from model.transformers_model import TransformersModel
from model.vllm_model import vllmModel

def load_model(model_name, **kwargs):
    """
    Load a model from the specified path with given model arguments.
    
    Args:
        model_name (str): Path to the model.
        model_kwargs (dict, optional): Additional arguments for loading the model.
        
    Returns:
        BaseModel: An instance of the loaded model.
    """

    if model_name in ("mercury", "mercury-coder"):
        return MercuryModel(model_name, **kwargs)
    elif model_name in ("Dream-org/Dream-v0-Instruct-7B", "Dream-org/Dream-Coder-v0-Instruct-7B", "apple/DiffuCoder-7B-Instruct", "apple/DiffuCoder-7B-cpGRPO", "hub/dream-v0-instruct-tiny-random"):
        return DreamModel(model_name, **kwargs)
    elif model_name in ("Gen-Verse/TraDo-4B-Instruct", "Gen-Verse/TraDo-8B-Instruct", "Gen-Verse/TraDo-8B-Thinking") or "SDAR" in model_name:
        return TradoModel(model_name, **kwargs)
    elif model_name in ("GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-1.5", "hub/llada-1.5-tiny-random") or "llada" in model_name.lower():
        return LladaModel(model_name, **kwargs)
    elif kwargs["accel_framework"] == "vllm":
        kwargs.pop("accel_framework")
        return vllmModel(model_name, **kwargs)
    # elif kwargs["accel_framework"] == "anthropic":
    elif model_name.startswith("claude-"):
        kwargs.pop("accel_framework", None)
        return AnthropicModel(model_name, **kwargs)
    elif kwargs["accel_framework"] == "transformers":
        kwargs.pop("accel_framework")
        return TransformersModel(model_name, **kwargs)
    elif "sedd" in model_name:
        from model.sedd_model import SeddModel
        return SeddModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")