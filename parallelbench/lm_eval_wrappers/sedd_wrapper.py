"""lm-eval wrapper for SEDD (Score-Entropy-Discrete-Diffusion) models."""

from __future__ import annotations

from lm_eval.api.registry import register_model

from parallelbench.models.base_model import BaseModel

# NOTE: Uses direct import for explicit model class selection.
from parallelbench.models.local.sedd.sedd_model import SeddModel
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase


@register_model("parallelbench_sedd")
class SEDDWrapper(DLLMBase):
    """lm-eval wrapper around SeddModel.

    Note: SEDD has constraints - block_length must equal max_tokens and temperature must be 1.0.
    """

    def __init__(
        self,
        model_path: str,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            **kwargs,
        )

    def _create_inner_model(self) -> BaseModel:
        return SeddModel(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        if "temperature" not in gen_kwargs:
            gen_kwargs = {**gen_kwargs, "temperature": 1.0}
        config = super()._build_generation_config(gen_kwargs)
        config["block_length"] = config["max_tokens"]
        return config
