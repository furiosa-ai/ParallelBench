"""lm-eval wrapper for LLaDA (Large Language Diffusion with mAsking) models."""

from __future__ import annotations

from lm_eval.api.registry import register_model

from parallelbench.models.base_model import BaseModel

# NOTE: Uses direct import for explicit model class selection.
from parallelbench.models.local.llada.llada_model import LladaModel
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase


@register_model("parallelbench_llada")
class LLaDAWrapper(DLLMBase):
    """lm-eval wrapper around LladaModel."""

    def _create_inner_model(self) -> BaseModel:
        return LladaModel(
            model_name=self.model_path,
            accel_framework=self.accel_framework,
        )

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        if "remasking" not in gen_kwargs:
            gen_kwargs = {**gen_kwargs, "remasking": "confidence_topk"}
        return super()._build_generation_config(gen_kwargs)
