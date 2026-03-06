"""lm-eval wrapper for DREAM (Diffusion Reasoning with Entropy-Aware Masking) models."""

from __future__ import annotations

from typing import Optional

from lm_eval.api.registry import register_model

from parallelbench.model.base_model import BaseModel

# NOTE: Uses direct import for explicit model class selection.
from parallelbench.model.local.dream.dream_model import DreamModel
from parallelbench.lm_eval_models.dllm_base import DLLMBase


@register_model("parallelbench_dream")
class DreamWrapper(DLLMBase):
    """lm-eval wrapper around DreamModel.

    Extra model_args:
        eps: float               - Epsilon parameter for diffusion
        top_p: float             - Top-p sampling threshold
        top_k: float             - Top-k sampling threshold
    """

    def __init__(
        self,
        model_path: str,
        accel_framework: Optional[str] = None,
        remasking: str = "origin",
        eps: float = 0,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        **kwargs,
    ) -> None:
        self._eps = eps
        self._top_p = top_p
        self._top_k = top_k
        super().__init__(
            model_path=model_path,
            accel_framework=accel_framework,
            remasking=remasking,
            **kwargs,
        )

    def _create_inner_model(self) -> BaseModel:
        return DreamModel(
            model_name=self.model_path,
            accel_framework=self.accel_framework,
            eps=self._eps,
        )

    def _build_generation_config(self) -> dict:
        config = super()._build_generation_config()
        if self._top_p is not None:
            config["top_p"] = self._top_p
        if self._top_k is not None:
            config["top_k"] = self._top_k
        return config
