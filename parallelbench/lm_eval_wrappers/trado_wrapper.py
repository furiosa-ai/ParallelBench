"""lm-eval wrapper for TraDo (Transformer Diffusion) models."""

from __future__ import annotations

from typing import Optional

from lm_eval.api.registry import register_model

from parallelbench.models.base_model import BaseModel

# NOTE: Uses direct import for explicit model class selection.
from parallelbench.models.local.trado.trado_model import TradoModel
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase


@register_model("parallelbench_trado")
class TradoWrapper(DLLMBase):
    """lm-eval wrapper around TradoModel.

    Extra model_args:
        top_p: float             - Top-p sampling threshold
        top_k: float             - Top-k sampling threshold
    """

    def __init__(
        self,
        model_path: str,
        accel_framework: Optional[str] = None,
        remasking: str = "random",
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        **kwargs,
    ) -> None:
        self._top_p = top_p
        self._top_k = top_k
        super().__init__(
            model_path=model_path,
            accel_framework=accel_framework,
            remasking=remasking,
            **kwargs,
        )

    def _create_inner_model(self) -> BaseModel:
        return TradoModel(
            model_name=self.model_path,
            accel_framework=self.accel_framework,
        )

    def _build_generation_config(self) -> dict:
        config = super()._build_generation_config()
        if self._top_p is not None:
            config["top_p"] = self._top_p
        if self._top_k is not None:
            config["top_k"] = self._top_k
        return config
