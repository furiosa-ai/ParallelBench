"""lm-eval wrapper for LLaDA (Large Language Diffusion with mAsking) models."""

from __future__ import annotations

from typing import Optional

from lm_eval.api.registry import register_model

from parallelbench.model.base_model import BaseModel

# FIXME: Consider using ModelRegistry.get_model_class() instead of direct import
#  to reduce coupling between lm_eval_models/ and model/ internals.
from parallelbench.model.local.llada.llada_model import LladaModel
from parallelbench.lm_eval_models.dllm_base import DLLMBase


@register_model("parallelbench_llada")
class LLaDAWrapper(DLLMBase):
    """lm-eval wrapper around LladaModel.

    Extra model_args (in addition to DLLMBase args):
        remdm_steps: int         - ReMDM variant steps
        remdm_number: int        - ReMDM variant number
        rcr_overtime_conf: bool  - RCR variant overtime confidence
    """

    def __init__(
        self,
        model_path: str,
        accel_framework: Optional[str] = None,
        remasking: str = "low_confidence",
        remdm_steps: Optional[int] = None,
        remdm_number: Optional[int] = None,
        rcr_overtime_conf: Optional[bool] = None,
        **kwargs,
    ) -> None:
        self._remdm_steps = remdm_steps
        self._remdm_number = remdm_number
        self._rcr_overtime_conf = rcr_overtime_conf
        super().__init__(
            model_path=model_path,
            accel_framework=accel_framework,
            remasking=remasking,
            **kwargs,
        )

    def _create_inner_model(self) -> BaseModel:
        return LladaModel(
            model_name=self.model_path,
            accel_framework=self.accel_framework,
        )

    def _build_generation_config(self) -> dict:
        config = super()._build_generation_config()
        if self._remdm_steps is not None:
            config["remdm_steps"] = self._remdm_steps
        if self._remdm_number is not None:
            config["remdm_number"] = self._remdm_number
        if self._rcr_overtime_conf is not None:
            config["rcr_overtime_conf"] = self._rcr_overtime_conf
        return config
