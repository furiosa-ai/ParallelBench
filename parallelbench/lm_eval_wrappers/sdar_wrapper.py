"""lm-eval wrapper for SDAR (Semi-autoregressive Discrete Absorbing diffusion with Refinement) models."""

from __future__ import annotations

from lm_eval.api.registry import register_model

from parallelbench.models.base_model import BaseModel

# NOTE: Uses direct import for explicit model class selection.
from parallelbench.models.local.sdar.sdar_model import SdarModel
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase


@register_model("parallelbench_sdar")
class SdarWrapper(DLLMBase):
    """lm-eval wrapper around SdarModel.

    Extra model_args:
        top_p: float             - Top-p sampling threshold
        top_k: float             - Top-k sampling threshold
    """

    def __init__(
        self,
        model_path: str,
        top_p: float | None = None,
        top_k: float | None = None,
        **kwargs,
    ) -> None:
        self._top_p = top_p
        self._top_k = top_k
        super().__init__(
            model_path=model_path,
            **kwargs,
        )

    def _create_inner_model(self) -> BaseModel:
        return SdarModel(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        # Default block_length=4 for SDAR block diffusion.
        # Derive steps from k manually since the parent's derive_topk
        # would set block_length=max_tokens which is wrong for block diffusion.
        gen_kwargs.setdefault("block_length", "4")
        if "k" in gen_kwargs and "steps" not in gen_kwargs:
            max_tokens = int(gen_kwargs.get("max_tokens", 128))
            k = int(float(gen_kwargs["k"]))
            gen_kwargs["steps"] = str(max_tokens // k)
        config = super()._build_generation_config(gen_kwargs)
        if self._top_p is not None:
            config["top_p"] = self._top_p
        if self._top_k is not None:
            config["top_k"] = self._top_k
        return config
