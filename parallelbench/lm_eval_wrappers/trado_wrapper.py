"""lm-eval wrapper for TraDo (Transformer Diffusion) models."""

from __future__ import annotations

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
        return TradoModel(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        # Default block_length=4 for TrADo block diffusion.
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
