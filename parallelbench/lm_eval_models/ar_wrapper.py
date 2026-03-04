"""lm-eval wrapper for autoregressive models (Transformers/vLLM backends)."""

from __future__ import annotations

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from parallelbench.model.base_model import BaseModel, DLLMOutput
from parallelbench.model.local.transformers_model import TransformersModel
from parallelbench.model.local.vllm_model import vllmModel
from parallelbench.lm_eval_models.dllm_base import DLLMBase
from parallelbench.lm_eval_models.metadata_store import (
    GenerationMetadata,
    MetadataStore,
)
from parallelbench.utils.perf_utils import pop_perf_stats


@register_model("parallelbench_ar")
class ARWrapper(DLLMBase):
    """lm-eval wrapper for autoregressive models.

    model_args:
        backend: str  - "transformers" or "vllm" (default: "transformers")
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        **kwargs,
    ) -> None:
        self._backend = backend
        # AR models don't use dLLM-specific params
        kwargs.pop("remasking", None)
        kwargs.pop("steps", None)
        kwargs.pop("block_length", None)
        kwargs.pop("alg_temp", None)
        kwargs.pop("alg_threshold", None)
        kwargs.pop("alg_factor", None)
        kwargs.pop("output_history", None)
        super().__init__(model_path=model_path, output_history=False, **kwargs)

    def _create_inner_model(self) -> BaseModel:
        if self._backend == "vllm":
            return vllmModel(model_name=self.model_path)
        return TransformersModel(model_name=self.model_path)

    def _build_generation_config(self) -> dict:
        return {
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """AR models generate without history tracking."""
        results: list[str] = []
        store = MetadataStore.instance()

        for request in requests:
            context, gen_kwargs = request.args
            messages = self._context_to_messages(context)
            gen_config = self._build_generation_config()

            dllm_output: DLLMOutput = self._inner_model.generate(
                messages=messages,
                gen_config=gen_config,
                output_history=False,
            )

            perf_stats = pop_perf_stats(flatten=True)

            store.append(
                GenerationMetadata(
                    nfe=dllm_output.nfe,
                    input_length=dllm_output.input_length,
                    output_length=dllm_output.output_length,
                    perf_stats=perf_stats,
                )
            )

            generated_text = dllm_output.output
            generated_text = self._apply_until_truncation(generated_text, gen_kwargs)
            results.append(generated_text)

        return results
