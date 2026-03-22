"""lm-eval wrapper for autoregressive models (Transformers/vLLM backends)."""

from __future__ import annotations

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from parallelbench.models.base_model import BaseModel, DLLMOutput

# NOTE: Uses direct imports instead of ModelRegistry dispatch for explicit backend selection.
# See https://github.com/<owner>/ParallelBench/issues/XX for potential refactor.
from parallelbench.models.local.transformers_model import TransformersModel
from parallelbench.models.local.vllm_model import vllmModel
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase
from parallelbench.lm_eval_wrappers.metadata_store import (
    GenerationMetadata,
    MetadataStore,
)


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
        super().__init__(model_path=model_path, output_history=False, **kwargs)

    def _create_inner_model(self) -> BaseModel:
        if self._backend == "vllm":
            return vllmModel(model_name=self.model_path)
        return TransformersModel(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        return {
            "max_tokens": int(gen_kwargs.get("max_tokens", 128)),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
        }

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """AR models generate without history tracking."""
        results: list[str] = []
        store = MetadataStore.instance()

        for request in requests:
            context, gen_kwargs = request.args
            messages = self._context_to_messages(context)
            gen_config = self._build_generation_config(gen_kwargs)

            dllm_output: DLLMOutput = self._inner_model.generate(
                messages=messages,
                gen_config=gen_config,
                output_history=False,
            )

            store.append(
                GenerationMetadata(
                    nfe=dllm_output.nfe,
                    input_length=dllm_output.input_length,
                    output_length=dllm_output.output_length,
                )
            )

            generated_text = dllm_output.output
            generated_text = self._apply_until_truncation(generated_text, gen_kwargs)
            results.append(generated_text)

        return results
