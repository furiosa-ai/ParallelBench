"""lm-eval wrapper for API-based models (Anthropic, Mercury, etc.)."""

from __future__ import annotations

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from parallelbench.models import load_model
from parallelbench.models.base_model import BaseModel, DLLMOutput
from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase
from parallelbench.lm_eval_wrappers.metadata_store import (
    GenerationMetadata,
    MetadataStore,
)


@register_model("parallelbench_api")
class ApiWrapper(DLLMBase):
    """lm-eval wrapper for API-based models.

    Uses model/registry.py dispatch to load the correct API model
    (AnthropicModel, MercuryModel, etc.) by name.
    """

    def __init__(
        self,
        model_path: str,
        **kwargs,
    ) -> None:
        super().__init__(model_path=model_path, output_history=False, **kwargs)

    def _create_inner_model(self) -> BaseModel:
        return load_model(model_name=self.model_path)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        return {
            "max_tokens": int(gen_kwargs.get("max_tokens", 128)),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
        }

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        store = MetadataStore.instance()

        for request in requests:
            context, gen_kwargs = request.args
            messages = self._context_to_messages(context)
            gen_config = self._build_generation_config(gen_kwargs)

            dllm_output: DLLMOutput = self._inner_model.generate(
                messages=messages,
                gen_config=gen_config,
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
