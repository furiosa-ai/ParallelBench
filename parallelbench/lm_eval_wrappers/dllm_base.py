"""Base class for wrapping ParallelBench dLLM models as lm-eval LM instances.

Subclasses provide model-specific initialization (_create_inner_model) and
generation config creation (_create_generation_config). This base handles the
common lm-eval interface (generate_until, loglikelihood stubs) and metadata
capture via MetadataStore.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
from accelerate import Accelerator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from parallelbench.models.base_model import BaseModel, DLLMOutput
from parallelbench.lm_eval_wrappers.metadata_store import (
    GenerationMetadata,
    MetadataStore,
)


class DLLMBase(LM):
    """Base lm-eval wrapper for ParallelBench dLLM models.

    Subclasses must implement:
        _create_inner_model() -> BaseModel

    Constructor model_args (passed via --model_args):
        model_path: str          - HuggingFace model name/path
        accel_framework: str     - None, "fast_dllm", etc.
        output_history: bool     - Whether to track generation history
        infill: bool             - Whether to use infill mode

    Generation parameters (passed via --gen_kwargs or task YAML generation_kwargs):
        steps: int               - Diffusion steps (default: 128)
        max_tokens: int          - Maximum generation length (default: 128)
        block_length: int        - Block length for semi-AR generation (default: 128)
        remasking: str           - Remasking strategy (default: "random")
        temperature: float       - Sampling temperature (default: 0.0)
        alg_temp: float          - Algorithm temperature (default: 0.0)
        alg_threshold: float     - Confidence threshold
        alg_factor: float        - Dynamic remasking factor
    """

    def __init__(
        self,
        model_path: str,
        accel_framework: Optional[str] = None,
        output_history: bool = True,
        infill: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        self._device = torch.device(str(self.accelerator.device))

        self.model_path = model_path
        self.accel_framework = accel_framework
        self._output_history = output_history
        self._infill = infill
        self._batch_size = batch_size
        self._extra_kwargs = kwargs

        self._inner_model: BaseModel = self._create_inner_model()

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def _create_inner_model(self) -> BaseModel:
        """Instantiate and return the inner ParallelBench model."""
        ...

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        """Build generation config dict from gen_kwargs.

        Reads generation parameters from gen_kwargs (task YAML generation_kwargs
        merged with --gen_kwargs CLI). Subclasses can override to change defaults
        or add model-specific parameters.
        """
        config = {
            "steps": int(gen_kwargs.get("steps", 128)),
            "max_tokens": int(gen_kwargs.get("max_tokens", 128)),
            "block_length": int(gen_kwargs.get("block_length", 128)),
            "remasking": gen_kwargs.get("remasking", "random"),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
            "alg_temp": float(gen_kwargs.get("alg_temp", 0.0)),
        }
        if "alg_threshold" in gen_kwargs:
            config["alg_threshold"] = float(gen_kwargs["alg_threshold"])
        if "alg_factor" in gen_kwargs:
            config["alg_factor"] = float(gen_kwargs["alg_factor"])
        return config

    # ─── lm-eval LM interface ────────────────────────────────────────────

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        store = MetadataStore.instance()

        for request in requests:
            context, gen_kwargs = request.args

            messages = self._context_to_messages(context)
            gen_config = self._build_generation_config(gen_kwargs)

            output_prefix = None
            if self._infill and hasattr(request, "doc") and request.doc:
                output_prefix_str = request.doc.get("output_prefix")
                if output_prefix_str is not None:
                    output_prefix = self._encode_output_prefix(output_prefix_str)

            dllm_output: DLLMOutput = self._inner_model.generate(
                messages=messages,
                gen_config=gen_config,
                output_prefix=output_prefix,
                output_history=self._output_history,
            )

            store.append(
                GenerationMetadata(
                    nfe=dllm_output.nfe,
                    history=dllm_output.history,
                    decoding_order=dllm_output.decoding_order,
                    decoding_order_corrs=dllm_output.decoding_order_corrs,
                    input_length=dllm_output.input_length,
                    output_length=dllm_output.output_length,
                )
            )

            generated_text = dllm_output.output
            generated_text = self._apply_until_truncation(generated_text, gen_kwargs)
            results.append(generated_text)

        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "ParallelBench dLLM models do not support loglikelihood. "
            "Use generate_until tasks only."
        )

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError(
            "ParallelBench dLLM models do not support loglikelihood_rolling. "
            "Use generate_until tasks only."
        )

    # ─── Helper methods ──────────────────────────────────────────────────

    def _context_to_messages(self, context: str | list) -> list[dict] | str:
        """Convert lm-eval context to the messages format expected by inner model.

        lm-eval may pass either a plain string or a list of chat messages.
        """
        if isinstance(context, list):
            return context
        return [{"role": "user", "content": context}]

    def _encode_output_prefix(self, prefix_str: str) -> str:
        """Return the output prefix string for infill mode.

        Models handle their own tokenization of the prefix.
        """
        return prefix_str

    @staticmethod
    def _apply_until_truncation(text: str, gen_kwargs: dict) -> str:
        """Truncate generated text at the first occurrence of any until-sequence."""
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]

        min_idx = len(text)
        for stop_seq in until:
            idx = text.find(stop_seq)
            if idx != -1:
                min_idx = min(min_idx, idx)
        return text[:min_idx]
