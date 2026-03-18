"""Base class for wrapping ParallelBench dLLM models as lm-eval LM instances.

Subclasses provide model-specific initialization (_create_inner_model) and
generation config creation (_create_generation_config). This base handles the
common lm-eval interface (generate_until, loglikelihood stubs) and metadata
capture via MetadataStore.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from accelerate import Accelerator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.utils import Collator

from parallelbench.models.base_model import BaseModel, DLLMOutput
from parallelbench.lm_eval_wrappers.metadata_store import (
    GenerationMetadata,
    MetadataStore,
)
from parallelbench.models.unmasking_registry import get_method_info


class DLLMBase(LM):
    """Base lm-eval wrapper for ParallelBench dLLM models.

    Subclasses must implement:
        _create_inner_model() -> BaseModel

    Constructor model_args (passed via --model_args):
        model_path: str          - HuggingFace model name/path
        output_history: bool     - Whether to track generation history
        infill: bool             - Whether to use infill mode

    Generation parameters (passed via --gen_kwargs or task YAML generation_kwargs):
        steps: int               - Diffusion steps (default: 128)
        max_tokens: int          - Maximum generation length (default: 128)
        block_length: int        - Block length for semi-AR generation (default: 128)
        unmasking: str           - Unmasking method (default: "random")
        temperature: float       - Sampling temperature (default: 0.0)
        alg_temp: float          - Algorithm temperature (default: 0.0)
        alg_threshold: float     - Confidence threshold
        alg_factor: float        - Dynamic unmasking factor
    """

    def __init__(
        self,
        model_path: str,
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
        self._output_history = output_history
        self._infill = infill
        self._batch_size = int(batch_size)
        self._extra_kwargs = kwargs

        self._inner_model: BaseModel = self._create_inner_model()
        self._apply_chat_template_active = False

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    @abstractmethod
    def _create_inner_model(self) -> BaseModel:
        """Instantiate and return the inner ParallelBench model."""
        ...

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        """Build generation config dict from gen_kwargs.

        Reads generation parameters from gen_kwargs (task YAML generation_kwargs
        merged with --gen_kwargs CLI). Subclasses can override to change defaults
        or add model-specific parameters.

        If "k" (tokens per step) is present in gen_kwargs and neither "steps"
        nor "block_length" are explicitly provided, derives steps and block_length
        from the unmasking registry's derive_fn for the given unmasking method.
        Explicit "steps"/"block_length" values always take priority.
        """
        max_tokens = int(gen_kwargs.get("max_tokens", 128))
        unmasking = gen_kwargs.get("unmasking", "random")

        steps_explicit = "steps" in gen_kwargs
        block_length_explicit = "block_length" in gen_kwargs

        if "k" in gen_kwargs and not steps_explicit and not block_length_explicit:
            info = get_method_info(unmasking)
            k = float(gen_kwargs["k"])
            derived = info.derive_fn(k, max_tokens)
            steps = derived["steps"]
            block_length = derived["block_length"]
        else:
            steps = int(gen_kwargs.get("steps", 128))
            block_length = int(gen_kwargs.get("block_length", 128))

        config = {
            "steps": steps,
            "max_tokens": max_tokens,
            "block_length": block_length,
            "unmasking": unmasking,
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
        if self._batch_size > 1:
            return self._generate_until_batched(requests)
        return self._generate_until_sequential(requests)

    def _generate_until_sequential(self, requests: list[Instance]) -> list[str]:
        """Process requests one at a time (batch_size=1 path)."""
        results: list[str] = []
        store = MetadataStore.instance()

        for request in requests:
            context, gen_kwargs = request.args
            gen_kwargs = self._bridge_generation_kwargs(gen_kwargs)
            messages = self._context_to_messages(context)
            gen_config = self._build_generation_config(gen_kwargs)
            output_prefix = self._get_output_prefix(request)

            dllm_output: DLLMOutput = self._inner_model.generate(
                messages=messages,
                gen_config=gen_config,
                output_prefix=output_prefix,
                output_history=self._output_history,
            )

            store.append(self._build_metadata(dllm_output, gen_config))
            generated_text = self._apply_until_truncation(
                dllm_output.output, gen_kwargs
            )
            results.append(generated_text)

        return results

    def _generate_until_batched(self, requests: list[Instance]) -> list[str]:
        """Process requests in batches using Collator (batch_size>1 path).

        Groups requests by gen_kwargs, sorts by descending context length,
        and calls generate_batch() on the inner model. Results and metadata
        are reordered back to the original request order.
        """
        # Bridge gen_kwargs for all requests before grouping
        bridged_args = []
        for request in requests:
            context, gen_kwargs = request.args
            gen_kwargs = self._bridge_generation_kwargs(gen_kwargs)
            bridged_args.append((context, gen_kwargs))

        def _collate(item):
            context, _gen_kwargs = item
            return -len(str(context)), str(context)

        collator = Collator(
            bridged_args,
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )

        results: list[str] = []
        metadata_list: list[GenerationMetadata] = []

        for chunk in collator.get_batched(n=self._batch_size):
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            gen_config = self._build_generation_config(gen_kwargs)

            messages_list = [self._context_to_messages(ctx) for ctx in contexts]
            # NOTE: Infill output_prefix is not supported in batched mode.
            # Collator reorders requests, so original request indices are not
            # available here. Infill mode is not used by any current model.
            output_prefix_list = None

            dllm_outputs: list[DLLMOutput] = self._inner_model.generate_batch(
                messages_list=messages_list,
                gen_config=gen_config,
                output_prefix_list=output_prefix_list,
                output_history=self._output_history,
            )

            for dllm_output in dllm_outputs:
                metadata_list.append(self._build_metadata(dllm_output, gen_config))
                generated_text = self._apply_until_truncation(
                    dllm_output.output, gen_kwargs
                )
                results.append(generated_text)

        # Restore original request order
        results = collator.get_original(results)
        metadata_list = collator.get_original(metadata_list)

        # Append metadata in original order
        store = MetadataStore.instance()
        for metadata in metadata_list:
            store.append(metadata)

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

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply chat template to render messages into a prompt string.

        Follows lm-eval's official HF model pattern. Called by lm-eval's
        fewshot_context() when --apply_chat_template is used.
        """
        return self._inner_model.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def chat_template(self, chat_template=False):
        """Set chat template and track activation for context handling."""
        result = super().chat_template(chat_template)
        self._apply_chat_template_active = True
        return result

    # ─── Helper methods ──────────────────────────────────────────────────

    BRIDGED_KEYS = (
        "k",
        "alg_threshold",
        "alg_factor",
        "steps",
        "block_length",
        "unmasking",
    )
    COERCE_FLOAT_KEYS = ("k", "alg_threshold", "alg_factor")

    def _bridge_generation_kwargs(self, gen_kwargs: dict) -> dict:
        """Bridge generation params from model_args (CLI) into gen_kwargs (task YAML)."""
        gen_kwargs = dict(gen_kwargs)
        for key in self.BRIDGED_KEYS:
            if key in self._extra_kwargs and key not in gen_kwargs:
                value = self._extra_kwargs[key]
                if key in self.COERCE_FLOAT_KEYS:
                    value = float(value)
                gen_kwargs[key] = value
        return gen_kwargs

    def _get_output_prefix(self, request: Instance) -> str | None:
        """Extract output prefix from request for infill mode."""
        if self._infill and hasattr(request, "doc") and request.doc:
            output_prefix_str = request.doc.get("output_prefix")
            if output_prefix_str is not None:
                return self._encode_output_prefix(output_prefix_str)
        return None

    @staticmethod
    def _build_metadata(
        dllm_output: DLLMOutput, gen_config: dict
    ) -> GenerationMetadata:
        """Build GenerationMetadata from a DLLMOutput."""
        tokens_per_step = (
            gen_config["max_tokens"] / dllm_output.nfe if dllm_output.nfe > 0 else None
        )
        return GenerationMetadata(
            nfe=dllm_output.nfe,
            tokens_per_step=tokens_per_step,
            history=dllm_output.history,
            decoding_order=dllm_output.decoding_order,
            decoding_order_corrs=dllm_output.decoding_order_corrs,
            input_length=dllm_output.input_length,
            output_length=dllm_output.output_length,
        )

    def _context_to_messages(self, context: str | list) -> list[dict] | str:
        """Convert lm-eval context to the format expected by inner model.

        When apply_chat_template is active, context is already a rendered
        prompt string from lm-eval's fewshot pipeline. Pass it directly
        so the inner model tokenizes without re-applying the template.
        When inactive, wrap as messages for the model to apply its template.
        """
        if isinstance(context, list):
            return context
        if getattr(self, "_apply_chat_template_active", False):
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
