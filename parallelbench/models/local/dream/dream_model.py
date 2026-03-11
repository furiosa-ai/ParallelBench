import functools
import types
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModel

from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.generation_config import DllmGenerationConfig
from parallelbench.models.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)
from parallelbench.models.registry import ModelRegistry

from .constants import DIFFUCODER_EPS, DREAM_MASK_TOKEN_ID, DREAM_VALID_STRATEGIES
from .dream_model_utils import sample_block


@dataclass
class DreamGenerationConfig(DllmGenerationConfig):
    unmasking: str = (
        "origin"  # Set the default unmasking strategy to "origin" for Dream models
    )
    block_length: int = 128  # Set the default block length for Dream models

    top_p: Optional[float] = None
    top_k: Optional[float] = None

    valid_strategies: set = field(default_factory=lambda: set(DREAM_VALID_STRATEGIES))

    def __post_init__(self):
        super().__post_init__()

        assert self.steps is None or self.steps <= self.max_tokens, (
            f"Steps must be less than or equal to max tokens. Got steps={self.steps}, max_tokens={self.max_tokens}"
        )

        if self.temperature is None or self.temperature == 0.0:
            self.top_p = None
            self.top_k = None

    def to_generation_kwargs(self):
        gen_kwargs = super().to_generation_kwargs()
        gen_length = gen_kwargs.pop("gen_length")
        unmasking = gen_kwargs.pop("unmasking", None)
        return {
            **gen_kwargs,
            "alg": unmasking,
            "max_new_tokens": gen_length,
            "return_dict_in_generate": True,
            "attention_mask": None,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


@ModelRegistry.register(
    lambda name: (
        name
        in (
            "Dream-org/Dream-v0-Instruct-7B",
            "Dream-org/Dream-Coder-v0-Instruct-7B",
            "apple/DiffuCoder-7B-Instruct",
            "apple/DiffuCoder-7B-cpGRPO",
        )
        or "dream" in name.lower()
    )
)
class DreamModel(LocalModel):
    def __init__(self, model_name, accel_framework=None, eps=0):
        super().__init__(
            model_name, model_class=AutoModel, accel_framework=accel_framework
        )

        self.eps = eps
        self.mask_id = DREAM_MASK_TOKEN_ID

    def patch_model(self, gen_config):
        # reset the model methods to the original ones
        self.model.diffusion_generate = types.MethodType(
            self.model.__class__.diffusion_generate, self.model
        )
        self.model._sample = types.MethodType(self.model.__class__._sample, self.model)
        self.model.forward = types.MethodType(self.model.__class__.forward, self.model)

        gen_kwargs = gen_config.to_generation_kwargs()

        if self.accel_framework == "fast_dllm":
            raise NotImplementedError(
                "Fast-dLLM Dream model patching is not implemented yet."
            )

        if (
            gen_kwargs.get("block_length") is not None
            or gen_kwargs.get("threshold") is not None
        ):
            # if block length is specified, we need to patch the model to use the block length
            self.model._sample = types.MethodType(
                functools.partial(
                    sample_block,
                    block_length=gen_kwargs["block_length"],
                    threshold=gen_kwargs.get("threshold"),
                    factor=gen_kwargs.get("factor"),
                ),
                self.model,
            )

        self.model.nfe = 0

        def forward_hook(self, *args, **kwargs):
            self.nfe += 1
            model_output = self.__class__.forward(self, *args, **kwargs)
            return model_output

        self.model.forward = types.MethodType(forward_hook, self.model)

    @property
    def _is_diffucoder(self):
        return self.model.name_or_path.lower() in (
            "apple/diffucoder-7b-instruct",
            "apple/diffucoder-7b-cpgrpo",
        )

    def _generate(self, input_ids, gen_config, output_history):
        self.patch_model(gen_config)

        gen_kwargs = dict(
            **gen_config.to_generation_kwargs(),
            output_history=output_history,
        )

        if self.eps is not None:
            gen_kwargs["eps"] = self.eps
        elif self._is_diffucoder:
            gen_kwargs["eps"] = DIFFUCODER_EPS

        return self.model.diffusion_generate(input_ids, **gen_kwargs), self.model.nfe

    def generate(
        self, messages, output_prefix=None, gen_config=None, output_history=False
    ):
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = messages

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        gen_config = DreamGenerationConfig(
            accel_framework=self.accel_framework, **gen_config
        )

        model_output, nfe = self._generate(
            input_ids, gen_config, output_history=output_history
        )
        output_ids = model_output.sequences[:, input_ids.shape[1] :]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        if output_history:
            history = [h[:, input_ids.shape[1] :] for h in model_output.history]
            decoding_order, decoding_order_corrs = (
                compute_decoding_order_correlation_from_history(self.tokenizer, history)
            )

            if output_history != "pt":
                history = decode_history(self.tokenizer, history)
        else:
            decoding_order, decoding_order_corrs = None, None
            history = None

        return DLLMOutput(
            output=output,
            input_ids=input_ids,
            output_ids=output_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,
            history=history,
            decoding_order=decoding_order,
            decoding_order_corrs=decoding_order_corrs,
        )
