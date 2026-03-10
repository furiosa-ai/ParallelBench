from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallelbench.datasets.task import PARALLEL_BENCH_MASK_TOKEN
from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.generation_config import DllmGenerationConfig
from parallelbench.models.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)
from parallelbench.models.registry import ModelRegistry

from .constants import TRADO_MASK_TOKEN_ID, TRADO_VALID_STRATEGIES
from .trado_model_utils import block_diffusion_generate


@dataclass
class TradoGenerationConfig(DllmGenerationConfig):
    remasking: str = "confidence_threshold"
    block_length: int = 128
    alg_threshold: Optional[float] = 0.85

    top_p: Optional[float] = None
    top_k: Optional[float] = None

    valid_strategies: set = field(default_factory=lambda: set(TRADO_VALID_STRATEGIES))

    def to_generation_kwargs(self):
        gen_kwargs = {
            "gen_length": self.max_tokens,
            "block_length": self.block_length,
            "steps": self.steps,
            "temperature": self.temperature,
            "top_k": self.top_k if self.top_k is not None else 0.0,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "threshold": self.alg_threshold if self.alg_threshold is not None else 0.85,
            "remasking": self.remasking,
        }

        return gen_kwargs


@ModelRegistry.register(
    lambda name: (
        name
        in (
            "Gen-Verse/TraDo-4B-Instruct",
            "Gen-Verse/TraDo-8B-Instruct",
            "Gen-Verse/TraDo-8B-Thinking",
        )
        or "trado" in name.lower()
        or "sdar" in name.lower()
    )
)
class TradoModel(LocalModel):
    def __init__(self, model_name, accel_framework=None):
        # Use AutoModelForCausalLM to load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.mask_token_id = TRADO_MASK_TOKEN_ID

        self.mask_id = TRADO_MASK_TOKEN_ID
        self._validate_and_set_framework(accel_framework)

    def _generate(self, input_ids, gen_config, output_history=False, output0_ids=None):
        gen_kwargs = gen_config.to_generation_kwargs()

        if self.accel_framework == "fast_dllm":
            raise NotImplementedError("fast_dllm is not yet supported for TraDo model.")

        generate_fn = block_diffusion_generate

        if output0_ids is not None:
            return generate_fn(
                self.model,
                input_ids,
                mask_id=self.mask_id,
                **gen_kwargs,
                output_history=output_history,
                output0_ids=output0_ids,
            )
        else:
            return generate_fn(
                self.model,
                input_ids,
                mask_id=self.mask_id,
                **gen_kwargs,
                output_history=output_history,
            )

    def generate(
        self, messages, output_prefix=None, gen_config=None, output_history=False
    ):
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = messages

        gen_config = TradoGenerationConfig(
            accel_framework=self.accel_framework, **gen_config
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        block_length = gen_config.block_length
        assert (
            block_length == 4
            or self.model.name_or_path.endswith(f"-b{block_length}")
            or "TraDo" in self.model.name_or_path
        ), (
            f"Block length {block_length} is not supported by the model {self.model.name_or_path}."
        )
        # pad input_ids to be multiple of block_length
        if input_ids.shape[1] % block_length != 0:
            pad_length = block_length - (input_ids.shape[1] % block_length)
            input_ids = F.pad(
                input_ids, (0, pad_length), value=self.tokenizer.pad_token_id
            )

        if output_prefix is not None:
            output_prefix = output_prefix.replace(
                PARALLEL_BENCH_MASK_TOKEN, self.tokenizer.mask_token
            )
            output0_ids = self.tokenizer(
                output_prefix,
                return_tensors="pt",
                padding="max_length",
                max_length=gen_config.max_tokens,
            ).input_ids.to(self.model.device)
            assert output0_ids.shape[1] == gen_config.max_tokens, (
                "output_prefix is too long"
            )
        else:
            output0_ids = None

        input_output_ids, nfe, history = self._generate(
            input_ids,
            gen_config,
            output_history=output_history,
            output0_ids=output0_ids,
        )
        output_ids = input_output_ids[:, input_ids.shape[1] :]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        assert not (output_history and history is None), (
            "History should not be None if output_history is True."
        )

        decoding_order, decoding_order_corrs = (
            compute_decoding_order_correlation_from_history(self.tokenizer, history)
        )

        return DLLMOutput(
            output=output,
            input_ids=input_ids,
            output_ids=output_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,
            history=decode_history(self.tokenizer, history),
            decoding_order=decoding_order,
            decoding_order_corrs=decoding_order_corrs,
        )
