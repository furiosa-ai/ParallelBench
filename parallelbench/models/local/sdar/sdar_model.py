import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from parallelbench.datasets.task import PARALLEL_BENCH_MASK_TOKEN
from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.generation_config import DllmGenerationConfig

from parallelbench.models.registry import ModelRegistry

from .constants import SDAR_MASK_TOKEN_ID, SDAR_VALID_METHODS
from parallelbench.models.local.block_diffusion_utils import block_diffusion_generate


@dataclass
class SdarGenerationConfig(DllmGenerationConfig):
    unmasking: str = "confidence_threshold"
    block_length: int = 4
    alg_threshold: Optional[float] = None

    top_p: Optional[float] = None
    top_k: Optional[float] = None

    valid_methods: set = field(default_factory=lambda: set(SDAR_VALID_METHODS))

    def __post_init__(self):
        # Auto-populate alg_threshold for threshold methods before parent validation
        if self.alg_threshold is None and self.unmasking in self.valid_methods:
            from parallelbench.models.unmasking_registry import get_method_type

            if get_method_type(self.unmasking) == "threshold":
                self.alg_threshold = 0.85
        super().__post_init__()

    def to_generation_kwargs(self):
        gen_kwargs = {
            "gen_length": self.max_tokens,
            "block_length": self.block_length,
            "steps": self.steps,
            "temperature": self.temperature,
            "top_k": self.top_k if self.top_k is not None else 0.0,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "threshold": self.alg_threshold if self.alg_threshold is not None else 0.85,
            "unmasking": self.unmasking,
        }

        return gen_kwargs


@ModelRegistry.register(lambda name: "sdar" in name.lower())
class SdarModel(LocalModel):
    def __init__(self, model_name):
        # Load SDAR using local patched modeling file.
        # The patched version replaces @torch.compile'd fused_flex_attention
        # with a dual-path function that falls back to SDPA for regular tensor
        # masks (block_diffusion_utils passes 4D tensors, not BlockMask objects).
        from .modeling_sdar_patched import SDARForCausalLM
        from .configuration_sdar import SDARConfig

        config = SDARConfig.from_pretrained(model_name)
        local_rank = os.environ.get("LOCAL_RANK")
        device_map = f"cuda:{local_rank}" if local_rank is not None else "cuda"
        self.model = SDARForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.mask_token_id = SDAR_MASK_TOKEN_ID

        self.mask_id = SDAR_MASK_TOKEN_ID

    def _generate(self, input_ids, gen_config, output_history=False, output0_ids=None):
        gen_kwargs = gen_config.to_generation_kwargs()

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

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )

        gen_config = SdarGenerationConfig(**gen_config)

        block_length = gen_config.block_length
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

        return DLLMOutput(
            output=output,
            input_ids=input_ids,
            output_ids=output_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,
        )
