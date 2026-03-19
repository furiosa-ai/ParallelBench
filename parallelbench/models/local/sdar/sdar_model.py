from dataclasses import dataclass, field
from typing import Optional

import sys
import types

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

from .constants import SDAR_MASK_TOKEN_ID, SDAR_VALID_METHODS
from parallelbench.models.local.block_diffusion_utils import block_diffusion_generate


@dataclass
class SdarGenerationConfig(DllmGenerationConfig):
    unmasking: str = "confidence_threshold"
    block_length: int = 128
    alg_threshold: Optional[float] = None

    top_p: Optional[float] = None
    top_k: Optional[float] = None

    valid_methods: set = field(default_factory=lambda: set(SDAR_VALID_METHODS))

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
    @staticmethod
    def _patch_missing_hf_file(model_name: str):
        """Create dummy fused_linear_diffusion_cross_entropy.py in HF cache.

        SDAR's HF modeling_sdar.py imports this module at top-level, but it's
        missing from the repo. Transformers tries to download it before Python
        import, so we must create a stub file in the cache directory.
        """
        try:
            from huggingface_hub import try_to_load_from_cache, snapshot_download
            import os

            # Ensure model files are cached
            cache_dir = snapshot_download(model_name, local_files_only=False)
            stub_path = os.path.join(cache_dir, "fused_linear_diffusion_cross_entropy.py")
            if not os.path.exists(stub_path):
                with open(stub_path, "w") as f:
                    f.write(
                        "import torch\n"
                        "import torch.nn as nn\n\n"
                        "class FusedLinearDiffusionCrossEntropyLoss(nn.Module):\n"
                        "    '''Dummy stub for inference — training-only loss function.'''\n"
                        "    pass\n"
                    )
        except Exception:
            pass  # Best effort — will fail at from_pretrained if file is truly needed

    def __init__(self, model_name):
        self._patch_missing_hf_file(model_name)

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

        gen_config = SdarGenerationConfig(**gen_config)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        block_length = gen_config.block_length
        assert block_length == 128 or self.model.name_or_path.endswith(
            f"-b{block_length}"
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
