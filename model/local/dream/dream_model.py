import functools
import types
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from model.base_model import BaseModel, DLLMOutput
from model.generation_config import BaseGenerationConfig
from model.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)
from model.registry import ModelRegistry
from utils.perf_utils import measure_time_mem

from .dream_model_utils import sample_block


class DreamRemaskingStrategy(str, Enum):
    RANDOM = "random"
    ORIGIN = "origin"
    MASKGIT_PLUS = "maskgit_plus"
    TOPK_MARGIN = "topk_margin"
    ENTROPY = "entropy"

    ORIGIN_THRESHOLD = "origin_threshold"
    MASKGIT_PLUS_THRESHOLD = "maskgit_plus_threshold"
    TOPK_MARGIN_THRESHOLD = "topk_margin_threshold"
    ENTROPY_THRESHOLD = "entropy_threshold"

    ORIGIN_FACTOR = "origin_factor"
    MASKGIT_PLUS_FACTOR = "maskgit_plus_factor"
    TOPK_MARGIN_FACTOR = "topk_margin_factor"
    ENTROPY_FACTOR = "entropy_factor"

    CONFIDENCE_THRESHOLD = "confidence_threshold"


@dataclass
class DreamGenerationConfig(BaseGenerationConfig):
    accel_framework: Optional[str] = None

    remasking: str = "origin"
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    remasking_temperature: float = 0.0

    remdm_number: Optional[int] = None

    def __post_init__(self):
        assert self.remdm_number is None, "remdm_number is only supported for LLaDA"
        assert self.steps is None or self.steps <= self.max_tokens, (
            f"Steps must be less than or equal to max tokens. Got steps={self.steps}, max_tokens={self.max_tokens}"
        )

        if self.temperature is None or self.temperature == 0.0:
            self.top_p = None
            self.top_k = None

        if self.accel_framework is None:
            assert not self.use_fast_dllm_cache, (
                "use_fast_dllm_cache is only supported in fast_dllm framework"
            )
            # assert self.block_length is None
        elif self.accel_framework == "fast_dllm":
            pass
            # assert False
            # assert self.remasking in [
            #     DreamRemaskingStrategy.ENTROPY,
            #     DreamRemaskingStrategy.CONFIDENCE_THRESHOLD,
            # ], f"Remasking must be one of {list(DreamRemaskingStrategy)}, got {self.remasking}"

            if not self.use_fast_dllm_cache:
                assert self.block_length is None, (
                    "block_length is only supported when use_fast_dllm_cache is True"
                )
                assert not self.use_fast_dllm_dual_cache, (
                    "use_fast_dllm_dual_cache is only supported when use_fast_dllm_cache is True"
                )

    def to_generate_kwargs(self):
        gen_kwargs = dict(
            attention_mask=None,
            max_new_tokens=self.max_tokens,
            return_dict_in_generate=True,
            steps=self.steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.remasking,
            alg_temp=self.remasking_temperature,
            block_length=self.block_length,
        )

        if self.accel_framework == "fast_dllm":
            gen_kwargs["dual_cache"] = self.use_fast_dllm_dual_cache

        if self.remasking in [
            DreamRemaskingStrategy.RANDOM,
            DreamRemaskingStrategy.ORIGIN,
            DreamRemaskingStrategy.MASKGIT_PLUS,
            DreamRemaskingStrategy.TOPK_MARGIN,
            DreamRemaskingStrategy.ENTROPY,
        ]:
            gen_kwargs["threshold"] = None
            gen_kwargs["factor"] = None
        elif self.remasking in [
            DreamRemaskingStrategy.ORIGIN_THRESHOLD,
            DreamRemaskingStrategy.MASKGIT_PLUS_THRESHOLD,
            DreamRemaskingStrategy.TOPK_MARGIN_THRESHOLD,
            DreamRemaskingStrategy.ENTROPY_THRESHOLD,
            DreamRemaskingStrategy.CONFIDENCE_THRESHOLD,
        ]:
            assert self.alg_threshold is not None, (
                f"alg_threshold must be provided for {self.remasking} algorithm"
            )
            gen_kwargs["threshold"] = self.alg_threshold
            gen_kwargs["factor"] = None

            gen_kwargs["alg"] = {
                DreamRemaskingStrategy.ORIGIN_THRESHOLD: DreamRemaskingStrategy.ORIGIN,
                DreamRemaskingStrategy.MASKGIT_PLUS_THRESHOLD: DreamRemaskingStrategy.MASKGIT_PLUS,
                DreamRemaskingStrategy.CONFIDENCE_THRESHOLD: DreamRemaskingStrategy.MASKGIT_PLUS,
                DreamRemaskingStrategy.TOPK_MARGIN_THRESHOLD: DreamRemaskingStrategy.TOPK_MARGIN,
                DreamRemaskingStrategy.ENTROPY_THRESHOLD: DreamRemaskingStrategy.ENTROPY,
            }[self.remasking]
        elif self.remasking in [
            DreamRemaskingStrategy.ORIGIN_FACTOR,
            DreamRemaskingStrategy.MASKGIT_PLUS_FACTOR,
            DreamRemaskingStrategy.TOPK_MARGIN_FACTOR,
            DreamRemaskingStrategy.ENTROPY_FACTOR,
        ]:
            assert self.alg_factor is not None, (
                f"alg_factor must be provided for {self.remasking} algorithm"
            )
            gen_kwargs["threshold"] = None
            gen_kwargs["factor"] = self.alg_factor

            gen_kwargs["alg"] = {
                DreamRemaskingStrategy.ORIGIN_FACTOR: DreamRemaskingStrategy.ORIGIN,
                DreamRemaskingStrategy.MASKGIT_PLUS_FACTOR: DreamRemaskingStrategy.MASKGIT_PLUS,
                DreamRemaskingStrategy.TOPK_MARGIN_FACTOR: DreamRemaskingStrategy.TOPK_MARGIN,
                DreamRemaskingStrategy.ENTROPY_FACTOR: DreamRemaskingStrategy.ENTROPY,
            }[self.remasking]
        else:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")

        return gen_kwargs



@ModelRegistry.register(
    lambda name: name
    in (
        "Dream-org/Dream-v0-Instruct-7B",
        "Dream-org/Dream-Coder-v0-Instruct-7B",
        "apple/DiffuCoder-7B-Instruct",
        "apple/DiffuCoder-7B-cpGRPO",

    )
)
class DreamModel(BaseModel):
    def __init__(self, model_name, accel_framework=None, eps=0):
        if accel_framework == "fast_dllm":
            raise NotImplementedError("Fast-dLLM Dream model loading is not implemented yet.")
        
        self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.eps = eps
        self.mask_id = 151666
        self.accel_framework = accel_framework

    def patch_model(self, gen_config):
        # reset the model methods to the original ones
        self.model.diffusion_generate = types.MethodType(
            self.model.__class__.diffusion_generate, self.model
        )
        self.model._sample = types.MethodType(self.model.__class__._sample, self.model)
        self.model.forward = types.MethodType(self.model.__class__.forward, self.model)

        gen_kwargs = gen_config.to_generate_kwargs()

        if self.accel_framework == "fast_dllm":
            raise NotImplementedError("Fast-dLLM Dream model patching is not implemented yet.")
        
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

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @property
    def _is_diffucoder(self):
        return self.model.name_or_path.lower() in (
            "apple/diffucoder-7b-instruct",
            "apple/diffucoder-7b-cpgrpo",
        )

    @measure_time_mem("generate")
    def model_generate(self, input_ids, gen_config, output_history):
        self.patch_model(gen_config)

        gen_kwargs = dict(
            **gen_config.to_generate_kwargs(),
            output_history=output_history,
        )

        if self._is_diffucoder:
            gen_kwargs["eps"] = 1e-12

        if self.eps is not None:
            gen_kwargs["eps"] = self.eps

        return self.model.diffusion_generate(input_ids, **gen_kwargs), self.model.nfe

    def generate(self, messages, gen_config=None, output_history=None):
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

        model_output, nfe = self.model_generate(
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
