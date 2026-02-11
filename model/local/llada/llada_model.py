from dataclasses import dataclass
from enum import Enum
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import torch

from dataset.parallel_bench.data.task import PARALLEL_BENCH_MASK_TOKEN
from model.base_model import BaseModel, DLLMOutput
from model.model_utils import (
    decode_history,
    compute_decoding_order_correlation_from_history,
)
from model.registry import ModelRegistry
from model.local.generate import generate
from utils.perf_utils import measure_time_mem


class LladaRemaskingStrategy(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    RANDOM = "random"
    TOPK_MARGIN = "topk_margin"
    ENTROPY = "entropy"

    # fast-dllm specific
    LOW_CONFIDENCE_THRESHOLD = "low_confidence_threshold"
    RANDOM_THRESHOLD = "random_threshold"
    TOPK_MARGIN_THRESHOLD = "topk_margin_threshold"
    ENTROPY_THRESHOLD = "entropy_threshold"

    LOW_CONFIDENCE_FACTOR = "low_confidence_factor"
    RANDOM_FACTOR = "random_factor"
    TOPK_MARGIN_FACTOR = "topk_margin_factor"
    ENTROPY_FACTOR = "entropy_factor"


@dataclass
class LladaGenerationConfig:
    accel_framework: Optional[str] = None

    max_tokens: int = 128
    steps: Optional[int] = 128
    temperature: float = 0.0
    alg_temp: float = 0.0
    remasking: str = "low_confidence"
    block_length: int = 128

    # fast-dllm specific
    fast_dllm_threshold: float = (
        0.9  # TODO assert none if not using LOW_CONFIDENCE_THRESHOLD
    )
    fast_dllm_factor: Optional[float] = None
    fast_dllm_use_cache: bool = False
    fast_dllm_dual_cache: bool = False

    remdm_steps: Optional[int] = None
    remdm_number: Optional[int] = None

    @property
    def num_blocks(self):
        return self.max_tokens // self.block_length

    def __post_init__(self):
        assert self.steps is None or self.steps <= self.max_tokens, (
            f"Steps must be less than or equal to max tokens. Got steps={self.steps}, max_tokens={self.max_tokens}"
        )
        assert self.max_tokens % self.block_length == 0, (
            f"Max tokens must be divisible by block length. Got max_tokens={self.max_tokens}, block_length={self.block_length}"
        )
        assert self.steps is None or (self.steps % self.num_blocks == 0), (
            f"Steps must be divisible by number of blocks. Got steps={self.steps}, num_blocks={self.num_blocks}"
        )
        assert self.remasking in list(LladaRemaskingStrategy), (
            f"Remasking must be one of {list(LladaRemaskingStrategy)}, got {self.remasking}"
        )
        assert not (self.accel_framework != "fast_dllm" and self.fast_dllm_use_cache)

    def to_generate_kwargs(self):
        gen_kwargs = dict(
            steps=self.steps,
            gen_length=self.max_tokens,
            block_length=self.block_length,
            temperature=self.temperature,
            alg_temp=self.alg_temp,
            remasking=self.remasking,
        )

        if self.remasking in [
            LladaRemaskingStrategy.LOW_CONFIDENCE,
            LladaRemaskingStrategy.RANDOM,
            LladaRemaskingStrategy.TOPK_MARGIN,
            LladaRemaskingStrategy.ENTROPY,
        ]:
            gen_kwargs["threshold"] = None
            gen_kwargs["factor"] = None
        elif self.is_mdpo_rcr():
            gen_kwargs["overtime_conf"] = True
            gen_kwargs["remasking"] = LladaRemaskingStrategy.LOW_CONFIDENCE
        elif self.is_remdm():
            gen_kwargs["remdm_number"] = self.remdm_number
            gen_kwargs["remdm_steps"] = self.remdm_steps
            gen_kwargs["remasking"] = LladaRemaskingStrategy.LOW_CONFIDENCE
        elif self.remasking in [
            LladaRemaskingStrategy.LOW_CONFIDENCE_THRESHOLD,
            LladaRemaskingStrategy.RANDOM_THRESHOLD,
            LladaRemaskingStrategy.TOPK_MARGIN_THRESHOLD,
            LladaRemaskingStrategy.ENTROPY_THRESHOLD,
        ]:
            assert self.fast_dllm_threshold is not None, (
                f"fast_dllm_threshold must be provided for {self.remasking} algorithm"
            )
            gen_kwargs["threshold"] = self.fast_dllm_threshold
            gen_kwargs["factor"] = None

            gen_kwargs["remasking"] = {
                LladaRemaskingStrategy.LOW_CONFIDENCE_THRESHOLD: LladaRemaskingStrategy.LOW_CONFIDENCE,
                LladaRemaskingStrategy.RANDOM_THRESHOLD: LladaRemaskingStrategy.RANDOM,
                LladaRemaskingStrategy.TOPK_MARGIN_THRESHOLD: LladaRemaskingStrategy.TOPK_MARGIN,
                LladaRemaskingStrategy.ENTROPY_THRESHOLD: LladaRemaskingStrategy.ENTROPY,
            }[self.remasking]
        elif self.remasking in [
            LladaRemaskingStrategy.LOW_CONFIDENCE_FACTOR,
            LladaRemaskingStrategy.RANDOM_FACTOR,
            LladaRemaskingStrategy.TOPK_MARGIN_FACTOR,
            LladaRemaskingStrategy.ENTROPY_FACTOR,
        ]:
            assert self.fast_dllm_factor is not None, (
                f"fast_dllm_factor must be provided for {self.remasking} algorithm"
            )
            gen_kwargs["threshold"] = None
            gen_kwargs["factor"] = self.fast_dllm_factor

            gen_kwargs["remasking"] = {
                LladaRemaskingStrategy.LOW_CONFIDENCE_FACTOR: LladaRemaskingStrategy.LOW_CONFIDENCE,
                LladaRemaskingStrategy.RANDOM_FACTOR: LladaRemaskingStrategy.RANDOM,
                LladaRemaskingStrategy.TOPK_MARGIN_FACTOR: LladaRemaskingStrategy.TOPK_MARGIN,
                LladaRemaskingStrategy.ENTROPY_FACTOR: LladaRemaskingStrategy.ENTROPY,
            }[self.remasking]
        else:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")

        return gen_kwargs


LLADA_MASK_TOKEN_ID = 126336


@ModelRegistry.register(
    lambda name: name
    in ("GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-1.5")
    or "llada" in name.lower()
)
class LladaModel(BaseModel):
    def __init__(self, model_name, accel_framework=None):
        if accel_framework == "fast_dllm":
            raise NotImplementedError("Fast dLLM LLADA model loading is not implemented yet.")
        else:
            model_class = AutoModel
            
        self.model = model_class.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.patch_model_forward(self.model, LLADA_MASK_TOKEN_ID)

        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False, dynamic=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.mask_token_id = LLADA_MASK_TOKEN_ID

        self.mask_id = LLADA_MASK_TOKEN_ID
        self.accel_framework = accel_framework

    def patch_model_forward(self, model, mask_token_id):
        fwd_fn = model.__class__.forward

        def wrapped_forward(*args, **kwargs):
            output = fwd_fn(*args, **kwargs)
            output.logits[:, :, mask_token_id] = -float(
                "Inf"
            )  # cannot sample mask token
            return output

        model.__class__.forward = wrapped_forward

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def _generate(self, input_ids, gen_config, output_history=False, output0_ids=None):
        if self.accel_framework == "fast_dllm":
            # FIXME: implement fast dLLM generation
            raise NotImplementedError("Fast dLLM LLADA generation is not implemented yet.")

        gen_kwargs = gen_config.to_generate_kwargs()

        return generate(
            self.model,
            input_ids,
            mask_id=self.mask_id,
            **gen_kwargs,
            output_history=output_history,
            output0_ids=output0_ids,
        )

    def generate(
        self, messages, output_prefix=None, gen_config=None, output_history=False
    )-> DLLMOutput:
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = messages

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        gen_config = LladaGenerationConfig(
            accel_framework=self.accel_framework, **gen_config
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
