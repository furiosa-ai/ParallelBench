from dataclasses import dataclass
from enum import Enum
import sys
import types
from typing import Optional

import torch

from transformers import AutoModel, AutoTokenizer
from model.base_model import BaseModel, DLLMOutput
from utils.perf_utils import measure_time_mem
from utils.utils import insert_import_path


FAST_DLLM_PATH = "src/Fast_dLLM/dream"


class DiffuCoderRemaskingStrategy(str, Enum):
    ORIGIN = 'origin'
    MASKGIT_PLUS = 'maskgit_plus'
    TOPK_MARGIN = 'topk_margin'
    ENTROPY = 'entropy'

    # fast_dllm specific
    CONFIDENCE_THRESHOLD = 'confidence_threshold'


@dataclass
class DiffuCoderGenerationConfig:
    accel_framework: Optional[str] = None

    max_tokens: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "origin"
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    remasking_temperature: float = 0.0

    # fast_dllm specific
    fast_dllm_use_cache: bool = False
    fast_dllm_threshold: float = 0.9
    fast_dllm_block_length: Optional[int] = None
    fast_dllm_dual_cache: bool = False

    def __post_init__(self):
        assert self.steps <= self.max_tokens, f"Steps must be less than or equal to max tokens. Got steps={self.steps}, max_tokens={self.max_tokens}"
        
        if self.accel_framework is None:
            assert self.remasking in [
                DiffuCoderRemaskingStrategy.ORIGIN,
                DiffuCoderRemaskingStrategy.MASKGIT_PLUS,
                DiffuCoderRemaskingStrategy.TOPK_MARGIN,
                DiffuCoderRemaskingStrategy.ENTROPY,
            ]

            assert not self.fast_dllm_use_cache, "fast_dllm_use_cache is only supported in fast_dllm framework"
            assert self.fast_dllm_block_length is None
        elif self.accel_framework == "fast_dllm":
            assert self.remasking in [
                DiffuCoderRemaskingStrategy.ENTROPY,
                DiffuCoderRemaskingStrategy.CONFIDENCE_THRESHOLD,
            ], f"Remasking must be one of {list(DiffuCoderRemaskingStrategy)}, got {self.remasking}"

            if not self.fast_dllm_use_cache:
                assert self.fast_dllm_block_length is None, "fast_dllm_block_length is only supported when fast_dllm_use_cache is True"
                assert not self.fast_dllm_dual_cache, "fast_dllm_dual_cache is only supported when fast_dllm_use_cache is True"

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
        )

        if self.accel_framework == "fast_dllm":
            gen_kwargs["threshold"] = self.fast_dllm_threshold
            gen_kwargs["block_length"] = self.fast_dllm_block_length
            gen_kwargs["dual_cache"] = self.fast_dllm_dual_cache

        return gen_kwargs


class DiffuCoderModel(BaseModel):
    def __init__(self, model_name, accel_framework=None):
        if accel_framework == "fast_dllm":
            with insert_import_path(FAST_DLLM_PATH):
                from model.modeling_dream import DiffuCoderModel as FastDllmDiffuCoderModel
            self.model = FastDllmDiffuCoderModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        elif accel_framework is None:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unsupported acceleration framework: {accel_framework}")

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.accel_framework = accel_framework

    def patch_model(self, gen_config):
        if self.accel_framework == "fast_dllm":
            with insert_import_path(FAST_DLLM_PATH):
                from model.generation_utils import DiffuCoderGenerationMixin as DiffuCoderGenerationMixinWithoutCache
                from model.generation_utils_block import DiffuCoderGenerationMixin as DiffuCoderGenerationMixinWithCache

            mixin_class = DiffuCoderGenerationMixinWithCache if gen_config.fast_dllm_use_cache else DiffuCoderGenerationMixinWithoutCache

            self.model.diffusion_generate = types.MethodType(mixin_class.diffusion_generate, self.model)
            self.model._sample = types.MethodType(mixin_class._sample, self.model)
        else:
            pass
            # self.model.diffusion_generate = types.MethodType(self.model.__class__.diffusion_generate, self.model)
            # self.model._sample = types.MethodType(self.model.__class__._sample, self.model)

        self.model.nfe = 0
        def forward_hook(self, *args, **kwargs):
            self.nfe += 1
            return self.__class__.forward(self, *args, **kwargs)

        self.model.forward = types.MethodType(forward_hook, self.model)

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def model_generate(self, input_ids, gen_config, output_history):
        self.patch_model(gen_config)

        return self.model.diffusion_generate(input_ids, **gen_config.to_generate_kwargs(), output_history=output_history), self.model.nfe

    def generate(self, messages, gen_config=None, output_history=False):
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        gen_config = DiffuCoderGenerationConfig(accel_framework=self.accel_framework, **gen_config)

        model_output, nfe = self.model_generate(input_ids, gen_config, output_history=output_history)
        output_ids = model_output.sequences[:, input_ids.shape[1]:]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        return DLLMOutput(
            output=output,
            input_ids=input_ids,
            output_ids=output_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,
            history=model_output.history
        )
