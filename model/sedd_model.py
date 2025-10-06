
from dataclasses import dataclass
from enum import Enum
import functools
import numpy as np
from typing import Optional
from transformers import GPT2TokenizerFast
import torch
import torch.nn.functional as F

from model.base_model import BaseModel, DLLMOutput
from model.model_utils import decode_history, compute_decoding_order_correlation_from_history
from utils.perf_utils import measure_time_mem
from utils.utils import insert_import_path


with insert_import_path("src/Score-Entropy-Discrete-Diffusion"):
    from load_model import load_model
    import sampling


class SeddPredictorType(str, Enum):
    NONE = 'none'
    EULER = 'euler'
    ANALYTIC = 'analytic'


@dataclass
class SeddGenerationConfig:
    accel_framework: Optional[str] = None

    max_tokens: int = 128
    steps: Optional[int] = 128
    block_length: Optional[int] = None
    temperature: float = 0.0
    predictor: SeddPredictorType = SeddPredictorType.ANALYTIC
    # block_length: int = 128

    # fast-dllm specific
    # fast_dllm_threshold: float = 0.9  # TODO assert none if not using LOW_CONFIDENCE_THRESHOLD
    # fast_dllm_factor: Optional[float] = None
    # fast_dllm_use_cache: bool = False
    # fast_dllm_dual_cache: bool = False

    # remdm_number: Optional[int] = None

    # @property
    # def num_blocks(self):
    #     return self.max_tokens // self.block_length

    def __post_init__(self):
        assert self.block_length is None or self.block_length == self.max_tokens, "Block length must be equal to max tokens if specified."
        assert self.temperature == 1.0


class SeddModel(BaseModel):
    def __init__(self, model_name, accel_framework=None):
        assert  accel_framework is None

        self.device = torch.device("cuda")
        self.model, self.graph, self.noise = load_model(model_name, self.device)
        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False, dynamic=True)

        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        self.accel_framework = accel_framework

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def model_generate(self, input_ids, gen_config, output_history=False):
        input_locs = torch.arange(len(input_ids[0]), device=input_ids.device)

        def proj_fun(x):
            x[:, input_locs] = input_ids
            return x

        batch_dims = (1, input_ids.shape[1] + gen_config.max_tokens)
        sampling_fn = sampling.get_pc_sampler(self.graph, self.noise, batch_dims, gen_config.predictor, gen_config.steps, denoise=True, device=self.device, proj_fun=proj_fun)

        input_output_ids = sampling_fn(self.model)
        nfe = gen_config.steps
        history = None

        return input_output_ids, nfe, history

    def generate(self, messages, gen_config=None, output_history=False):
        if isinstance(messages, list):
            # prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt = "\n\n".join(m["content"] for m in messages) + "\n\n"
        else:
            prompt = messages
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        gen_config = SeddGenerationConfig(accel_framework=self.accel_framework, **gen_config)

        input_output_ids, nfe, history = self.model_generate(input_ids, gen_config, output_history=output_history)
        output_ids = input_output_ids[:, input_ids.shape[1]:]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # assert not (output_history and history is None), "History should not be None if output_history is True."

        if history is not None:
            decoding_order, decoding_order_corrs = compute_decoding_order_correlation_from_history(self.tokenizer, history)
        else:
            decoding_order, decoding_order_corrs = None, None

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
