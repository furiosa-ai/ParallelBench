
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

from model.base_model import BaseModel, DLLMOutput
from model.model_utils import decode_history
from utils.perf_utils import measure_time_mem
from utils.utils import insert_import_path



@dataclass
class vllmGenerationConfig:
    max_tokens: int
    temperature: float = 0.0

    def __post_init__(self):
        pass

    def to_sampling_params(self):
        from vllm import SamplingParams
        assert self.temperature == 0.0, "vllmGenerationConfig only supports temperature=0.0"

        return SamplingParams(
            best_of=1,
            temperature=self.temperature,
            top_p=1,
            top_k=-1,
            # use_beam_search=False,
            max_tokens=self.max_tokens,
            presence_penalty=0,
            frequency_penalty=0,
            detokenize=True,
        )


class vllmModel(BaseModel):
    def __init__(self, model_name, chat_template_kwargs=None, max_model_len=2**15):
        from vllm import LLM

        # assert "Qwen3" not in model_name, "vllm does not support Qwen3 models without thinking"
        assert chat_template_kwargs is None, "vllm does not support chat template kwargs"

        self.model = LLM(model=model_name, dtype=torch.bfloat16, max_model_len=max_model_len)

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def generate(self, messages, gen_config=None, output_history=False):
        sampling_params = vllmGenerationConfig(**gen_config).to_sampling_params()

        output = self.model.chat(messages, sampling_params)
        output_txt = output[0].outputs[0].text

        return DLLMOutput(
            output=output_txt,
            input_ids=None,
            output_ids=None,
            pad_token_id=None,
            nfe=0,
            history=None
        )
