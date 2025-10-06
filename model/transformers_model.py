
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
class TransformersGenerationConfig:
    max_tokens: int
    temperature: float = 0.0

    def __post_init__(self):
        pass

    def to_generate_kwargs(self):
        assert self.temperature == 0.0, "TransformersGenerationConfig only supports temperature=0.0"

        return dict(
            top_p=1,
            top_k=1,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            # use_beam_search=False,
            max_new_tokens=self.max_tokens,
        )


class TransformersModel(BaseModel):
    def __init__(self, model_name, chat_template_kwargs=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chat_template_kwargs = chat_template_kwargs or {}

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def generate(self, messages, gen_config=None, output_history=False):
        generate_kwargs = TransformersGenerationConfig(**gen_config).to_generate_kwargs()

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        model_output = self.model.generate(**model_inputs, **generate_kwargs)
        output_ids = model_output[:, model_inputs["input_ids"].shape[1]:]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        return DLLMOutput(
            output=output,
            input_ids=None,
            output_ids=None,
            pad_token_id=None,
            nfe=0,
            history=None
        )
