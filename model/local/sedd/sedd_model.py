from dataclasses import dataclass
from enum import Enum

import torch
from sedd import sampling
from sedd.load_model import load_model
from transformers import GPT2TokenizerFast

from model.base_model import BaseModel, DLLMOutput
from model.generation_config import BaseGenerationConfig
from model.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)
from model.registry import ModelRegistry
from utils.perf_utils import measure_time_mem


class SeddPredictorType(str, Enum):
    NONE = "none"
    EULER = "euler"
    ANALYTIC = "analytic"


@dataclass
class SeddGenerationConfig(BaseGenerationConfig):
    predictor: SeddPredictorType = SeddPredictorType.ANALYTIC


    def __post_init__(self):
        assert self.block_length is None or self.block_length == self.max_tokens, (
            "Block length must be equal to max tokens if specified."
        )
        assert self.temperature == 1.0, "Temperature must be 1.0 for SEDD models."

    def to_generation_kwargs(self) -> dict:
        return dict(
            predictor=self.predictor,
            steps=self.steps,
            max_tokens=self.max_tokens,
        )


@ModelRegistry.register(lambda name: "sedd" in name.lower())
class SeddModel(BaseModel):
    def __init__(self, model_name, accel_framework=None):
        assert accel_framework is None

        self.device = torch.device("cuda")
        self.model, self.graph, self.noise = load_model(model_name, self.device)
        self.model.eval()

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        self.accel_framework = accel_framework


    @measure_time_mem("generate")
    def _generate(self, input_ids, gen_config, output_history=False):
        generate_kwargs = gen_config.to_generation_kwargs()

        input_locs = torch.arange(len(input_ids[0]), device=input_ids.device)

        def proj_fun(x):
            x[:, input_locs] = input_ids
            return x

        batch_dims = (1, input_ids.shape[1] + generate_kwargs["max_tokens"])
        sampling_fn = sampling.get_pc_sampler(
            self.graph,
            self.noise,
            batch_dims,
            generate_kwargs["predictor"],
            generate_kwargs["steps"],
            denoise=True,
            device=self.device,
            proj_fun=proj_fun,
        )

        input_output_ids = sampling_fn(self.model)
        nfe = generate_kwargs["steps"]
        history = None

        return input_output_ids, nfe, history

    def generate(self, messages, gen_config=None, output_history=False):
        if isinstance(messages, list):
            prompt = "\n\n".join(m["content"] for m in messages) + "\n\n"
        else:
            prompt = messages

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        gen_config = SeddGenerationConfig(
            accel_framework=self.accel_framework, **gen_config
        )

        input_output_ids, nfe, history = self._generate(
            input_ids, gen_config, output_history=output_history
        )
        output_ids = input_output_ids[:, input_ids.shape[1] :]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        if history is not None:
            decoding_order, decoding_order_corrs = (
                compute_decoding_order_correlation_from_history(self.tokenizer, history)
            )
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
