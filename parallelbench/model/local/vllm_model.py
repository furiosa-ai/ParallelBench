from dataclasses import dataclass

import torch

from parallelbench.model.base_model import BaseModel, DLLMOutput


@dataclass
class vllmGenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.0

    def to_sampling_params(self):
        from vllm import SamplingParams

        assert self.temperature == 0.0, (
            "vllmGenerationConfig only supports temperature=0.0"
        )

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
        assert chat_template_kwargs is None, (
            "vllm does not support chat template kwargs"
        )

        self.model = LLM(
            model=model_name, dtype=torch.bfloat16, max_model_len=max_model_len
        )

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
            history=None,
        )
