
from dataclasses import dataclass
import json
import os

from model.base_model import BaseModel, DLLMOutput

import requests


@dataclass
class MercuryGenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.0
    # presence_penalty: float = 1.5


class MercuryModel(BaseModel):
    def __init__(self, model_name):
        assert model_name in ("mercury", "mercury-coder")

        self.model_name = model_name
        self.api_key = os.environ["INCEPTION_API_KEY"]

    def fill(self, prompt, suffix, gen_config=None):
        gen_config = MercuryGenerationConfig(**gen_config)

        response = requests.post('https://api.inceptionlabs.ai/v1/fim/completions', headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }, json={
            "model": self.model_name,
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
        })

        return response.json()['choices'][0]['message']['content']

    def generate(self, messages, gen_config=None, output_history=False):
        gen_config = MercuryGenerationConfig(**gen_config)

        response = requests.post('https://api.inceptionlabs.ai/v1/chat/completions', headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }, json={
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
            "stream": True,
            "diffusing": True,
        })

        output_json_lines = response.content.decode()
        output_json_lines = "{" + output_json_lines.split("{", 1)[1].rsplit("}", 1)[0] + "}"
        output_json_lines = output_json_lines.split("\n\ndata: ")

        output_full = [json.loads(o) for o in output_json_lines if o.strip()]
        history = [o['choices'][0]['delta'].get('content') for o in output_full]
        history = [h for h in history if h is not None]
        output = history[-1]

        return DLLMOutput(
            output=output,
            output_full=output_full,
            input_ids=None,
            output_ids=None,
            pad_token_id=None,
            nfe=len(history),
            history=None
        )
