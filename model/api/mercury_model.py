import json
import os
from dataclasses import dataclass

import requests

from model.base_model import ApiModel, DLLMOutput
from model.generation_config import ApiGenerationConfig
from model.registry import ModelRegistry


@dataclass
class MercuryGenerationConfig(ApiGenerationConfig):
    # presence_penalty: float = 1.5
    pass


@ModelRegistry.register(lambda name: name in ("mercury", "mercury-coder"))
class MercuryModel(ApiModel):
    def __init__(self, model_name):
        assert model_name in ("mercury", "mercury-coder")

        self.model_name = model_name

        api_key = os.environ.get("INCEPTION_API_KEY")
        if api_key is None:
            raise EnvironmentError(
                "INCEPTION_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        self.api_key = api_key

    def generate(self, messages, gen_config=None, output_history=False):
        gen_config = MercuryGenerationConfig(**gen_config)

        response = requests.post(
            "https://api.inceptionlabs.ai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": gen_config.max_tokens,
                "temperature": gen_config.temperature,
                "stream": True,
                "diffusing": True,
            },
        )
        response.raise_for_status()

        try:
            output_json_lines = response.content.decode()
            output_json_lines = (
                "{" + output_json_lines.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            )
            output_json_lines = output_json_lines.split("\n\ndata: ")
            output_full = [json.loads(o) for o in output_json_lines if o.strip()]
        except (IndexError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to parse Mercury API SSE response: {e}"
            ) from e

        history = [
            o["choices"][0]["delta"].get("content")
            for o in output_full
            if o.get("choices")
        ]
        history = [h for h in history if h is not None]

        if not history:
            raise ValueError(
                "Mercury API returned no content in the response. "
                "Check the request parameters and API status."
            )
        output = history[-1]

        return DLLMOutput(
            output=output,
            output_full=output_full,
            input_ids=None,
            output_ids=None,
            pad_token_id=None,
            nfe=len(history),
            history=None,
        )
