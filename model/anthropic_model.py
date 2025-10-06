
from dataclasses import dataclass
from model.base_model import BaseModel, DLLMOutput


@dataclass
class AnthropicGenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.0


# export ANTHROPIC_API_KEY='your-api-key-here'
class AnthropicModel(BaseModel):
    def __init__(self, model_name):
        import anthropic

        self.model_name = model_name

        self.client = anthropic.Anthropic()

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError("AnthropicModel does not support fill method.")

    def generate(self, messages, gen_config=None, output_history=False):
        gen_config = AnthropicGenerationConfig(**gen_config)

        gen_kwargs = {
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
        }

        if gen_kwargs["temperature"] is None:
            del gen_kwargs["temperature"]

        message = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            **gen_kwargs,
        )

        output = message.content[0].text

        return DLLMOutput(
            output=output,
            input_ids=None,
            output_ids=None,
            pad_token_id=None,
            nfe=0,
            history=None
        )
