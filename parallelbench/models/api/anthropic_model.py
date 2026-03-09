from dataclasses import dataclass
from parallelbench.models.base_model import ApiModel, DLLMOutput
from parallelbench.models.generation_config import ApiGenerationConfig
from parallelbench.models.registry import ModelRegistry


@dataclass
class AnthropicGenerationConfig(ApiGenerationConfig):
    pass


@ModelRegistry.register(lambda name: name.startswith("claude-"))
class AnthropicModel(ApiModel):
    def __init__(self, model_name):
        import anthropic

        self.model_name = model_name

        self.client = anthropic.Anthropic()

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
            history=None,
        )
