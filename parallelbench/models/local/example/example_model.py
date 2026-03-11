from dataclasses import dataclass
from typing import Union

import torch

from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.generation_config import DllmGenerationConfig
from parallelbench.models.model_utils import decode_history


@dataclass
class ExampleGenerationConfig(DllmGenerationConfig):
    example_field: str = "example_default_value"

    def _validate_unmasking(self):
        super()._validate_unmasking()
        # Add any additional validation for example_field if needed
        assert isinstance(self.example_field, str), "example_field must be a string."

    def to_generation_kwargs(self):
        gen_kwargs = super().to_generation_kwargs()
        gen_kwargs.update(
            {
                "example_field": self.example_field,
            }
        )
        return gen_kwargs


class ExampleModel(LocalModel):
    """
    Example LocalModel implementation.

    This class is not registered in ModelRegistry by default. To use it in a
    real deployment, add an appropriate @ModelRegistry.register(...) decorator
    with a predicate that matches your desired model name(s).
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        # Initialize the model here (e.g., load weights, set up tokenizer, etc.)

    def generate(
        self,
        messages: Union[list[str], str],
        output_prefix: torch.Tensor = None,
        gen_config: dict = None,
        output_history: bool = False,
    ) -> "DLLMOutput":
        """Generate output from input messages.
        Args:
            messages (Union[List[str], str]): Input messages in chat format or a single string prompt.
            output_prefix (torch.Tensor, optional): Optional prefix for the output. Defaults to None.
            gen_config (dict, optional): Generation configuration parameters. Defaults to None.
            output_history (bool, optional): Whether to output the generation history. Defaults to False.
        Returns:
            DLLMOutput: Generated output with metadata.
        """

        # Do something here

        output = "generated output"
        input_ids = torch.tensor([[0, 1, 2]])  # Dummy input
        output_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Dummy output
        nfe = 10  # Dummy number of function evaluations
        history = []  # Dummy history
        decoding_order = []  # Dummy decoding order
        decoding_order_corrs = []  # Dummy decoding order correlations

        return DLLMOutput(
            output=output,
            input_ids=input_ids,
            output_ids=output_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,
            history=decode_history(self.tokenizer, history) if output_history else None,
            decoding_order=decoding_order if output_history else None,
            decoding_order_corrs=decoding_order_corrs if output_history else None,
        )
