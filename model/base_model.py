from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, Dict

import torch

VALID_ACCEL_FRAMEWORKS = {None, "vllm", "transformers", "fast_dllm"}


class BaseModel(ABC):
    """Abstract base class for all models (API and local)."""

    @abstractmethod
    def generate(
        self, messages: Union[List[str], str], gen_config: Dict=None, output_prefix: Optional[torch.tensor]=None, output_history: bool=False
    ) -> "DLLMOutput":
        """Generate output from input messages.

        Args:
            messages: Input messages in chat format
            gen_config: Generation configuration dictionary passed to the model-specific GenerationConfig
            output_prefix: Prefix token tensor to prepend to the generated output
            output_history: If True, return intermediate decoding states in history field of DLLMOutput
        Returns:
            DLLMOutput: Generated output with metadata
        """
        pass


class ApiModel(BaseModel):
    """Base class for API-backed models (no local weight loading)."""
    pass


class LocalModel(BaseModel):
    """Base class for local models that load weights via transformers."""

    def __init__(self, model_name, accel_framework=None):
        from transformers import AutoModel, AutoTokenizer

        if accel_framework not in VALID_ACCEL_FRAMEWORKS:
            raise ValueError(
                f"Invalid accel_framework: {accel_framework}. "
                f"Valid options are: {VALID_ACCEL_FRAMEWORKS}"
            )

        if accel_framework == "fast_dllm":
            raise NotImplementedError("Fast dLLM model loading is not implemented yet.")
        self.accel_framework = accel_framework

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


@dataclass
class DLLMOutput:
    output: str
    output_full: Optional[dict] = None
    nfe: int = 0
    input_ids: Optional[torch.Tensor] = None
    output_ids: Optional[torch.Tensor] = None
    history: Optional[dict] = None
    pad_token_id: Optional[int] = None
    decoding_order: Optional[torch.Tensor] = None
    decoding_order_corrs: Optional[dict] = None

    @property
    def input_length(self):
        return self.input_ids.size(1) if self.input_ids is not None else None

    @property
    def output_length(self):
        if self.pad_token_id is None:
            return self.output_ids.size(1) if self.output_ids is not None else None

        return (
            (self.output_ids.squeeze() != self.pad_token_id).sum().item()
            if self.output_ids is not None
            else None
        )
