from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

VALID_ACCEL_FRAMEWORKS = {None, "vllm", "transformers", "fast_dllm"}

class BaseModel(ABC):
    def __init__(self, model_name, accel_framework=None):

        if accel_framework not in VALID_ACCEL_FRAMEWORKS:
            raise ValueError(f"Invalid accel_framework: {accel_framework}. Valid options are: {VALID_ACCEL_FRAMEWORKS}")

        if accel_framework == "fast_dllm":
            raise NotImplementedError("Fast dLLM model loading is not implemented yet.")
        self.accel_framework = accel_framework

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @abstractmethod
    def generate(self, messages, **kwargs) -> "DLLMOutput":
        """Generate output from input messages.

        Args:
            messages: Input messages in chat format
            **kwargs: Additional generation parameters (model-specific)

        Returns:
            DLLMOutput: Generated output with metadata
        """
        pass


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

    def __post_init__(self):
        pass
