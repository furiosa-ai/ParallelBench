import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from transformers import AutoModel, AutoTokenizer
import torch


class BaseModel(ABC):
    """Abstract base class for all models (API and local)."""

    @abstractmethod
    def generate(
        self,
        messages: Union[List[str], str],
        gen_config: Dict = None,
        output_prefix: Optional[str] = None,
        output_history: bool = False,
    ) -> "DLLMOutput":
        """Generate output from input messages.

        Args:
            messages: Input messages in chat format
            gen_config: Generation configuration dictionary passed to the model-specific GenerationConfig
            output_prefix: Prefix string to prepend to the generated output
            output_history: If True, return intermediate decoding states in history field of DLLMOutput
        Returns:
            DLLMOutput: Generated output with metadata
        """
        pass


class ApiModel(BaseModel):
    """Base class for API-backed models (no local weight loading)."""

    pass


class LocalModel(BaseModel):
    """Base class for local models that load weights via transformers.

    Note: trust_remote_code=True is required for research models (LLaDA, Dream, etc.)
    and is hardcoded in from_pretrained() calls. Some subclasses (SEDD, Trado) handle
    model loading independently and bypass __init__; they also use trust_remote_code=True.
    """

    def __init__(self, model_name, model_class=AutoModel):
        local_rank = os.environ.get("LOCAL_RANK")
        device_map = f"cuda:{local_rank}" if local_rank is not None else "cuda"
        self.model = model_class.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )


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
