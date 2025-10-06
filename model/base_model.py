from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class BaseModel:
    @property
    def num_workers(self):
        return 0


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
        
        return (self.output_ids.squeeze() != self.pad_token_id).sum().item() if self.output_ids is not None else None

    def __post_init__(self):
        pass
