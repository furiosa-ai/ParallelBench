"""Global metadata store for passing dLLM-specific data between generate_until() and process_results().

lm-eval's generate_until() can only return list[str], but dLLM models produce
rich metadata (NFE, decoding history, decoding order). This store acts as a
side-channel buffer: the model wrapper appends metadata during generation, and
the task's process_results() pops it in the same order.

Order guarantee: lm-eval processes requests sequentially per task, so
append/pop order is consistent.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GenerationMetadata:
    """Metadata captured from a single dLLM generation call."""

    nfe: int = 0
    history: Optional[list] = None
    decoding_order: Optional[torch.Tensor] = None
    decoding_order_corrs: Optional[dict] = None
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    perf_stats: Optional[dict] = None


class MetadataStore:
    """Thread-safe global singleton for buffering generation metadata.

    Usage in model wrapper:
        MetadataStore.instance().append(GenerationMetadata(nfe=42, ...))

    Usage in task process_results:
        meta = MetadataStore.instance().pop()
    """

    _instance: Optional[MetadataStore] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._buffer: deque[GenerationMetadata] = deque()

    @classmethod
    def instance(cls) -> MetadataStore:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    def append(self, metadata: GenerationMetadata) -> None:
        self._buffer.append(metadata)

    def pop(self) -> GenerationMetadata:
        """Pop the oldest metadata entry. Returns empty metadata if buffer is empty."""
        if self._buffer:
            return self._buffer.popleft()
        return GenerationMetadata()

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
