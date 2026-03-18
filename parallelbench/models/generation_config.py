from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from parallelbench.models.unmasking_registry import get_method_type


DEFAULT_VALID_METHODS = {
    "random",
    "origin",
    "confidence_topk",
    "topk_margin",
    "entropy_topk",
    "confidence_threshold",
    "confidence_factor",
}

# Backward-compatible alias
DEFAULT_VALID_STRATEGIES = DEFAULT_VALID_METHODS


@dataclass
class BaseGenerationConfig(ABC):
    """Common generation config shared by all model types."""

    max_tokens: int = 128
    temperature: float = 0.0


@dataclass
class DllmGenerationConfig(BaseGenerationConfig):
    """Generation config for discrete diffusion language models (dLLMs)."""

    unmasking: Optional[str] = None
    steps: Optional[int] = 128
    block_length: Optional[int] = None
    alg_temp: float = 0.0
    alg_threshold: Optional[float] = None
    alg_factor: Optional[float] = None
    valid_methods: set = field(default_factory=lambda: set(DEFAULT_VALID_METHODS))

    def __post_init__(self):
        if self.block_length is None:
            self.block_length = self.max_tokens
        self._validate_dllm_gen_configs()
        self._validate_unmasking()

    @property
    def num_blocks(self):
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.block_length, int) or self.block_length <= 0:
            raise ValueError("block_length must be a positive integer")
        return self.max_tokens // self.block_length

    def _validate_dllm_gen_configs(self):
        if self.steps is not None:
            if self.steps > self.max_tokens:
                raise ValueError("steps cannot be greater than max_tokens")

            if self.steps % self.num_blocks != 0:
                raise ValueError("steps must be divisible by num_blocks")

        if self.max_tokens % self.block_length != 0:
            raise ValueError("max_tokens must be divisible by block_length")

    def _validate_unmasking(self):
        if self.unmasking not in self.valid_methods:
            raise ValueError(f"Unsupported unmasking method: {self.unmasking}")

        self.is_threshold_unmasking = get_method_type(self.unmasking) == "threshold"
        self.is_factor_unmasking = get_method_type(self.unmasking) == "factor"
        self.is_default_unmasking = get_method_type(self.unmasking) == "topk"

        # Validate default unmasking
        if self.is_default_unmasking:
            if self.alg_threshold is not None and self.alg_threshold != 0.0:
                raise ValueError(
                    "alg_threshold must be None or 0.0 for default unmasking methods"
                )
            if self.alg_factor is not None and self.alg_factor != 1.0:
                raise ValueError(
                    "alg_factor must be None or 1.0 for default unmasking methods"
                )

        # Validate threshold unmasking
        if self.is_threshold_unmasking:
            if self.alg_threshold is None:
                raise ValueError(
                    f"alg_threshold must be provided for {self.unmasking} algorithm"
                )
            if self.alg_factor is not None:
                raise ValueError(
                    f"alg_factor must be None for {self.unmasking} algorithm"
                )

        # Validate factor unmasking
        if self.is_factor_unmasking:
            if self.alg_factor is None:
                raise ValueError(
                    f"alg_factor must be provided for {self.unmasking} algorithm"
                )
            if self.alg_threshold is not None:
                raise ValueError(
                    f"alg_threshold must be None for {self.unmasking} algorithm"
                )

    def _backend_unmasking(self) -> str:
        """Map internal unmasking name to the backend name."""
        backend_map: dict[str, str] = {}
        return backend_map.get(self.unmasking, self.unmasking)

    def to_generation_kwargs(self) -> dict:
        return dict(
            steps=self.steps,
            gen_length=self.max_tokens,
            block_length=self.block_length,
            temperature=self.temperature,
            alg_temp=self.alg_temp,
            unmasking=self._backend_unmasking(),
            threshold=self.alg_threshold,
            factor=self.alg_factor,
        )


@dataclass
class ARGenerationConfig(BaseGenerationConfig):
    """Generation config for autoregressive models."""

    pass


@dataclass
class ApiGenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.0
