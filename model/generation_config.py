from abc import ABC
from dataclasses import dataclass, field
from typing import Optional


VALID_BASE_STRATEGIES = {
    "random",
    "low_confidence",
    "topk_margin",
    "entropy"
}


@dataclass
class BaseGenerationConfig(ABC):
    """Common generation config shared by all model types."""
    accel_framework: Optional[str] = None
    max_tokens: int = 128
    temperature: float = 0.0


@dataclass
class DllmGenerationConfig(BaseGenerationConfig):
    """Generation config for discrete diffusion language models (dLLMs)."""
    remasking: Optional[str] = None
    steps: Optional[int] = 128
    block_length: Optional[int] = None
    alg_temp: float = 0.0
    alg_threshold: Optional[float] = None
    alg_factor: Optional[float] = None
    use_fast_dllm_cache: bool = False
    use_fast_dllm_dual_cache: bool = False
    valid_base_strategies: set = field(default_factory=lambda: set(VALID_BASE_STRATEGIES))

    def __post_init__(self):
        self._validate_dllm_gen_configs()
        self._validate_remasking()

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

        # Validate fast_dllm cache usage
        if self.accel_framework != "fast_dllm":
            if self.use_fast_dllm_cache:
                raise ValueError(
                    "use_fast_dllm_cache can only be True when accel_framework is 'fast_dllm'"
                )

            if self.use_fast_dllm_dual_cache:
                raise ValueError(
                    "use_fast_dllm_dual_cache can only be True when accel_framework is 'fast_dllm'"
                )

    def _validate_remasking(self):
        base = self.remasking.removesuffix("_threshold").removesuffix("_factor")
        if base not in self.valid_base_strategies:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")

        self.is_threshold_remasking = self.remasking.endswith("_threshold")
        self.is_factor_remasking = self.remasking.endswith("_factor")
        self.is_default_remasking = not (self.is_threshold_remasking or self.is_factor_remasking)

        # Validate default remasking
        if self.is_default_remasking:
            if self.alg_threshold is not None and self.alg_threshold != 0.0:
                raise ValueError(
                    "alg_threshold must be None or 0.0 for default remasking strategies"
                )
            if self.alg_factor is not None and self.alg_factor != 1.0:
                raise ValueError("alg_factor must be None or 1.0 for default remasking strategies")

        # Validate threshold remasking
        if self.is_threshold_remasking:
            if self.alg_threshold is None:
                raise ValueError(f"alg_threshold must be provided for {self.remasking} algorithm")
            if self.alg_factor is not None:
                raise ValueError(f"alg_factor must be None for {self.remasking} algorithm")

        # Validate factor remasking
        if self.is_factor_remasking:
            if self.alg_factor is None:
                raise ValueError(f"alg_factor must be provided for {self.remasking} algorithm")
            if self.alg_threshold is not None:
                raise ValueError(f"alg_threshold must be None for {self.remasking} algorithm")

    def to_generation_kwargs(self) -> dict:
        return dict(
            steps=self.steps,
            gen_length=self.max_tokens,
            block_length=self.block_length,
            temperature=self.temperature,
            alg_temp=self.alg_temp,
            remasking=self.remasking,
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
