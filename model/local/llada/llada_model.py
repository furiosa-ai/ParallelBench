from dataclasses import dataclass
from typing import Optional, Union, List

import torch
from transformers import PreTrainedModel

from dataset.parallel_bench.data.task import PARALLEL_BENCH_MASK_TOKEN
from model.base_model import BaseModel, DLLMOutput
from model.generation_config import BaseGenerationConfig
from model.local.generate import generate
from model.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)
from model.registry import ModelRegistry
from utils.perf_utils import measure_time_mem
from .constants import LLADA_MASK_TOKEN_ID


@dataclass
class LladaGenerationConfig(BaseGenerationConfig):
    remasking: str = "low_confidence" # Set the default remasking strategy
    block_length: int = 128 # Set the default block length


@ModelRegistry.register(
    lambda name: name
    in ("GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-1.5")
    or "llada" in name.lower()
)
class LladaModel(BaseModel):
    def __init__(self, model_name: str, accel_framework: Optional[str]=None):
        """Initialize the LladaModel.

        Args:
            model_name (str): The name of the model to load.
            accel_framework (Optional[str]): The acceleration framework to use. Defaults to None.
        """
        super().__init__(model_name, accel_framework)

        self.patch_model_forward(self.model, LLADA_MASK_TOKEN_ID)
        self.tokenizer.mask_token_id = LLADA_MASK_TOKEN_ID
        self.mask_id = LLADA_MASK_TOKEN_ID

    def patch_model_forward(self, model: PreTrainedModel, mask_token_id: int):
        """Patch the model's forward method to prevent sampling the mask token.

        Args:
            model (PreTrainedModel): The model whose forward method will be patched.
            mask_token_id (int): The token ID of the mask token to be prevented from sampling.
        Returns:
            None
        Side Effects:
            Modifies the model's forward method in-place to set the logits of the mask token to -inf.
        """
        fwd_fn = model.__class__.forward

        def wrapped_forward(*args, **kwargs):
            output = fwd_fn(*args, **kwargs)
            output.logits[:, :, mask_token_id] = -float(
                "Inf"
            )  # cannot sample mask token
            return output

        model.__class__.forward = wrapped_forward

    @measure_time_mem("generate")
    def _generate(self, input_ids: torch.Tensor, gen_config: LladaGenerationConfig, output_history: bool=False, output0_ids: Optional[torch.Tensor]=None) -> (torch.Tensor, int, Optional[dict]):
        """Internal generation method that handles the actual generation logic based on the acceleration framework.
        Args:
            input_ids (torch.Tensor): The input token IDs for generation.
            gen_config (LladaGenerationConfig): The generation configuration parameters.
            output_history (bool): Whether to output the generation history. Defaults to False.
            output0_ids (Optional[torch.Tensor]): Optional initial output token IDs for generation. Defaults to None.
        Returns:
            Tuple[torch.Tensor, int, Optional[dict]]: A tuple containing the generated token IDs, the number of forward evaluations (NFE), and optionally the generation history if output_history is True.
        """
        if self.accel_framework == "fast_dllm":
            # FIXME: implement fast dLLM generation
            raise NotImplementedError("Fast dLLM LLADA generation is not implemented yet.")

        gen_kwargs = gen_config.to_generation_kwargs()

        return generate(
            self.model,
            input_ids,
            mask_id=self.mask_id,
            **gen_kwargs,
            output_history=output_history,
            output0_ids=output0_ids,
        )

    def generate(
        self,
        messages: Union[List[str], str],
        output_prefix:torch.Tensor=None,
        gen_config:dict=None,
        output_history:bool=False
    )-> DLLMOutput:
        """Generate output from input messages.
        Args:
            messages (Union[List[str], str]): Input messages in chat format or a single string prompt.
            output_prefix (torch.Tensor, optional): Optional prefix for the output. Defaults to None.
            gen_config (dict, optional): Generation configuration parameters. Defaults to None.
            output_history (bool, optional): Whether to output the generation history. Defaults to False.
        Returns:
            DLLMOutput: Generated output with metadata.
        """
        
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = messages

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        gen_config = LladaGenerationConfig(
            accel_framework=self.accel_framework, **gen_config
        )

        if output_prefix is not None:
            output_prefix = output_prefix.replace(
                PARALLEL_BENCH_MASK_TOKEN, self.tokenizer.mask_token
            )
            output0_ids = self.tokenizer(
                output_prefix,
                return_tensors="pt",
                padding="max_length",
                max_length=gen_config.max_tokens,
            ).input_ids.to(self.model.device)
            if output0_ids.shape[1] != gen_config.max_tokens:
                raise ValueError(
                    f"output_prefix length {output0_ids.shape[1]} does not match gen_config.max_tokens {gen_config.max_tokens}"
                )
        else:
            output0_ids = None

        input_output_ids, nfe, history = self._generate(
            input_ids,
            gen_config,
            output_history=output_history,
            output0_ids=output0_ids,
        )
        output_ids = input_output_ids[:, input_ids.shape[1] :]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        assert not (output_history and history is None), (
            "History should not be None if output_history is True."
        )
        
        decoding_order, decoding_order_corrs = (
            compute_decoding_order_correlation_from_history(self.tokenizer, history)
        )

        return DLLMOutput(
            output=output,
            input_ids=input_ids,
            output_ids=output_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            nfe=nfe,
            history=decode_history(self.tokenizer, history),
            decoding_order=decoding_order,
            decoding_order_corrs=decoding_order_corrs,
        )
