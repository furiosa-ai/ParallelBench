from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import types

import torch
from transformers import AutoModel, PreTrainedModel

from parallelbench.datasets.task import PARALLEL_BENCH_MASK_TOKEN
from parallelbench.models.base_model import DLLMOutput, LocalModel
from parallelbench.models.generation_config import DllmGenerationConfig
from parallelbench.models.local.generate import generate, generate_batch
from parallelbench.models.local.llada.constants import (
    LLADA_MASK_TOKEN_ID,
    LLADA_VALID_STRATEGIES,
)
from parallelbench.models.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)
from parallelbench.models.registry import ModelRegistry


@dataclass
class LladaGenerationConfig(DllmGenerationConfig):
    unmasking: str = "confidence_topk"  # Set the default unmasking strategy
    block_length: int = 128  # Set the default block length

    valid_strategies: set = field(default_factory=lambda: set(LLADA_VALID_STRATEGIES))

    pass


@ModelRegistry.register(
    lambda name: (
        name in ("GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-1.5")
        or "llada" in name.lower()
    )
)
class LladaModel(LocalModel):
    def __init__(self, model_name: str):
        """Initialize the LladaModel.

        Args:
            model_name (str): The name of the model to load.
        """
        super().__init__(model_name, model_class=AutoModel)

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

        def wrapped_forward(self_model, *args, **kwargs):
            output = self_model.__class__.forward(self_model, *args, **kwargs)
            output.logits[:, :, mask_token_id] = -float(
                "Inf"
            )  # cannot sample mask token
            return output

        model.forward = types.MethodType(wrapped_forward, model)

    def _generate(
        self,
        input_ids: torch.Tensor,
        gen_config: LladaGenerationConfig,
        output_history: bool = False,
        output0_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int, Optional[list]]:
        """Internal generation method that handles the actual generation logic based on the acceleration framework.
        Args:
            input_ids (torch.Tensor): The input token IDs for generation.
            gen_config (LladaGenerationConfig): The generation configuration parameters.
            output_history (bool): Whether to output the generation history. Defaults to False.
            output0_ids (Optional[torch.Tensor]): Optional initial output token IDs for generation. Defaults to None.
        Returns:
            Tuple[torch.Tensor, int, Optional[dict]]: A tuple containing the generated token IDs, the number of forward evaluations (NFE), and optionally the generation history if output_history is True.
        """
        gen_kwargs = gen_config.to_generation_kwargs()

        gen_kwargs.update(
            {
                "model": self.model,
                "prompt": input_ids,
                "mask_id": self.mask_id,
                "output_history": output_history,
                "output0_ids": output0_ids,
            }
        )

        return generate(**gen_kwargs)

    @property
    def supports_batch(self) -> bool:
        return True

    def generate(
        self,
        messages: Union[list[str], str],
        output_prefix: torch.Tensor = None,
        gen_config: dict = None,
        output_history: bool = False,
    ) -> DLLMOutput:
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
        gen_config = LladaGenerationConfig(**gen_config)

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

        if history is not None:
            decoding_order, decoding_order_corrs = (
                compute_decoding_order_correlation_from_history(self.tokenizer, history)
            )
        else:
            decoding_order, decoding_order_corrs = None, None

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

    def generate_batch(
        self,
        messages_list: List[Union[List[dict], str]],
        gen_config: Dict = None,
        output_prefix_list: Optional[List[Optional[str]]] = None,
        output_history: bool = False,
    ) -> List[DLLMOutput]:
        """Generate outputs for a batch of inputs in a single forward pass.

        NOTE: This is NOT an official LLaDA implementation. LLaDA does not officially
        support batched generation (see https://github.com/ML-GSAI/LLaDA/issues/78).

        Uses [prompt | mask | pad] layout where pad tokens follow mask tokens,
        preserving correct RoPE position encoding without model modification.

        Args:
            messages_list: List of per-sample messages.
            gen_config: Generation configuration shared across the batch.
            output_prefix_list: Not supported in batch mode (ignored).
            output_history: Whether to output the generation history.

        Returns:
            List of DLLMOutput in the same order as messages_list.
        """
        # Tokenize all prompts into 1D tensors
        prompts = []
        for messages in messages_list:
            if isinstance(messages, list):
                prompt_str = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt_str = messages
            input_ids = self.tokenizer(
                prompt_str, return_tensors="pt"
            ).input_ids.squeeze(0)
            prompts.append(input_ids)

        gen_config = LladaGenerationConfig(**gen_config)
        gen_kwargs = gen_config.to_generation_kwargs()

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        gen_kwargs.update(
            {
                "model": self.model,
                "prompts": prompts,
                "mask_id": self.mask_id,
                "pad_id": pad_id,
                "output_history": output_history,
            }
        )

        x, nfe, histories = generate_batch(**gen_kwargs)

        prompt_lengths = [p.shape[0] for p in prompts]
        outputs = []
        for i in range(len(messages_list)):
            pl = prompt_lengths[i]
            input_ids = prompts[i].unsqueeze(0)
            output_ids = x[i, pl : pl + gen_config.max_tokens].unsqueeze(0)
            output_text = self.tokenizer.decode(
                output_ids.squeeze(0), skip_special_tokens=True
            )

            sample_history = histories[i] if histories else None
            if sample_history is not None:
                decoding_order, decoding_order_corrs = (
                    compute_decoding_order_correlation_from_history(
                        self.tokenizer, sample_history
                    )
                )
            else:
                decoding_order, decoding_order_corrs = None, None

            outputs.append(
                DLLMOutput(
                    output=output_text,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                    nfe=nfe,
                    history=decode_history(self.tokenizer, sample_history),
                    decoding_order=decoding_order,
                    decoding_order_corrs=decoding_order_corrs,
                )
            )

        return outputs
