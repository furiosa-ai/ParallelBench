import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


def sample_block(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    threshold: Optional[float] = 0.9,
    factor: Optional[float] = None,
    block_length: Optional[int] = 32,
    generation_tokens_hook_func: Optional[callable] = None,
    generation_logits_hook_func: Optional[callable] = None
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    assert not (threshold is not None and factor is not None), "threshold and factor cannot be both set"
    assert generation_tokens_hook_func is None or generation_tokens_hook_func(0, 0, 0) == 0, "generation_tokens_hook_func is not supported in block generation"
    assert generation_logits_hook_func is None or generation_logits_hook_func(0, 0, 0) == 0, "generation_logits_hook_func is not supported in block generation"

    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp

    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    gen_length = max_length - input_ids.shape[1]
    
    # Handle block configuration
    if block_length is None:
        block_length = gen_length  # Default: single block (original behavior)
    
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    
    if steps is not None:
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)
    else:
        steps_per_block = None
        timesteps = None

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    if alg == "confidence_threshold":
        assert threshold is not None, "threshold must be provided for confidence_threshold algorithm"
        alg = "maskgit_plus"
    else:
        pass
        # assert threshold is None, "threshold should not be provided for non-confidence_threshold algorithms"

    # Process each block
    for num_block in range(num_blocks):
        
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
            
        i = 0
        while True:
            mask_index = torch.full((1, x.shape[1]), False, dtype=torch.bool, device=x.device)
            mask_index[:, current_block_start:current_block_end] = (x[:, current_block_start:current_block_end] == mask_token_id)

            model_output = self(x, attention_mask, tok_idx)

            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            mask_index[:, current_block_end:] = False
            if alg == 'origin':
                t = timesteps[i]
                s = timesteps[i + 1]
                mask_logits = logits[mask_index]

                p_transfer = 1 - s / t if i < steps_per_block - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()

                i += 1
            else:
                mask_logits = logits[mask_index]

                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                elif alg == 'random':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k)
                    confidence = torch.rand_like(confidence)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                full_confidence[:, current_block_end:] = -torch.inf

                if factor is not None:
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()

                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum(1)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)

                    for j in range(full_confidence.shape[0]):
                        ns=list(range(1,current_transfer_tokens[j]+1))
                        es=[factor/(n+1) for n in ns]
                        threshs=[1-e for e in es]

                        # at least one token is transferred
                        threshs[0]=-1
                        sorted_confidence=torch.sort(full_confidence[j][mask_index[j]],dim=-1,descending=True)[0]
                        assert len(sorted_confidence)==len(threshs)
                        for top_i in range(len(threshs)):
                            if sorted_confidence[top_i]<threshs[top_i]:
                                break

                        if top_i == 0 or top_i == len(threshs)-1:
                            top_i+=1

                        _, select_index = torch.topk(full_confidence[j], k=top_i)
                        transfer_index[j, select_index] = True
                    
                    x[transfer_index] = x_[transfer_index]
                elif threshold is not None:
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()

                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False

                    x[transfer_index] = x_[transfer_index]
                else:
                    num_mask_token = mask_index.sum() / mask_index.shape[0]

                    t = timesteps[i]
                    s = timesteps[i + 1]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)

                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        
                        x[row_indices,transfer_index] = x_[row_indices,transfer_index]
                    i += 1

            if histories is not None:
                histories.append(x.clone())

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break

    
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x