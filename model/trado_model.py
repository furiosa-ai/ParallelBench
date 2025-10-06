
from dataclasses import dataclass
from enum import Enum
import functools
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

from dataset.parallel_bench.data.task import PARALLEL_BENCH_MASK_TOKEN
from model.base_model import BaseModel, DLLMOutput
from model.model_utils import decode_history, compute_decoding_order_correlation_from_history
from utils.perf_utils import measure_time_mem
from utils.utils import insert_import_path

from transformers.cache_utils import DynamicCache


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0 and temperature > 0.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    if temperature == 0.0:
        token = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
    model,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    # denoising_steps=8,
    steps=0,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.85,
    stopping_criteria_idx=None,
    output_history=False,
):
    nfe = 0
    history = [] if output_history else None
    input_length = prompt.shape[1]

    model.eval()
    input_ids = prompt  # ['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    decode_blocks = num_blocks - prefill_blocks
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        nfe += 1

    if steps is None:
        # threshold, set steps to max to give threshold time
        steps = decode_blocks * block_length

    assert steps % decode_blocks == 0, f"Steps must be divisible by number of decode blocks. Got steps={steps}, decode_blocks={decode_blocks}"
    denoising_steps = steps // decode_blocks
    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                nfe += 1
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits
            nfe += 1

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        # number of high confidence tokens is larger or equal to step schedule
                        transfer_index[j] = high_conf_mask
                    else:
                        # insufficient high confidence tokens, just do low_confidence_static
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

            x[:, num_block*block_length:(num_block+1)*block_length] = cur_x

            if history is not None:
                history.append(x[:, input_length:].cpu().clone())  # clone

        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x, nfe, history


class TradoRemaskingStrategy(str, Enum):
    LOW_CONFIDENCE = 'low_confidence'
    LOW_CONFIDENCE_THRESHOLD = 'low_confidence_threshold'


@dataclass
class TradoGenerationConfig:
    accel_framework: Optional[str] = None

    max_tokens: int = 128
    steps: Optional[int] = 128
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    # alg_temp: float = 0.0
    remasking: str = "low_confidence"
    block_length: int = 128

    # fast-dllm specific
    fast_dllm_threshold: float = 0.9  # TODO assert none if not using LOW_CONFIDENCE_THRESHOLD
    # fast_dllm_factor: Optional[float] = None
    # fast_dllm_use_cache: bool = False
    # fast_dllm_dual_cache: bool = False

    # remdm_steps: Optional[int] = None
    # remdm_number: Optional[int] = None

    @property
    def num_blocks(self):
        return self.max_tokens // self.block_length

    def __post_init__(self):
        assert self.steps is None or self.steps <= self.max_tokens, f"Steps must be less than or equal to max tokens. Got steps={self.steps}, max_tokens={self.max_tokens}"
        assert self.max_tokens % self.block_length == 0, f"Max tokens must be divisible by block length. Got max_tokens={self.max_tokens}, block_length={self.block_length}"
        assert self.steps is None or (self.steps % self.num_blocks == 0), f"Steps must be divisible by number of blocks. Got steps={self.steps}, num_blocks={self.num_blocks}"
        assert self.remasking in list(TradoRemaskingStrategy), f"Remasking must be one of {list(TradoRemaskingStrategy)}, got {self.remasking}"
        # assert not (self.accel_framework != "fast_dllm" and self.fast_dllm_use_cache)
        assert self.accel_framework is None

    def is_mdpo_rcr(self):
        return False
    
    def is_remdm(self):
        return False

    def to_generate_kwargs(self):
        gen_kwargs = dict(
            steps=self.steps, 
            gen_length=self.max_tokens,
            block_length=self.block_length, 
            temperature=self.temperature,
            top_p=self.top_p if self.top_p is not None else 1.0,
            top_k=self.top_k if self.top_k is not None else 0,
            # alg_temp=self.alg_temp,
        )

        if self.remasking == TradoRemaskingStrategy.LOW_CONFIDENCE:
            gen_kwargs["remasking_strategy"] = "low_confidence_static"
        elif self.remasking == TradoRemaskingStrategy.LOW_CONFIDENCE_THRESHOLD:
            gen_kwargs["remasking_strategy"] = "low_confidence_dynamic"
            gen_kwargs["confidence_threshold"] = self.fast_dllm_threshold
        else:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")

        return gen_kwargs


TRADO_MASK_TOKEN_ID = 151669


class TradoModel(BaseModel):
    def __init__(self, model_name, accel_framework=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False, dynamic=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.mask_token_id = TRADO_MASK_TOKEN_ID

        self.mask_id = TRADO_MASK_TOKEN_ID
        self.accel_framework = accel_framework

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def model_generate(self, input_ids, gen_config, output_history=False, output0_ids=None):
        gen_kwargs = gen_config.to_generate_kwargs()

        if gen_config.is_mdpo_rcr():
            assert False
            # assert self.accel_framework is None, "MDPO-RCR is not supported with fast-dllm"

            # from model.remasking.llada.mdpo_rcr import generate_dlm as generate
            # generate_fn = generate
        elif gen_config.is_remdm():
            assert False
            assert self.accel_framework is None, "ReMDM is not supported with fast-dllm"

            from model.remasking.llada.remdm import llada_remdm_sample as generate
            generate_fn = functools.partial(generate, tokenizer=self.tokenizer)
        elif self.accel_framework == "fast_dllm":
            assert False
            if gen_config.fast_dllm_use_cache:
                if gen_config.fast_dllm_dual_cache:
                    generate_fn = generate_with_dual_cache
                else:
                    generate_fn = generate_with_prefix_cache
            else:
                assert False
        elif self.accel_framework is None:
            generate_fn = block_diffusion_generate

        if output0_ids is not None:
            return generate_fn(self.model, input_ids, mask_id=self.mask_id, **gen_kwargs, output_history=output_history, output0_ids=output0_ids)
        else:
            return generate_fn(self.model, input_ids, mask_id=self.mask_id, **gen_kwargs, output_history=output_history)

    def generate(self, messages, output_prefix=None, gen_config=None, output_history=False):
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = messages
        
        gen_config = TradoGenerationConfig(accel_framework=self.accel_framework, **gen_config)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        block_length = gen_config.block_length
        assert block_length == 4 or self.model.name_or_path.endswith(f"-b{block_length}") or "TraDo" in self.model.name_or_path, f"Block length {block_length} is not supported by the model {self.model.name_or_path}."
        # pad input_ids to be multiple of block_length
        if input_ids.shape[1] % block_length != 0:
            pad_length = block_length - (input_ids.shape[1] % block_length)
            input_ids = F.pad(input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)

        if output_prefix is not None:
            output_prefix = output_prefix.replace(PARALLEL_BENCH_MASK_TOKEN, self.tokenizer.mask_token)
            output0_ids = self.tokenizer(output_prefix, return_tensors="pt", padding="max_length", max_length=gen_config.max_tokens).input_ids.to(self.model.device)
            assert output0_ids.shape[1] == gen_config.max_tokens, "output_prefix is too long"
        else:
            output0_ids = None

        input_output_ids, nfe, history = self.model_generate(input_ids, gen_config, output_history=output_history, output0_ids=output0_ids)
        output_ids = input_output_ids[:, input_ids.shape[1]:]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        assert not (output_history and history is None), "History should not be None if output_history is True."

        decoding_order, decoding_order_corrs = compute_decoding_order_correlation_from_history(self.tokenizer, history)

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
