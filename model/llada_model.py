
from dataclasses import dataclass
from enum import Enum
import functools
import numpy as np
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

from dataset.parallel_bench.data.task import PARALLEL_BENCH_MASK_TOKEN
from model.base_model import BaseModel, DLLMOutput
from model.model_utils import decode_history, compute_decoding_order_correlation_from_history
from utils.perf_utils import measure_time_mem
from utils.utils import insert_import_path


FAST_DLLM_PATH = "src/Fast_dLLM/llada"


# def add_gumbel_noise(logits, temperature):
#     '''
#     The Gumbel max is a method for sampling categorical distributions.
#     According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#     Thus, we use float64.
#     '''
#     if temperature == 0:
#         return logits
#     logits = logits.to(torch.float64)
#     noise = torch.rand_like(logits, dtype=torch.float64)
#     gumbel_noise = (- torch.log(noise)) ** temperature
#     return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens


# @ torch.no_grad()
# def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336, output_history=False):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     history = [] if output_history else None

#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     prompt_index = (x != mask_id)

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps = steps // num_blocks

#     input_length = prompt.shape[1]

#     nfe = 0
#     for num_block in range(num_blocks):
#         block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        
#         # compute the number of tokens to unmask at each step
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         for i in range(steps):
#             # all masked tokens
#             mask_index = (x == mask_id)
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 logits = model(x_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits
#             nfe += 1

#             # sample from logits
#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            
#             # selected ids
#             x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

#             # normalize logits
#             p = F.softmax(logits, dim=-1)

#             if remasking == 'low_confidence':
#                 # get probabilities of selected ids (confidence)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'topk_margin':
#                 sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
#                 # Extract top1 and top2 probabilities
#                 top1_probs = sorted_probs[..., 0] 
#                 top2_probs = sorted_probs[..., 1] 
#                 # Calculate confidence as top1 - top2
#                 x0_p = top1_probs - top2_probs
#             elif remasking == 'entropy':
#                 epsilon = 1e-10
#                 log_probs = torch.log(p + epsilon)
#                 x0_p = torch.sum(p * log_probs, dim=-1)
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             # mask future blocks
#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

#             # restore already unmasked tokens from current x
#             x0 = torch.where(mask_index, x0, x)
#             # set confidence of already unmasked tokens to -inf
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             # select tokens to transfer from x0 to x based on highest confidence
#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):  # per batch
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]

#             if history is not None:
#                 history.append(x[:, input_length:].cpu().clone())  # clone

#     return x, nfe, history



class LladaRemaskingStrategy(str, Enum):
    LOW_CONFIDENCE = 'low_confidence'
    RANDOM = 'random'
    TOPK_MARGIN = 'topk_margin'
    ENTROPY = 'entropy'

    # fast-dllm specific
    LOW_CONFIDENCE_THRESHOLD = 'low_confidence_threshold'
    RANDOM_THRESHOLD = 'random_threshold'
    TOPK_MARGIN_THRESHOLD = 'topk_margin_threshold'
    ENTROPY_THRESHOLD = 'entropy_threshold'

    LOW_CONFIDENCE_FACTOR = 'low_confidence_factor'
    RANDOM_FACTOR = 'random_factor'
    TOPK_MARGIN_FACTOR = 'topk_margin_factor'
    ENTROPY_FACTOR = 'entropy_factor'

    LOW_CONFIDENCE_RCR = 'low_confidence_rcr'

    LOW_CONFIDENCE_REMDM = 'low_confidence_remdm'


@dataclass
class LladaGenerationConfig:
    accel_framework: Optional[str] = None

    max_tokens: int = 128
    steps: Optional[int] = 128
    temperature: float = 0.0
    alg_temp: float = 0.0
    remasking: str = "low_confidence"
    block_length: int = 128

    # fast-dllm specific
    fast_dllm_threshold: float = 0.9  # TODO assert none if not using LOW_CONFIDENCE_THRESHOLD
    fast_dllm_factor: Optional[float] = None
    fast_dllm_use_cache: bool = False
    fast_dllm_dual_cache: bool = False

    remdm_steps: Optional[int] = None
    remdm_number: Optional[int] = None

    @property
    def num_blocks(self):
        return self.max_tokens // self.block_length

    def __post_init__(self):
        assert self.steps is None or self.steps <= self.max_tokens, f"Steps must be less than or equal to max tokens. Got steps={self.steps}, max_tokens={self.max_tokens}"
        assert self.max_tokens % self.block_length == 0, f"Max tokens must be divisible by block length. Got max_tokens={self.max_tokens}, block_length={self.block_length}"
        assert self.steps is None or (self.steps % self.num_blocks == 0), f"Steps must be divisible by number of blocks. Got steps={self.steps}, num_blocks={self.num_blocks}"
        assert self.remasking in list(LladaRemaskingStrategy), f"Remasking must be one of {list(LladaRemaskingStrategy)}, got {self.remasking}"
        assert not (self.accel_framework != "fast_dllm" and self.fast_dllm_use_cache)

    def is_mdpo_rcr(self):
        return self.remasking == LladaRemaskingStrategy.LOW_CONFIDENCE_RCR
    
    def is_remdm(self):
        return self.remasking == LladaRemaskingStrategy.LOW_CONFIDENCE_REMDM

    def to_generate_kwargs(self):
        gen_kwargs = dict(
            steps=self.steps, 
            gen_length=self.max_tokens,
            block_length=self.block_length, 
            temperature=self.temperature,
            alg_temp=self.alg_temp,
            remasking=self.remasking
        )

        if self.remasking in [
            LladaRemaskingStrategy.LOW_CONFIDENCE,
            LladaRemaskingStrategy.RANDOM,
            LladaRemaskingStrategy.TOPK_MARGIN,
            LladaRemaskingStrategy.ENTROPY,
        ]:
            gen_kwargs["threshold"] = None
            gen_kwargs["factor"] = None
        elif self.is_mdpo_rcr():
            gen_kwargs["overtime_conf"] = True
            gen_kwargs["remasking"] = LladaRemaskingStrategy.LOW_CONFIDENCE
        elif self.is_remdm():
            gen_kwargs["remdm_number"] = self.remdm_number
            gen_kwargs["remdm_steps"] = self.remdm_steps
            gen_kwargs["remasking"] = LladaRemaskingStrategy.LOW_CONFIDENCE
        elif self.remasking in [
            LladaRemaskingStrategy.LOW_CONFIDENCE_THRESHOLD,
            LladaRemaskingStrategy.RANDOM_THRESHOLD,
            LladaRemaskingStrategy.TOPK_MARGIN_THRESHOLD,
            LladaRemaskingStrategy.ENTROPY_THRESHOLD,
        ]:
            assert self.fast_dllm_threshold is not None, f"fast_dllm_threshold must be provided for {self.remasking} algorithm"
            gen_kwargs["threshold"] = self.fast_dllm_threshold
            gen_kwargs["factor"] = None

            gen_kwargs["remasking"] = {
                LladaRemaskingStrategy.LOW_CONFIDENCE_THRESHOLD: LladaRemaskingStrategy.LOW_CONFIDENCE,
                LladaRemaskingStrategy.RANDOM_THRESHOLD: LladaRemaskingStrategy.RANDOM,
                LladaRemaskingStrategy.TOPK_MARGIN_THRESHOLD: LladaRemaskingStrategy.TOPK_MARGIN,
                LladaRemaskingStrategy.ENTROPY_THRESHOLD: LladaRemaskingStrategy.ENTROPY,
            }[self.remasking]
        elif self.remasking in [
            LladaRemaskingStrategy.LOW_CONFIDENCE_FACTOR,
            LladaRemaskingStrategy.RANDOM_FACTOR,
            LladaRemaskingStrategy.TOPK_MARGIN_FACTOR,
            LladaRemaskingStrategy.ENTROPY_FACTOR,
        ]:
            assert self.fast_dllm_factor is not None, f"fast_dllm_factor must be provided for {self.remasking} algorithm"
            gen_kwargs["threshold"] = None
            gen_kwargs["factor"] = self.fast_dllm_factor

            gen_kwargs["remasking"] = {
                LladaRemaskingStrategy.LOW_CONFIDENCE_FACTOR: LladaRemaskingStrategy.LOW_CONFIDENCE,
                LladaRemaskingStrategy.RANDOM_FACTOR: LladaRemaskingStrategy.RANDOM,
                LladaRemaskingStrategy.TOPK_MARGIN_FACTOR: LladaRemaskingStrategy.TOPK_MARGIN,
                LladaRemaskingStrategy.ENTROPY_FACTOR: LladaRemaskingStrategy.ENTROPY,
            }[self.remasking]
        else:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")

        return gen_kwargs


LLADA_MASK_TOKEN_ID = 126336


class LladaModel(BaseModel):
    def __init__(self, model_name, accel_framework=None):
        if accel_framework == "fast_dllm":
            with insert_import_path(FAST_DLLM_PATH):
                from model.modeling_llada import LLaDAModelLM

            model_class = LLaDAModelLM
        else:
            model_class = AutoModel

        if model_name == "llada-tiny_random":
            self.model = model_class.from_config(
                model_name,)
        else:
            self.model = model_class.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        self.patch_model_forward(self.model, LLADA_MASK_TOKEN_ID)
        
        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False, dynamic=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.mask_token_id = LLADA_MASK_TOKEN_ID

        self.mask_id = LLADA_MASK_TOKEN_ID
        self.accel_framework = accel_framework

    def patch_model_forward(self, model, mask_token_id):
        fwd_fn = model.__class__.forward

        def wrapped_forward(*args, **kwargs):
            output = fwd_fn(*args, **kwargs)
            output.logits[:, :, mask_token_id] = -float('Inf')  # cannot sample mask token
            return output

        model.__class__.forward = wrapped_forward

    def fill(self, prompt, suffix, gen_config=None):
        raise NotImplementedError

    @measure_time_mem("generate")
    def model_generate(self, input_ids, gen_config, output_history=False, output0_ids=None):
        with insert_import_path(FAST_DLLM_PATH):
            from generate import generate as generate_no_cache, generate_with_prefix_cache, generate_with_dual_cache

        gen_kwargs = gen_config.to_generate_kwargs()

        if gen_config.is_mdpo_rcr():
            assert self.accel_framework is None, "MDPO-RCR is not supported with fast-dllm"

            from model.remasking.llada.mdpo_rcr import generate_dlm as generate
            generate_fn = generate
        elif gen_config.is_remdm():
            assert self.accel_framework is None, "ReMDM is not supported with fast-dllm"

            from model.remasking.llada.remdm import llada_remdm_sample as generate
            generate_fn = functools.partial(generate, tokenizer=self.tokenizer)
        elif self.accel_framework == "fast_dllm":
            if gen_config.fast_dllm_use_cache:
                if gen_config.fast_dllm_dual_cache:
                    generate_fn = generate_with_dual_cache
                else:
                    generate_fn = generate_with_prefix_cache
            else:
                assert False
        elif self.accel_framework is None:
            generate_fn = generate_no_cache

        if output0_ids is not None:
            return generate_fn(self.model, input_ids, mask_id=self.mask_id, **gen_kwargs, output_history=output_history, output0_ids=output0_ids)
        else:
            return generate_fn(self.model, input_ids, mask_id=self.mask_id, **gen_kwargs, output_history=output_history)

    def generate(self, messages, output_prefix=None, gen_config=None, output_history=False):
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = messages
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        gen_config = LladaGenerationConfig(accel_framework=self.accel_framework, **gen_config)

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
