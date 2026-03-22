"""Tests for batched generation in generate.py.

Validates the [prompt | mask | pad] layout, attention_mask construction,
and per-sample block boundary handling used by generate_batch().
"""

import torch

from parallelbench.models.local.generate import generate_batch


class FakeModel(torch.nn.Module):
    """Minimal model that records inputs and returns uniform logits.

    On each forward call, unmasks all mask tokens to token id 1 by returning
    logits that heavily favor token id 1 over all others.
    """

    def __init__(self, vocab_size=100, mask_id=99):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.call_log = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None):
        self.call_log.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone()
                if attention_mask is not None
                else None,
            }
        )
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        # Strongly favor token 1 so all masks get filled in one step
        logits[:, :, 1] = 100.0
        # Prevent mask token from being sampled
        logits[:, :, self.mask_id] = -float("inf")

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


MASK_ID = 99
PAD_ID = 0


class TestGenerateBatchLayout:
    """Verify [prompt | mask | pad] layout construction."""

    def test_layout_uniform_prompt_lengths(self):
        """When all prompts have the same length, no padding should be added."""
        prompts = [
            torch.tensor([10, 20, 30]),
            torch.tensor([40, 50, 60]),
        ]
        model = FakeModel(mask_id=MASK_ID)
        x, nfe, history = generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=4,
            block_length=4,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        # Shape: (2, 3 + 4) = (2, 7), no padding needed
        assert x.shape == (2, 7)
        # All mask tokens should be filled (no mask_id remaining)
        assert (x[:, 3:] != MASK_ID).all()

    def test_layout_variable_prompt_lengths(self):
        """Different prompt lengths should produce correct right-padding."""
        prompts = [
            torch.tensor([10, 20]),  # short prompt
            torch.tensor([30, 40, 50, 60]),  # long prompt
        ]
        model = FakeModel(mask_id=MASK_ID)

        # Capture the initial state by checking the first forward call
        x, nfe, _ = generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=2,
            block_length=2,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        # max_prompt_len=4, gen_length=2 -> max_total=6
        assert x.shape == (2, 6)

        # Check initial layout from first forward call
        first_call = model.call_log[0]["input_ids"]
        # Sample 0: [10, 20, MASK, MASK, PAD, PAD]
        assert first_call[0, 0].item() == 10
        assert first_call[0, 1].item() == 20
        assert first_call[0, 2].item() == MASK_ID
        assert first_call[0, 3].item() == MASK_ID
        assert first_call[0, 4].item() == PAD_ID
        assert first_call[0, 5].item() == PAD_ID
        # Sample 1: [30, 40, 50, 60, MASK, MASK]
        assert first_call[1, 0].item() == 30
        assert first_call[1, 3].item() == 60
        assert first_call[1, 4].item() == MASK_ID
        assert first_call[1, 5].item() == MASK_ID


class TestGenerateBatchAttentionMask:
    """Verify attention_mask is correct: 1 for prompt+mask, 0 for pad."""

    def test_attention_mask_uniform(self):
        """No padding -> attention_mask should be all 1s."""
        prompts = [torch.tensor([10, 20]), torch.tensor([30, 40])]
        model = FakeModel(mask_id=MASK_ID)
        generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=2,
            block_length=2,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        attn_mask = model.call_log[0]["attention_mask"]
        assert (attn_mask == 1.0).all()

    def test_attention_mask_variable(self):
        """Padding positions should have attention_mask=0."""
        prompts = [
            torch.tensor([10]),  # prompt_len=1
            torch.tensor([20, 30, 40]),  # prompt_len=3
        ]
        model = FakeModel(mask_id=MASK_ID)
        generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=2,
            block_length=2,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        attn_mask = model.call_log[0]["attention_mask"]
        # max_total = 3 + 2 = 5
        # Sample 0: [prompt(1), mask(2), pad(2)] -> [1, 1, 1, 0, 0]
        assert attn_mask[0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
        # Sample 1: [prompt(3), mask(2)] -> [1, 1, 1, 1, 1]
        assert attn_mask[1].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


class TestGenerateBatchBlockBoundary:
    """Verify per-sample block boundaries with different prompt lengths."""

    def test_multi_block_with_variable_prompts(self):
        """Two blocks with different prompt lengths should fill all masks."""
        prompts = [
            torch.tensor([10, 20]),
            torch.tensor([30, 40, 50]),
        ]
        model = FakeModel(mask_id=MASK_ID)
        x, nfe, _ = generate_batch(
            model=model,
            prompts=prompts,
            steps=2,  # 2 steps total, 1 per block
            gen_length=4,
            block_length=2,  # 2 blocks of 2
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        # max_total = 3 + 4 = 7
        assert x.shape == (2, 7)
        # Sample 0: prompt=[10,20], gen starts at 2, pad at 6
        # All mask positions should be filled
        assert (x[0, 2:6] != MASK_ID).all()
        assert x[0, 6].item() == PAD_ID  # trailing pad
        # Sample 1: prompt=[30,40,50], gen starts at 3
        assert (x[1, 3:7] != MASK_ID).all()

    def test_nfe_is_shared_across_batch(self):
        """NFE count should reflect total forward passes (shared, not per-sample)."""
        prompts = [torch.tensor([10]), torch.tensor([20])]
        model = FakeModel(mask_id=MASK_ID)
        _, nfe, _ = generate_batch(
            model=model,
            prompts=prompts,
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        # FakeModel fills all masks in 1 step (strong logit for token 1),
        # but num_transfer_tokens distributes across 4 steps
        assert nfe >= 1
        assert nfe == len(model.call_log)


class TestGenerateBatchHistory:
    """Verify per-sample history tracking."""

    def test_history_none_when_disabled(self):
        prompts = [torch.tensor([10])]
        model = FakeModel(mask_id=MASK_ID)
        _, _, history = generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=2,
            block_length=2,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            output_history=False,
            temperature=0.0,
            alg_temp=0.0,
        )
        assert history is None

    def test_history_per_sample_when_enabled(self):
        prompts = [torch.tensor([10]), torch.tensor([20, 30])]
        model = FakeModel(mask_id=MASK_ID)
        _, nfe, history = generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=2,
            block_length=2,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            output_history=True,
            temperature=0.0,
            alg_temp=0.0,
        )
        assert history is not None
        assert len(history) == 2  # one list per sample
        # Each sample should have nfe history entries
        assert len(history[0]) == nfe
        assert len(history[1]) == nfe
        # Each entry is a (1, gen_length) tensor matching decode_history contract
        assert history[0][0].shape == (1, 2)
        assert history[1][0].shape == (1, 2)


class TestGenerateBatchEdgeCases:
    """Edge cases for generate_batch."""

    def test_single_sample_batch(self):
        """Batch of 1 should produce the same result shape as single generation."""
        prompts = [torch.tensor([10, 20, 30])]
        model = FakeModel(mask_id=MASK_ID)
        x, nfe, _ = generate_batch(
            model=model,
            prompts=prompts,
            steps=2,
            gen_length=4,
            block_length=4,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        assert x.shape == (1, 7)
        assert (x[0, 3:] != MASK_ID).all()

    def test_2d_prompt_input(self):
        """Prompts with shape (1, L) should be squeezed automatically."""
        prompts = [
            torch.tensor([[10, 20]]),  # 2D input
            torch.tensor([30, 40]),  # 1D input
        ]
        model = FakeModel(mask_id=MASK_ID)
        x, _, _ = generate_batch(
            model=model,
            prompts=prompts,
            steps=1,
            gen_length=2,
            block_length=2,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            alg_temp=0.0,
        )
        assert x.shape == (2, 4)
