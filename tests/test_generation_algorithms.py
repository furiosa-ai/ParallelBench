"""Tests for generation algorithm pure functions."""

import numpy as np
import pytest
import torch

from model.local.generate import get_num_transfer_tokens
from model.local.generate_rcr import gamma_func, get_num_transfer_tokens_maskgit


# -- get_num_transfer_tokens --


def test_get_num_transfer_tokens_steps_none_returns_none():
    mask_index = torch.tensor([[True, True, False, True]])
    assert get_num_transfer_tokens(mask_index, steps=None) is None


def test_get_num_transfer_tokens_even_division():
    # 10 masks, 5 steps -> [2, 2, 2, 2, 2]
    mask_index = torch.tensor([[True] * 10])
    result = get_num_transfer_tokens(mask_index, steps=5)
    assert result.shape == (1, 5)
    assert result.sum().item() == 10
    assert (result == 2).all()


def test_get_num_transfer_tokens_with_remainder():
    # 10 masks, 3 steps -> remainder 1 goes to first step: [4, 3, 3]
    mask_index = torch.tensor([[True] * 10])
    result = get_num_transfer_tokens(mask_index, steps=3)
    assert result.shape == (1, 3)
    assert result.sum().item() == 10
    assert result[0, 0].item() == 4
    assert result[0, 1].item() == 3
    assert result[0, 2].item() == 3


def test_get_num_transfer_tokens_single_step():
    mask_index = torch.tensor([[True] * 7])
    result = get_num_transfer_tokens(mask_index, steps=1)
    assert result.shape == (1, 1)
    assert result[0, 0].item() == 7


def test_get_num_transfer_tokens_no_masks():
    mask_index = torch.tensor([[False, False, False]])
    result = get_num_transfer_tokens(mask_index, steps=3)
    assert result.shape == (1, 3)
    assert (result == 0).all()


def test_get_num_transfer_tokens_batch_size_2():
    mask_index = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, False, False],
    ])
    result = get_num_transfer_tokens(mask_index, steps=2)
    assert result.shape == (2, 2)
    assert result[0].sum().item() == 4
    assert result[1].sum().item() == 2


def test_get_num_transfer_tokens_steps_equal_mask_count():
    # 5 masks, 5 steps -> all ones
    mask_index = torch.tensor([[True] * 5])
    result = get_num_transfer_tokens(mask_index, steps=5)
    assert (result == 1).all()


# -- gamma_func --


def test_gamma_func_linear():
    assert gamma_func(0.5, "linear") == pytest.approx(0.5)


def test_gamma_func_cosine_at_zero():
    assert gamma_func(0.0, "cosine") == pytest.approx(1.0)


def test_gamma_func_pow2():
    # 1 - 0.5^2 = 0.75
    assert gamma_func(0.5, "pow2") == pytest.approx(0.75)


def test_gamma_func_log():
    result = gamma_func(0.5, "log", total_num=512)
    assert 0.0 < result <= 1.0


def test_gamma_func_unknown_mode_raises():
    with pytest.raises(NotImplementedError):
        gamma_func(0.5, "unknown_mode")


def test_gamma_func_result_clipped():
    # Result should always be in [1e-6, 1.0]
    for mode in ["linear", "cosine", "pow2"]:
        for r in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = gamma_func(r, mode)
            assert 1e-6 <= result <= 1.0, f"gamma_func({r}, {mode}) = {result} out of range"


# -- get_num_transfer_tokens_maskgit --


def test_get_num_transfer_tokens_maskgit_basic():
    mask_index = torch.tensor([[True] * 10])
    result = get_num_transfer_tokens_maskgit(mask_index, steps=5, mode="linear")
    assert result.shape == (1, 5)
    assert result.dtype == torch.int64
    assert result.sum().item() == 10


def test_get_num_transfer_tokens_maskgit_sum_equals_total_masks():
    mask_index = torch.tensor([[True, True, False, True, True, True]])
    total_masks = mask_index.sum().item()
    result = get_num_transfer_tokens_maskgit(mask_index, steps=3, mode="cosine")
    assert result.sum().item() == total_masks
