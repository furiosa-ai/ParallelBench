"""Tests for generation algorithm pure functions."""

import torch

from parallelbench.models.local.generate import get_num_transfer_tokens


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
    mask_index = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    result = get_num_transfer_tokens(mask_index, steps=2)
    assert result.shape == (2, 2)
    assert result[0].sum().item() == 4
    assert result[1].sum().item() == 2


def test_get_num_transfer_tokens_steps_equal_mask_count():
    # 5 masks, 5 steps -> all ones
    mask_index = torch.tensor([[True] * 5])
    result = get_num_transfer_tokens(mask_index, steps=5)
    assert (result == 1).all()
