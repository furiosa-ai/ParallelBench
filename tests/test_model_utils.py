"""Tests for model/model_utils.py utility functions."""

from unittest import mock

import numpy as np
import pytest
import torch

from parallelbench.models.model_utils import (
    compute_decoding_order_correlation,
    compute_decoding_order_correlation_from_history,
    compute_history_decoding_order,
    decode_history,
)


def _make_tokenizer(mask_token_id=126336, pad_token_id=0):
    """Create a mock tokenizer with the required attributes."""
    tokenizer = mock.MagicMock()
    tokenizer.mask_token_id = mask_token_id
    tokenizer.pad_token_id = pad_token_id
    tokenizer.decode = mock.MagicMock(
        side_effect=lambda ids, **kw: f"decoded:{ids.tolist()}"
    )
    tokenizer.encode = mock.MagicMock(
        side_effect=lambda text, **kw: torch.tensor(
            [[int(c) for c in text.split(",") if c.strip()]]
        )
    )
    return tokenizer


# -- decode_history --


def test_decode_history_none_returns_none():
    tokenizer = _make_tokenizer()
    assert decode_history(tokenizer, None) is None


def test_decode_history_slices_by_input_length():
    tokenizer = _make_tokenizer()
    # Each history entry: shape (1, 6). input_length=2 -> decode indices [2:]
    history = [
        torch.tensor([[10, 20, 30, 40, 50, 60]]),
        torch.tensor([[10, 20, 31, 41, 51, 61]]),
    ]
    result = decode_history(tokenizer, history, input_length=2)
    assert len(result) == 2
    # Check that decode was called with sliced tensors (indices 2 onwards)
    first_call_arg = tokenizer.decode.call_args_list[0][0][0]
    assert first_call_arg.tolist() == [30, 40, 50, 60]


# -- compute_history_decoding_order --


def test_compute_history_decoding_order_sequential_unmask():
    """Tokens unmasked one at a time should give sequential order."""
    mask_id = 99
    tokenizer = _make_tokenizer(mask_token_id=mask_id)

    # 3 steps, 3 tokens. Step 0: unmask pos 0, Step 1: unmask pos 1, Step 2: unmask pos 2
    history = [
        torch.tensor([1, mask_id, mask_id]),
        torch.tensor([1, 2, mask_id]),
        torch.tensor([1, 2, 3]),
    ]
    history_tensor = torch.stack(history)

    order = compute_history_decoding_order(tokenizer, history_tensor)
    assert order.tolist() == [0, 1, 2]


def test_compute_history_decoding_order_simultaneous_unmask():
    """Tokens unmasked at same step should have the same order value."""
    mask_id = 99
    tokenizer = _make_tokenizer(mask_token_id=mask_id)

    # Step 0: unmask pos 0 and pos 1 simultaneously
    history = [
        torch.tensor([1, 2, mask_id]),
        torch.tensor([1, 2, 3]),
    ]
    history_tensor = torch.stack(history)

    order = compute_history_decoding_order(tokenizer, history_tensor)
    # pos 0 and pos 1 unmasked at step 0
    assert order[0].item() == order[1].item() == 0
    assert order[2].item() == 1


def test_compute_history_decoding_order_ignore_pad():
    """ignore_pad=True should exclude pad positions from the result."""
    mask_id = 99
    pad_id = 0
    tokenizer = _make_tokenizer(mask_token_id=mask_id, pad_token_id=pad_id)

    # Final output has pad at position 2
    history = [
        torch.tensor([1, mask_id, pad_id]),
        torch.tensor([1, 2, pad_id]),
    ]
    history_tensor = torch.stack(history)

    order = compute_history_decoding_order(tokenizer, history_tensor, ignore_pad=True)
    # Only 2 non-pad tokens
    assert order.shape[0] == 2


def test_compute_history_decoding_order_string_history():
    """String history should be encoded first via tokenizer."""
    mask_id = 99
    tokenizer = _make_tokenizer(mask_token_id=mask_id)
    # Override encode to return specific tensors
    tokenizer.encode = mock.MagicMock(
        side_effect=[
            torch.tensor([[1, mask_id, mask_id]]),
            torch.tensor([[1, 2, mask_id]]),
            torch.tensor([[1, 2, 3]]),
        ]
    )

    history = ["step0", "step1", "step2"]
    order = compute_history_decoding_order(tokenizer, history)
    assert tokenizer.encode.call_count == 3
    assert order.tolist() == [0, 1, 2]


# -- compute_decoding_order_correlation --


def test_compute_decoding_order_correlation_sorted_input():
    """Sorted input should give kendall/spearman == 1.0."""
    order = np.array([0, 1, 2, 3, 4])
    result = compute_decoding_order_correlation(order)
    assert result["dec_order_kendall"] == pytest.approx(1.0)
    assert result["dec_order_spearman"] == pytest.approx(1.0)


def test_compute_decoding_order_correlation_reversed_input():
    """Reversed input should give kendall/spearman == -1.0."""
    order = np.array([4, 3, 2, 1, 0])
    result = compute_decoding_order_correlation(order)
    assert result["dec_order_kendall"] == pytest.approx(-1.0)
    assert result["dec_order_spearman"] == pytest.approx(-1.0)


def test_compute_decoding_order_correlation_returns_correct_keys():
    order = np.array([0, 2, 1, 3])
    result = compute_decoding_order_correlation(order)
    assert "dec_order_kendall" in result
    assert "dec_order_spearman" in result
    assert len(result) == 2


def test_compute_decoding_order_correlation_accepts_tensor():
    """Should accept torch.Tensor input (converted to numpy internally)."""
    order = torch.tensor([0, 1, 2, 3])
    result = compute_decoding_order_correlation(order)
    assert result["dec_order_kendall"] == pytest.approx(1.0)


# -- compute_decoding_order_correlation_from_history --


def test_compute_decoding_order_correlation_from_history_returns_tuple():
    mask_id = 99
    pad_id = 0
    tokenizer = _make_tokenizer(mask_token_id=mask_id, pad_token_id=pad_id)

    history = [
        torch.tensor([[1, mask_id, mask_id]]),
        torch.tensor([[1, 2, mask_id]]),
        torch.tensor([[1, 2, 3]]),
    ]

    order_list, corr_dict = compute_decoding_order_correlation_from_history(
        tokenizer, history
    )
    assert isinstance(order_list, list)
    assert isinstance(corr_dict, dict)


def test_compute_decoding_order_correlation_from_history_has_ignore_pad_keys():
    mask_id = 99
    pad_id = 0
    tokenizer = _make_tokenizer(mask_token_id=mask_id, pad_token_id=pad_id)

    history = [
        torch.tensor([[1, mask_id, mask_id]]),
        torch.tensor([[1, 2, mask_id]]),
        torch.tensor([[1, 2, 3]]),
    ]

    _, corr_dict = compute_decoding_order_correlation_from_history(tokenizer, history)
    assert "dec_order_kendall" in corr_dict
    assert "dec_order_spearman" in corr_dict
    assert "dec_order_kendall_ignore_pad" in corr_dict
    assert "dec_order_spearman_ignore_pad" in corr_dict
