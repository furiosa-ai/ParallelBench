"""Tests for LocalModel __init__ validation in model/base_model.py."""

from unittest import mock

import pytest
import torch

from model.base_model import LocalModel, VALID_ACCEL_FRAMEWORKS


def _create_local_model(accel_framework=None, model_name="test-model"):
    """Create a LocalModel with mocked from_pretrained calls."""
    mock_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    with mock.patch("model.base_model.AutoModel.from_pretrained", return_value=mock_model), \
         mock.patch("model.base_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):

        # LocalModel is abstract (has abstract `generate`), so create a concrete subclass
        class ConcreteLocalModel(LocalModel):
            def generate(self, messages, gen_config=None, output_prefix=None, output_history=False):
                pass

        instance = ConcreteLocalModel(model_name, accel_framework=accel_framework)
    return instance, mock_model, mock_tokenizer


def test_invalid_accel_framework_raises():
    with pytest.raises(ValueError, match="Invalid accel_framework"):
        _create_local_model(accel_framework="invalid")


def test_non_fast_dllm_framework_sets_is_fast_dllm_false():
    instance, _, _ = _create_local_model(accel_framework=None)
    assert instance.is_fast_dllm is False


def test_valid_accel_framework_fast_dllm():
    instance, _, _ = _create_local_model(accel_framework="fast_dllm")
    assert instance.is_fast_dllm is True


def test_calls_from_pretrained_with_correct_args():
    mock_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    with mock.patch("model.base_model.AutoModel.from_pretrained", return_value=mock_model) as mock_from_pretrained, \
         mock.patch("model.base_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):

        class ConcreteLocalModel(LocalModel):
            def generate(self, messages, gen_config=None, output_prefix=None, output_history=False):
                pass

        ConcreteLocalModel("my-model-path")

    mock_from_pretrained.assert_called_once_with(
        "my-model-path",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def test_calls_model_eval():
    mock_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    with mock.patch("model.base_model.AutoModel.from_pretrained", return_value=mock_model), \
         mock.patch("model.base_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):

        class ConcreteLocalModel(LocalModel):
            def generate(self, messages, gen_config=None, output_prefix=None, output_history=False):
                pass

        ConcreteLocalModel("my-model-path")

    mock_model.eval.assert_called_once()
