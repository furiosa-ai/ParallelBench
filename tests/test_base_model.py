"""Tests for BaseModel abstract base class and LocalModel __init__ validation."""

from unittest import mock

import pytest
import torch

from parallelbench.model.base_model import BaseModel, DLLMOutput, LocalModel


def test_subclass_without_generate_cannot_be_instantiated():
    """A subclass that doesn't implement generate() should fail."""

    class IncompleteModel(BaseModel):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteModel()


def test_subclass_with_generate_can_be_instantiated():
    """A subclass that implements generate() should succeed."""

    class ConcreteModel(BaseModel):
        def generate(self, messages, **kwargs):
            return DLLMOutput(output="test")

    model = ConcreteModel()
    assert model is not None
    result = model.generate([{"role": "user", "content": "hi"}])
    assert isinstance(result, DLLMOutput)
    assert result.output == "test"


def test_dllm_output_properties():
    """DLLMOutput dataclass should work with input_length and output_length properties."""
    # Test with input_ids
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    output = DLLMOutput(output="test", input_ids=input_ids)
    assert output.input_length == 5

    # Test with output_ids (no padding)
    output_ids = torch.tensor([[6, 7, 8]])
    output = DLLMOutput(output="test", output_ids=output_ids)
    assert output.output_length == 3

    # Test with output_ids and pad_token_id
    output_ids = torch.tensor([[6, 7, 8, 0, 0]])  # 0 is pad token
    output = DLLMOutput(output="test", output_ids=output_ids, pad_token_id=0)
    assert output.output_length == 3

    # Test None cases
    output = DLLMOutput(output="test")
    assert output.input_length is None
    assert output.output_length is None


# ============================================================
# LocalModel __init__ validation
# ============================================================


def _create_local_model(accel_framework=None, model_name="test-model"):
    """Create a LocalModel with mocked from_pretrained calls."""
    mock_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    with (
        mock.patch(
            "parallelbench.model.base_model.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
        mock.patch(
            "parallelbench.model.base_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):

        class ConcreteLocalModel(LocalModel):
            def generate(
                self,
                messages,
                gen_config=None,
                output_prefix=None,
                output_history=False,
            ):
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

    with (
        mock.patch(
            "parallelbench.model.base_model.AutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained,
        mock.patch(
            "parallelbench.model.base_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):

        class ConcreteLocalModel(LocalModel):
            def generate(
                self,
                messages,
                gen_config=None,
                output_prefix=None,
                output_history=False,
            ):
                pass

        ConcreteLocalModel("my-model-path")

    mock_from_pretrained.assert_called_once_with(
        "my-model-path",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )


def test_calls_model_eval():
    mock_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    with (
        mock.patch(
            "parallelbench.model.base_model.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
        mock.patch(
            "parallelbench.model.base_model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):

        class ConcreteLocalModel(LocalModel):
            def generate(
                self,
                messages,
                gen_config=None,
                output_prefix=None,
                output_history=False,
            ):
                pass

        ConcreteLocalModel("my-model-path")

    mock_model.eval.assert_called_once()
