"""Tests for BaseModel abstract base class and LocalModel __init__ validation."""

from unittest import mock

import pytest
import torch

from parallelbench.models.base_model import BaseModel, DLLMOutput, LocalModel


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


def test_calls_from_pretrained_with_correct_args():
    mock_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    with (
        mock.patch(
            "parallelbench.models.base_model.AutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained,
        mock.patch(
            "parallelbench.models.base_model.AutoTokenizer.from_pretrained",
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
            "parallelbench.models.base_model.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
        mock.patch(
            "parallelbench.models.base_model.AutoTokenizer.from_pretrained",
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


# ============================================================
# Batch generation API
# ============================================================


def test_supports_batch_returns_false_by_default():
    """BaseModel.supports_batch should return False by default."""

    class ConcreteModel(BaseModel):
        def generate(self, messages, **kwargs):
            return DLLMOutput(output="test")

    model = ConcreteModel()
    assert model.supports_batch is False


def test_generate_batch_raises_not_implemented_with_class_name():
    """BaseModel.generate_batch() should raise NotImplementedError with the class name."""

    class MyCustomModel(BaseModel):
        def generate(self, messages, **kwargs):
            return DLLMOutput(output="test")

    model = MyCustomModel()
    with pytest.raises(NotImplementedError, match="MyCustomModel"):
        model.generate_batch(
            messages_list=[[{"role": "user", "content": "hi"}]],
            gen_config={"steps": 32},
        )


def test_subclass_with_batch_support():
    """A subclass that implements generate_batch() and supports_batch=True should work."""

    class BatchModel(BaseModel):
        @property
        def supports_batch(self) -> bool:
            return True

        def generate(self, messages, **kwargs):
            return DLLMOutput(output="single")

        def generate_batch(
            self,
            messages_list,
            gen_config=None,
            output_prefix_list=None,
            output_history=False,
        ):
            return [DLLMOutput(output=f"batch_{i}") for i in range(len(messages_list))]

    model = BatchModel()
    assert model.supports_batch is True

    results = model.generate_batch(
        messages_list=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        gen_config={"steps": 32},
    )
    assert len(results) == 2
    assert results[0].output == "batch_0"
    assert results[1].output == "batch_1"
