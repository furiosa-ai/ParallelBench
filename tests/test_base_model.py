"""Tests for BaseModel abstract base class."""

import pytest
import torch

from parallelbench.model.base_model import BaseModel, DLLMOutput


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
