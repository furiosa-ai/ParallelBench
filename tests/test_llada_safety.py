"""Tests for LLaDA safety fixes: instance-level patching and history=None handling."""

import types
from unittest import mock

import pytest

from model.local.llada.llada_model import LladaModel
from model.model_utils import (
    compute_decoding_order_correlation_from_history,
    decode_history,
)


class MockModelClass:
    """Mock model class for testing instance-level patching isolation."""

    def forward(self, *args, **kwargs):
        """Original class-level forward method."""
        return {"logits": mock.MagicMock()}


class TestPatchModelForwardInstanceLevelPatching:
    """Test instance-level patching isolation for patch_model_forward."""

    def test_patch_model_forward_uses_instance_level_binding(self):
        """Verify patch_model_forward uses instance-level binding with types.MethodType.

        This test ensures:
        1. The patched model.forward exists in instance __dict__ (instance-level)
        2. The original class-level forward is not modified
        3. A second instance is not affected by first instance's patch
        """
        model1 = MockModelClass()
        model2 = MockModelClass()

        original_class_forward = MockModelClass.forward

        llada = object.__new__(LladaModel)
        llada.patch_model_forward(model1, mask_token_id=50257)

        assert "forward" in model1.__dict__
        assert isinstance(model1.__dict__["forward"], types.MethodType), (
            "forward should be bound method"
        )

        assert MockModelClass.forward is original_class_forward

        assert "forward" not in model2.__dict__
        assert MockModelClass.forward is original_class_forward

    def test_patch_model_forward_isolation_between_instances(self):
        """Verify patching one instance doesn't affect another instance."""
        model1 = MockModelClass()
        model2 = MockModelClass()

        llada = object.__new__(LladaModel)

        # Patch model1
        llada.patch_model_forward(model1, mask_token_id=100)

        # Only model1 should have patched forward in its __dict__
        assert "forward" in model1.__dict__
        assert "forward" not in model2.__dict__

        # Both should still have access to class method (through MRO)
        assert callable(model2.forward)

    def test_patch_model_forward_mask_token_handling(self):
        """Verify the wrapped forward correctly sets mask token logits to -inf."""
        model = MockModelClass()
        llada = object.__new__(LladaModel)

        # Create mock output with logits tensor (vocab_size=60000 to fit mask token 50257)
        import torch

        mock_output = mock.MagicMock()
        mock_output.logits = torch.zeros((1, 5, 60000))

        # Mock the class forward to return our mock output
        with mock.patch.object(MockModelClass, "forward", return_value=mock_output):
            llada.patch_model_forward(model, mask_token_id=50257)

            # Call the patched forward
            result = model.forward()

            # Verify mask token logits are set to -inf
            assert result.logits[0, 0, 50257].item() == -float("inf")
            # Verify other logits are unchanged (still 0)
            assert result.logits[0, 0, 50256].item() == 0.0


class TestGenerateHistoryNoneSafety:
    """Test history=None safe handling in generate method."""

    def test_compute_decoding_order_correlation_requires_non_none_history(self):
        """Verify compute_decoding_order_correlation_from_history raises with None.

        This regression test confirms that the function requires a valid history
        and would fail if called with None, justifying the guard in generate().
        """
        mock_tokenizer = mock.MagicMock()

        with pytest.raises(AttributeError):
            compute_decoding_order_correlation_from_history(
                mock_tokenizer, history=None
            )

    def test_generate_guards_history_none_before_correlation_computation(self):
        """Verify generate() guards history=None before calling correlation function.

        This test uses mock.patch to verify the guard prevents calling the
        correlation function when history is None.
        """
        import torch

        llada = object.__new__(LladaModel)
        llada.tokenizer = mock.MagicMock()
        llada.model = mock.MagicMock()
        llada.mask_id = 50257
        llada.accel_framework = "test"

        output_ids = torch.tensor([[1, 2, 3]])
        with mock.patch.object(llada, "_generate", return_value=(output_ids, 10, None)):
            with mock.patch(
                "model.local.llada.llada_model.compute_decoding_order_correlation_from_history"
            ) as mock_correlation:
                with mock.patch(
                    "model.local.llada.llada_model.decode_history",
                    return_value=None,
                ):
                    llada.tokenizer.batch_decode.return_value = ["output"]
                    llada.tokenizer.pad_token_id = 0
                    llada.tokenizer.apply_chat_template.return_value = "test"
                    llada.tokenizer.return_value = mock.MagicMock(
                        input_ids=torch.tensor([[1]])
                    )
                    llada.model.device = "cpu"

                    result = llada.generate(
                        messages="test", gen_config={}, output_history=False
                    )

                    mock_correlation.assert_not_called()
                    assert result.decoding_order is None
                    assert result.decoding_order_corrs is None

    def test_generate_calls_correlation_when_history_is_not_none(self):
        """Verify generate() DOES call correlation function when history is not None."""
        import torch

        llada = object.__new__(LladaModel)
        llada.tokenizer = mock.MagicMock()
        llada.model = mock.MagicMock()
        llada.mask_id = 50257
        llada.accel_framework = "test"

        output_ids = torch.tensor([[1, 2, 3]])
        mock_history = [torch.tensor([[1, 2, 3]])]

        with mock.patch.object(
            llada, "_generate", return_value=(output_ids, 10, mock_history)
        ):
            with mock.patch(
                "model.local.llada.llada_model.compute_decoding_order_correlation_from_history",
                return_value=([1, 2, 3], {"dec_order_kendall": 0.9}),
            ) as mock_correlation:
                with mock.patch(
                    "model.local.llada.llada_model.decode_history",
                    return_value=["decoded"],
                ):
                    llada.tokenizer.batch_decode.return_value = ["output"]
                    llada.tokenizer.pad_token_id = 0
                    llada.tokenizer.apply_chat_template.return_value = "test"
                    llada.tokenizer.return_value = mock.MagicMock(
                        input_ids=torch.tensor([[1]])
                    )
                    llada.model.device = "cpu"

                    result = llada.generate(
                        messages="test", gen_config={}, output_history=True
                    )

                    mock_correlation.assert_called_once()
                    assert result.decoding_order == [1, 2, 3]
                    assert result.decoding_order_corrs == {"dec_order_kendall": 0.9}


class TestDecodeHistoryNoneSafety:
    """Test decode_history None safety (confirm existing behavior)."""

    def test_decode_history_with_none_returns_none(self):
        """Verify decode_history(tokenizer, history=None) returns None."""
        mock_tokenizer = mock.MagicMock()

        result = decode_history(mock_tokenizer, history=None)

        assert result is None

    def test_decode_history_with_valid_history_returns_list(self):
        """Verify decode_history with valid history returns list of decoded strings."""
        import torch

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.decode.return_value = "decoded_text"

        history = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]

        result = decode_history(mock_tokenizer, history=history)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "decoded_text"
        assert result[1] == "decoded_text"
