"""Tests for LLaDA safety fixes: instance-level patching and history=None handling."""

import types
from unittest import mock


from parallelbench.models.local.llada.llada_model import LladaModel


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
