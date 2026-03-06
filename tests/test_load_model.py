"""Tests for load_model() dispatch in model/__init__.py."""

from unittest import mock

import pytest

from parallelbench.model.base_model import ApiModel


@mock.patch("parallelbench.model.ModelRegistry.get_model_class")
def test_load_model_registry_local_model_passes_accel_framework(mock_get):
    from parallelbench.model import load_model

    # Create a real class so issubclass() works
    class FakeLocalModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    mock_get.return_value = FakeLocalModel

    result = load_model("my-model", accel_framework="fast_dllm", extra_arg=42)
    assert result.args == ("my-model",)
    assert result.kwargs == {"accel_framework": "fast_dllm", "extra_arg": 42}


@mock.patch("parallelbench.model.ModelRegistry.get_model_class")
def test_load_model_registry_api_model_omits_accel_framework(mock_get):
    from parallelbench.model import load_model

    # Create a real class that is a subclass of ApiModel
    class FakeApiModel(ApiModel):
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def generate(
            self, messages, gen_config=None, output_prefix=None, output_history=False
        ):
            pass

    mock_get.return_value = FakeApiModel

    result = load_model("api-model", accel_framework="vllm", api_key="xyz")
    # accel_framework should NOT be passed to ApiModel
    assert result._args == ("api-model",)
    assert result._kwargs == {"api_key": "xyz"}


@mock.patch("parallelbench.model.ModelRegistry.get_model_class", side_effect=ValueError)
@mock.patch("parallelbench.model.vllmModel")
def test_load_model_fallback_to_vllm(mock_vllm, mock_get):
    from parallelbench.model import load_model

    load_model("unknown-model", accel_framework="vllm")
    mock_vllm.assert_called_once_with("unknown-model")


@mock.patch("parallelbench.model.ModelRegistry.get_model_class", side_effect=ValueError)
@mock.patch("parallelbench.model.TransformersModel")
def test_load_model_fallback_to_transformers(mock_transformers, mock_get):
    from parallelbench.model import load_model

    load_model("unknown-model", accel_framework="transformers")
    mock_transformers.assert_called_once_with("unknown-model")


@mock.patch("parallelbench.model.ModelRegistry.get_model_class", side_effect=ValueError)
def test_load_model_raises_when_no_match_and_no_fallback(mock_get):
    from parallelbench.model import load_model

    with pytest.raises(ValueError, match="not supported"):
        load_model("unknown-model")


@mock.patch("parallelbench.model.ModelRegistry.get_model_class", side_effect=ValueError)
@mock.patch("parallelbench.model.vllmModel")
def test_load_model_pops_accel_framework_from_fallback_kwargs(mock_vllm, mock_get):
    from parallelbench.model import load_model

    load_model("unknown-model", accel_framework="vllm", batch_size=4)
    # accel_framework should have been popped, not forwarded to vllmModel
    mock_vllm.assert_called_once_with("unknown-model", batch_size=4)
