"""Tests for lm-eval model wrappers using mock inner models."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from parallelbench.models.base_model import DLLMOutput
from parallelbench.lm_eval_wrappers.metadata_store import MetadataStore


@pytest.fixture(autouse=True)
def reset_store():
    MetadataStore.reset()
    yield
    MetadataStore.reset()


def _make_dllm_output(text="hello world", nfe=42, input_len=10, output_len=5):
    """Create a mock DLLMOutput."""
    input_ids = torch.zeros(1, input_len, dtype=torch.long)
    output_ids = torch.zeros(1, output_len, dtype=torch.long)
    return DLLMOutput(
        output=text,
        nfe=nfe,
        input_ids=input_ids,
        output_ids=output_ids,
        history=None,
        decoding_order=None,
        decoding_order_corrs={"dec_order_kendall": 0.5, "dec_order_spearman": 0.6},
    )


def _make_mock_instance(context="What is 2+2?", gen_kwargs=None):
    """Create a mock lm-eval Instance."""
    if gen_kwargs is None:
        gen_kwargs = {"until": ["\n\n"], "do_sample": False}
    instance = MagicMock()
    instance.args = (context, gen_kwargs)
    return instance


class TestDLLMBase:
    def test_apply_until_truncation_basic(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        text = "Hello world\n\nMore text"
        result = DLLMBase._apply_until_truncation(text, {"until": ["\n\n"]})
        assert result == "Hello world"

    def test_apply_until_truncation_multiple_stops(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        text = "Hello. More text\n\nEven more"
        result = DLLMBase._apply_until_truncation(text, {"until": ["\n\n", "."]})
        assert result == "Hello"

    def test_apply_until_truncation_no_match(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        text = "Hello world"
        result = DLLMBase._apply_until_truncation(text, {"until": ["\n\n"]})
        assert result == "Hello world"

    def test_apply_until_truncation_empty_until(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        text = "Hello world"
        result = DLLMBase._apply_until_truncation(text, {})
        assert result == "Hello world"

    def test_context_to_messages_string(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        # Instantiate via subclass mock - just test the static/class methods
        result = DLLMBase._context_to_messages(None, "What is 2+2?")
        assert result == [{"role": "user", "content": "What is 2+2?"}]

    def test_context_to_messages_list(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = DLLMBase._context_to_messages(None, msgs)
        assert result is msgs

    def test_loglikelihood_raises(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        # Create a minimal mock subclass
        class MockWrapper(DLLMBase):
            def _create_inner_model(self):
                return MagicMock()

        with patch.object(MockWrapper, "__init__", lambda self: None):
            wrapper = MockWrapper.__new__(MockWrapper)
            with pytest.raises(NotImplementedError):
                wrapper.loglikelihood([])

    def test_loglikelihood_rolling_raises(self):
        from parallelbench.lm_eval_wrappers.dllm_base import DLLMBase

        class MockWrapper(DLLMBase):
            def _create_inner_model(self):
                return MagicMock()

        with patch.object(MockWrapper, "__init__", lambda self: None):
            wrapper = MockWrapper.__new__(MockWrapper)
            with pytest.raises(NotImplementedError):
                wrapper.loglikelihood_rolling([])


class TestLLaDAWrapperIntegration:
    """Test LLaDAWrapper with mocked LladaModel."""

    @patch("parallelbench.lm_eval_wrappers.llada_wrapper.LladaModel")
    def test_generate_until_basic(self, MockLladaModel):
        mock_model = MagicMock()
        mock_model.generate.return_value = _make_dllm_output("4\n\nmore stuff")
        MockLladaModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.llada_wrapper import LLaDAWrapper

        wrapper = LLaDAWrapper(
            model_path="GSAI-ML/LLaDA-1.5",
        )

        instances = [
            _make_mock_instance(
                "What is 2+2?",
                gen_kwargs={
                    "until": ["\n\n"],
                    "do_sample": False,
                    "steps": 32,
                    "max_tokens": 32,
                    "block_length": 32,
                    "unmasking": "confidence_topk",
                },
            )
        ]
        results = wrapper.generate_until(instances)

        assert len(results) == 1
        assert results[0] == "4"  # truncated at \n\n

        # Check metadata was stored
        store = MetadataStore.instance()
        meta = store.pop()
        assert meta.nfe == 42
        assert meta.decoding_order_corrs["dec_order_kendall"] == 0.5

    @patch("parallelbench.lm_eval_wrappers.llada_wrapper.LladaModel")
    def test_generation_config_passed_correctly(self, MockLladaModel):
        mock_model = MagicMock()
        mock_model.generate.return_value = _make_dllm_output("result")
        MockLladaModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.llada_wrapper import LLaDAWrapper

        wrapper = LLaDAWrapper(
            model_path="GSAI-ML/LLaDA-1.5",
        )

        instances = [
            _make_mock_instance(
                gen_kwargs={
                    "until": ["\n\n"],
                    "do_sample": False,
                    "steps": 64,
                    "max_tokens": 128,
                    "block_length": 64,
                    "unmasking": "random",
                    "temperature": 0.5,
                }
            )
        ]
        wrapper.generate_until(instances)

        call_kwargs = mock_model.generate.call_args
        gen_config = call_kwargs.kwargs["gen_config"]
        assert gen_config["steps"] == 64
        assert gen_config["max_tokens"] == 128
        assert gen_config["block_length"] == 64
        assert gen_config["unmasking"] == "random"
        assert gen_config["temperature"] == 0.5


class TestDreamWrapperIntegration:
    @patch("parallelbench.lm_eval_wrappers.dream_wrapper.DreamModel")
    def test_generate_until_basic(self, MockDreamModel):
        mock_model = MagicMock()
        mock_model.generate.return_value = _make_dllm_output("dream output")
        MockDreamModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.dream_wrapper import DreamWrapper

        wrapper = DreamWrapper(
            model_path="Dream-org/Dream-v0-Instruct-7B",
        )

        instances = [
            _make_mock_instance(
                gen_kwargs={
                    "until": ["\n\n"],
                    "do_sample": False,
                    "steps": 32,
                    "max_tokens": 32,
                    "block_length": 32,
                    "unmasking": "origin",
                }
            )
        ]
        results = wrapper.generate_until(instances)

        assert len(results) == 1
        assert results[0] == "dream output"
