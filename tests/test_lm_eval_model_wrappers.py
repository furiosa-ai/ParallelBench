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


# ============================================================
# Batch generation tests
# ============================================================

_COMMON_GEN_KWARGS = {
    "until": ["\n\n"],
    "do_sample": False,
    "steps": 32,
    "max_tokens": 32,
    "block_length": 32,
    "unmasking": "random",
}


class TestBatchGenerationSequentialRegression:
    """batch_size=1 should behave identically to the original sequential loop."""

    @patch("parallelbench.lm_eval_wrappers.llada_wrapper.LladaModel")
    def test_batch_size_1_uses_sequential_path(self, MockLladaModel):
        mock_model = MagicMock()
        mock_model.generate.return_value = _make_dllm_output("result_a")
        MockLladaModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.llada_wrapper import LLaDAWrapper

        wrapper = LLaDAWrapper(model_path="GSAI-ML/LLaDA-1.5", batch_size=1)

        instances = [
            _make_mock_instance("ctx_1", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
            _make_mock_instance("ctx_2", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
        ]
        results = wrapper.generate_until(instances)

        assert len(results) == 2
        # Sequential path calls generate() per request, not generate_batch()
        assert mock_model.generate.call_count == 2
        mock_model.generate_batch.assert_not_called()

        # Metadata stored in request order
        store = MetadataStore.instance()
        meta_1 = store.pop()
        meta_2 = store.pop()
        assert meta_1.nfe == 42
        assert meta_2.nfe == 42


class TestBatchGenerationUnsupported:
    """batch_size>1 with a model that doesn't support batching should raise."""

    @patch("parallelbench.lm_eval_wrappers.llada_wrapper.LladaModel")
    def test_batch_size_gt1_raises_not_implemented(self, MockLladaModel):
        from parallelbench.models.base_model import BaseModel

        mock_model = MagicMock(spec=BaseModel)
        # Use the real BaseModel.generate_batch (raises NotImplementedError)
        mock_model.generate_batch.side_effect = NotImplementedError(
            "LladaModel does not support batch generation."
        )
        MockLladaModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.llada_wrapper import LLaDAWrapper

        wrapper = LLaDAWrapper(model_path="GSAI-ML/LLaDA-1.5", batch_size=4)

        instances = [
            _make_mock_instance("ctx_1", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
            _make_mock_instance("ctx_2", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
        ]
        with pytest.raises(NotImplementedError, match="does not support batch"):
            wrapper.generate_until(instances)


class TestBatchGenerationSupported:
    """batch_size>1 with a model that supports batching should call generate_batch()."""

    @patch("parallelbench.lm_eval_wrappers.dream_wrapper.DreamModel")
    def test_batch_calls_generate_batch(self, MockDreamModel):
        mock_model = MagicMock()
        mock_model.supports_batch = True
        mock_model.generate_batch.return_value = [
            _make_dllm_output("out_a", nfe=10),
            _make_dllm_output("out_b", nfe=20),
        ]
        MockDreamModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.dream_wrapper import DreamWrapper

        wrapper = DreamWrapper(
            model_path="Dream-org/Dream-v0-Instruct-7B", batch_size=4
        )

        instances = [
            _make_mock_instance("short", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
            _make_mock_instance(
                "a longer context", gen_kwargs=dict(_COMMON_GEN_KWARGS)
            ),
        ]
        results = wrapper.generate_until(instances)

        assert len(results) == 2
        mock_model.generate_batch.assert_called_once()
        # generate() should not be called in batch path
        mock_model.generate.assert_not_called()

    @patch("parallelbench.lm_eval_wrappers.dream_wrapper.DreamModel")
    def test_metadata_ordering_after_collator_reorder(self, MockDreamModel):
        """Metadata should be in original request order, not Collator's sorted order."""
        mock_model = MagicMock()
        mock_model.supports_batch = True

        # Collator sorts by descending context length, so order will be:
        # "very long context" (idx 1), "medium ctx" (idx 2), "hi" (idx 0)
        # We return outputs in that sorted order
        mock_model.generate_batch.return_value = [
            _make_dllm_output("out_long", nfe=30),
            _make_dllm_output("out_medium", nfe=20),
            _make_dllm_output("out_short", nfe=10),
        ]
        MockDreamModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.dream_wrapper import DreamWrapper

        wrapper = DreamWrapper(
            model_path="Dream-org/Dream-v0-Instruct-7B", batch_size=8
        )

        instances = [
            _make_mock_instance("hi", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
            _make_mock_instance(
                "very long context", gen_kwargs=dict(_COMMON_GEN_KWARGS)
            ),
            _make_mock_instance("medium ctx", gen_kwargs=dict(_COMMON_GEN_KWARGS)),
        ]
        results = wrapper.generate_until(instances)

        # Results should be in original order: hi, very long context, medium ctx
        assert results[0] == "out_short"
        assert results[1] == "out_long"
        assert results[2] == "out_medium"

        # Metadata should also be in original order
        store = MetadataStore.instance()
        meta_0 = store.pop()  # "hi" — nfe=10
        meta_1 = store.pop()  # "very long context" — nfe=30
        meta_2 = store.pop()  # "medium ctx" — nfe=20
        assert meta_0.nfe == 10
        assert meta_1.nfe == 30
        assert meta_2.nfe == 20

    @patch("parallelbench.lm_eval_wrappers.dream_wrapper.DreamModel")
    def test_different_gen_kwargs_grouped_separately(self, MockDreamModel):
        """Requests with different gen_kwargs should be in separate batches."""
        mock_model = MagicMock()
        mock_model.supports_batch = True
        mock_model.generate_batch.side_effect = [
            [_make_dllm_output("batch1_a"), _make_dllm_output("batch1_b")],
            [_make_dllm_output("batch2_a")],
        ]
        MockDreamModel.return_value = mock_model

        from parallelbench.lm_eval_wrappers.dream_wrapper import DreamWrapper

        wrapper = DreamWrapper(
            model_path="Dream-org/Dream-v0-Instruct-7B", batch_size=8
        )

        gen_kwargs_a = dict(_COMMON_GEN_KWARGS, steps=32)
        gen_kwargs_b = dict(_COMMON_GEN_KWARGS, steps=64)

        instances = [
            _make_mock_instance("ctx_1", gen_kwargs=dict(gen_kwargs_a)),
            _make_mock_instance("ctx_2", gen_kwargs=dict(gen_kwargs_a)),
            _make_mock_instance("ctx_3", gen_kwargs=dict(gen_kwargs_b)),
        ]
        results = wrapper.generate_until(instances)

        assert len(results) == 3
        # Two separate generate_batch() calls (one per gen_kwargs group)
        assert mock_model.generate_batch.call_count == 2
