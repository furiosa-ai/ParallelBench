"""Tests for generate() in model/local/generate.py."""

from unittest import mock

from parallelbench.models.local.generate import generate


def test_generate_is_callable():
    """generate() should be a callable function."""
    assert callable(generate)


@mock.patch("parallelbench.models.local.generate.get_num_transfer_tokens")
def test_generate_accepts_expected_kwargs(mock_transfer):
    """generate() should accept standard generation kwargs without error."""
    # We just check the signature accepts these params (actual execution needs GPU)
    import inspect

    sig = inspect.signature(generate)
    params = set(sig.parameters.keys())
    expected = {
        "model",
        "prompt",
        "steps",
        "gen_length",
        "block_length",
        "temperature",
        "unmasking",
        "mask_id",
        "threshold",
        "factor",
        "output_history",
        "output0_ids",
        "alg_temp",
        "eb_sampler_gamma",
    }
    assert expected.issubset(params)


def test_generate_does_not_accept_fast_dllm_cache_params():
    """generate() should no longer accept use_fast_dllm_cache or use_fast_dllm_dual_cache."""
    import inspect

    sig = inspect.signature(generate)
    params = set(sig.parameters.keys())
    assert "use_fast_dllm_cache" not in params
    assert "use_fast_dllm_dual_cache" not in params
