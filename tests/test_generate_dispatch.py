"""Tests for generate() dispatch logic in model/local/generate.py."""

from unittest import mock

import pytest


@mock.patch("model.local.generate.generate_with_no_cache")
def test_generate_default_calls_no_cache(mock_no_cache):
    from model.local.generate import generate

    sentinel = object()
    mock_no_cache.return_value = sentinel

    result = generate("model", "prompt", steps=10)
    mock_no_cache.assert_called_once_with("model", "prompt", steps=10)
    assert result is sentinel


@mock.patch("model.local.generate.generate_with_prefix_cache")
def test_generate_fast_dllm_cache_calls_prefix_cache(mock_prefix_cache):
    from model.local.generate import generate

    sentinel = object()
    mock_prefix_cache.return_value = sentinel

    result = generate("model", "prompt", use_fast_dllm_cache=True, steps=10)
    mock_prefix_cache.assert_called_once_with("model", "prompt", steps=10)
    assert result is sentinel


@mock.patch("model.local.generate.generate_with_dual_cache")
def test_generate_fast_dllm_dual_cache_calls_dual_cache(mock_dual_cache):
    from model.local.generate import generate

    sentinel = object()
    mock_dual_cache.return_value = sentinel

    result = generate("model", "prompt", use_fast_dllm_dual_cache=True, steps=10)
    mock_dual_cache.assert_called_once_with("model", "prompt", steps=10)
    assert result is sentinel


@mock.patch("model.local.generate.generate_with_no_cache")
def test_generate_passes_kwargs_to_no_cache(mock_no_cache):
    from model.local.generate import generate

    generate("m", "p", steps=5, temperature=0.8, remasking="random")
    mock_no_cache.assert_called_once_with("m", "p", steps=5, temperature=0.8, remasking="random")


@mock.patch("model.local.generate.generate_with_prefix_cache")
def test_generate_passes_kwargs_to_prefix_cache(mock_prefix_cache):
    from model.local.generate import generate

    # use_fast_dllm_cache should be consumed by generate(), not forwarded
    generate("m", "p", use_fast_dllm_cache=True, steps=5, temperature=0.8)
    mock_prefix_cache.assert_called_once_with("m", "p", steps=5, temperature=0.8)
