"""Smoke test to verify basic model package functionality."""


def test_import_model_package():
    """Test that model package can be imported and load_model is callable."""
    from model import load_model

    assert callable(load_model), "load_model should be callable"
