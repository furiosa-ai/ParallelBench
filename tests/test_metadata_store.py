"""Tests for DLLMMetadataStore."""

import threading

import pytest

from parallelbench.lm_eval_models.metadata_store import (
    GenerationMetadata,
    MetadataStore,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset singleton before each test."""
    MetadataStore.reset()
    yield
    MetadataStore.reset()


class TestGenerationMetadata:
    def test_default_values(self):
        meta = GenerationMetadata()
        assert meta.nfe == 0
        assert meta.history is None
        assert meta.decoding_order is None
        assert meta.decoding_order_corrs is None
        assert meta.input_length is None
        assert meta.output_length is None
        assert meta.perf_stats is None

    def test_custom_values(self):
        meta = GenerationMetadata(nfe=42, input_length=10, output_length=20)
        assert meta.nfe == 42
        assert meta.input_length == 10
        assert meta.output_length == 20


class TestMetadataStore:
    def test_singleton(self):
        store1 = MetadataStore.instance()
        store2 = MetadataStore.instance()
        assert store1 is store2

    def test_append_and_pop_fifo_order(self):
        store = MetadataStore.instance()

        store.append(GenerationMetadata(nfe=1))
        store.append(GenerationMetadata(nfe=2))
        store.append(GenerationMetadata(nfe=3))

        assert store.pop().nfe == 1
        assert store.pop().nfe == 2
        assert store.pop().nfe == 3

    def test_pop_empty_returns_default(self):
        store = MetadataStore.instance()
        meta = store.pop()
        assert meta.nfe == 0
        assert meta.history is None

    def test_len(self):
        store = MetadataStore.instance()
        assert len(store) == 0

        store.append(GenerationMetadata(nfe=1))
        assert len(store) == 1

        store.append(GenerationMetadata(nfe=2))
        assert len(store) == 2

        store.pop()
        assert len(store) == 1

    def test_clear(self):
        store = MetadataStore.instance()
        store.append(GenerationMetadata(nfe=1))
        store.append(GenerationMetadata(nfe=2))

        store.clear()
        assert len(store) == 0

    def test_reset_creates_new_instance(self):
        store1 = MetadataStore.instance()
        store1.append(GenerationMetadata(nfe=1))

        MetadataStore.reset()
        store2 = MetadataStore.instance()

        assert store1 is not store2
        assert len(store2) == 0

    def test_thread_safety(self):
        """Multiple threads can safely append and pop without data loss."""
        store = MetadataStore.instance()
        num_items = 100

        def append_items(start):
            for i in range(start, start + num_items):
                store.append(GenerationMetadata(nfe=i))

        threads = [
            threading.Thread(target=append_items, args=(i * num_items,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(store) == 4 * num_items

        nfes = set()
        for _ in range(4 * num_items):
            nfes.add(store.pop().nfe)

        assert len(nfes) == 4 * num_items
