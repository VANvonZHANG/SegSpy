"""Tests for SegSpy.backends package-level registration of built-in backends.

These assert that importing the backends subpackage (or SegSpy itself)
auto-registers the two built-in backends: ``traditional_cv`` and ``sam``.
"""

import pytest

from SegSpy.backends import (
    SAMSegmenter,
    SegmentationBackend,
    SegmentationRegistry,
    TraditionalCVSegmenter,
)


@pytest.fixture(autouse=True)
def restore_registry():
    registered = dict(SegmentationRegistry._backends)
    yield
    SegmentationRegistry._backends.clear()
    SegmentationRegistry._backends.update(registered)


def test_builtins_registered():
    backends = SegmentationRegistry.list_backends()
    assert "traditional_cv" in backends
    assert "sam" in backends


def test_get_traditional_returns_correct_class():
    backend = SegmentationRegistry.get("traditional_cv")
    assert isinstance(backend, TraditionalCVSegmenter)


def test_get_sam_returns_correct_class():
    backend = SegmentationRegistry.get("sam")
    assert isinstance(backend, SAMSegmenter)


def test_public_symbols_re_exported():
    # The subpackage re-exports the ABC, registry, and both backends.
    assert issubclass(TraditionalCVSegmenter, SegmentationBackend)
    assert issubclass(SAMSegmenter, SegmentationBackend)


def test_registration_order_is_traditional_then_sam():
    backends = SegmentationRegistry.list_backends()
    assert backends.index("traditional_cv") < backends.index("sam")
