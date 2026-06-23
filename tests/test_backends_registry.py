"""Tests for SegSpy.backends.registry.SegmentationRegistry."""
import numpy as np
import pytest

from SegSpy.backends.base import SegmentationBackend
from SegSpy.backends.registry import SegmentationRegistry


class DummyBackend(SegmentationBackend):
    name = "dummy"

    def segment(self, signal, config):
        return np.zeros_like(np.asarray(signal.data), dtype=np.uint8)


@pytest.fixture(autouse=True)
def isolated_registry():
    """Give each test an empty registry, then restore the built-ins afterwards.

    The registry is module-level global state; clearing it and leaving it empty
    would unregister the built-in backends for the rest of the test session, so
    we snapshot and restore (mirrors tests/test_backends_init.py).
    """
    saved = dict(SegmentationRegistry._backends)
    SegmentationRegistry.clear()
    yield
    SegmentationRegistry._backends.clear()
    SegmentationRegistry._backends.update(saved)


def test_register_and_get():
    SegmentationRegistry.register(DummyBackend)
    backend = SegmentationRegistry.get("dummy")
    assert isinstance(backend, DummyBackend)


def test_get_unknown_raises():
    with pytest.raises(ValueError, match="Unknown segmentation backend"):
        SegmentationRegistry.get("nonexistent")


def test_list_backends_empty_then_filled():
    assert SegmentationRegistry.list_backends() == []
    SegmentationRegistry.register(DummyBackend)
    assert SegmentationRegistry.list_backends() == ["dummy"]


def test_find_compatible():
    class SootOnly(SegmentationBackend):
        name = "soot_only"

        @classmethod
        def supports(cls, mt, pt):
            return pt == "soot"

        def segment(self, signal, config):
            return np.zeros((2, 2), np.uint8)

    SegmentationRegistry.register(SootOnly)
    assert SegmentationRegistry.find_compatible("TEM", "soot") == "soot_only"
    with pytest.raises(ValueError):
        SegmentationRegistry.find_compatible("TEM", "spherical")


def test_register_duplicate_name_raises():
    SegmentationRegistry.register(DummyBackend)

    class AnotherDummy(SegmentationBackend):
        name = "dummy"

        def segment(self, signal, config):
            return np.zeros((2, 2), np.uint8)

    with pytest.raises(ValueError, match="already registered"):
        SegmentationRegistry.register(AnotherDummy)


def test_register_same_class_again_ok():
    SegmentationRegistry.register(DummyBackend)
    SegmentationRegistry.register(DummyBackend)  # idempotent — should not raise
    assert SegmentationRegistry.list_backends() == ["dummy"]


def test_register_without_name_raises():
    class NoName(SegmentationBackend):
        def segment(self, signal, config):
            return np.zeros((2, 2), np.uint8)

    with pytest.raises(ValueError, match="must define a 'name'"):
        SegmentationRegistry.register(NoName)


def test_register_non_backend_raises():
    with pytest.raises(TypeError):
        SegmentationRegistry.register(str)


def test_clear():
    SegmentationRegistry.register(DummyBackend)
    assert SegmentationRegistry.list_backends() == ["dummy"]
    SegmentationRegistry.clear()
    assert SegmentationRegistry.list_backends() == []
