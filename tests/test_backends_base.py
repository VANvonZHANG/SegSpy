"""Tests for SegSpy.backends.base.SegmentationBackend."""
import pytest

from SegSpy.backends.base import SegmentationBackend


def test_segmentation_backend_is_abstract():
    with pytest.raises(TypeError):
        SegmentationBackend()
