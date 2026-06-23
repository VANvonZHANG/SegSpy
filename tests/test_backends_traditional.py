"""Tests for SegSpy.backends.traditional.TraditionalCVSegmenter."""
import numpy as np
import pytest
import hyperspy.api as hs

from SegSpy.backends.traditional import TraditionalCVSegmenter
from SegSpy.config import SegConfig


@pytest.fixture
def segmenter():
    return TraditionalCVSegmenter()


@pytest.fixture
def dark_blob_on_bright_bg():
    """200x200 bright image (200) with a darker square blob (50) in the middle."""
    data = np.ones((200, 200), dtype=np.float32) * 200
    data[50:150, 50:150] = 50.0
    return hs.signals.Signal2D(data)


def test_name_is_traditional_cv(segmenter):
    assert TraditionalCVSegmenter.name == "traditional_cv"


def test_supports_is_particle_agnostic():
    # SegSpy backends are microscope/particle agnostic (spec §5.1).
    assert TraditionalCVSegmenter.supports("TEM", "soot") is True
    assert TraditionalCVSegmenter.supports("SEM", "spherical") is True
    assert TraditionalCVSegmenter.supports("AFM", "nanorod") is True


def test_segment_produces_uint8_mask_of_input_shape(segmenter, dark_blob_on_bright_bg):
    config = SegConfig(min_area=50, use_grabcut_refinement=False)
    mask = segmenter.segment(dark_blob_on_bright_bg, config)
    assert mask.dtype == np.uint8
    assert mask.shape == dark_blob_on_bright_bg.data.shape
    assert set(np.unique(mask)).issubset({0, 255})


def test_segment_marks_blob_interior_as_255(segmenter, dark_blob_on_bright_bg):
    config = SegConfig(min_area=50, use_grabcut_refinement=False)
    mask = segmenter.segment(dark_blob_on_bright_bg, config)
    # The center of the dark blob should be inside the detected mask.
    cy, cx = 100, 100
    assert mask[cy, cx] == 255


def test_extract_objects_finds_at_least_one_object(segmenter, dark_blob_on_bright_bg):
    config = SegConfig(min_area=50, use_grabcut_refinement=False)
    mask = segmenter.segment(dark_blob_on_bright_bg, config)
    objects = segmenter.extract_objects(dark_blob_on_bright_bg, mask, config)
    assert len(objects) >= 1
    for obj in objects:
        assert obj.area_px >= config.min_area
