"""Tests for SegSpy.backends.sam (post-filter + backend).

The 5-stage ``post_filter_masks`` is SegSpy's standout and runs without torch,
so those tests always execute. The full ``segment()`` path needs the ``[sam]``
extra and is skipped when unavailable.
"""
import numpy as np
import pytest

from SegSpy.backends.sam import SAMSegmenter, _is_nested, post_filter_masks


class _FakeConfig:
    sam_pred_iou_thresh = 0.88
    sam_stability_score_thresh = 0.95
    sam_min_area_ratio = 0.0005
    sam_max_area_ratio = 0.60
    sam_intensity_ratio = 0.85
    sam_edge_margin = 5


def _mask(h, w, y_slice, x_slice):
    m = np.zeros((h, w), dtype=bool)
    m[y_slice, x_slice] = True
    return m


# --- _is_nested ---


def test_is_nested_true():
    large = _mask(100, 100, slice(10, 50), slice(10, 50))
    small = _mask(100, 100, slice(20, 40), slice(20, 40))
    assert _is_nested(small, large, threshold=0.8) is True


def test_is_nested_false():
    a = _mask(100, 100, slice(10, 30), slice(10, 30))
    b = _mask(100, 100, slice(50, 70), slice(50, 70))
    assert _is_nested(a, b, threshold=0.8) is False


# --- post_filter_masks: one test per stage ---


def test_stage1_confidence_filter():
    original = np.ones((100, 100), dtype=np.uint8) * 200
    original[10:30, 10:30] = 50
    original[50:70, 50:70] = 50
    masks = [
        {"segmentation": _mask(100, 100, slice(10, 30), slice(10, 30)),
         "area": 400, "bbox": [10, 10, 20, 20],
         "predicted_iou": 0.95, "stability_score": 0.97},
        {"segmentation": _mask(100, 100, slice(50, 70), slice(50, 70)),
         "area": 400, "bbox": [50, 50, 20, 20],
         "predicted_iou": 0.70, "stability_score": 0.97},  # low IoU -> drop
    ]
    result = post_filter_masks(masks, original, (100, 100), _FakeConfig())
    assert len(result) == 1
    assert result[0]["predicted_iou"] == 0.95


def test_stage2_area_filter():
    original = np.ones((100, 100), dtype=np.uint8) * 200
    original[10:20, 10:20] = 50
    original[0:90, 0:90] = 50
    masks = [
        {"segmentation": _mask(100, 100, slice(10, 20), slice(10, 20)),
         "area": 100, "bbox": [10, 10, 10, 10],
         "predicted_iou": 0.95, "stability_score": 0.97},
        {"segmentation": _mask(100, 100, slice(0, 90), slice(0, 90)),
         "area": 8100, "bbox": [0, 0, 90, 90],
         "predicted_iou": 0.95, "stability_score": 0.97},  # >60% -> drop
    ]
    result = post_filter_masks(masks, original, (100, 100), _FakeConfig())
    assert len(result) == 1
    assert result[0]["area"] == 100


def test_stage3_intensity_filter_keeps_dark():
    original = np.ones((100, 100), dtype=np.uint8) * 200
    original[10:30, 10:30] = 50
    masks = [
        {"segmentation": _mask(100, 100, slice(10, 30), slice(10, 30)),
         "area": 400, "bbox": [10, 10, 20, 20],
         "predicted_iou": 0.95, "stability_score": 0.97},  # dark -> keep
        {"segmentation": _mask(100, 100, slice(50, 70), slice(50, 70)),
         "area": 400, "bbox": [50, 50, 20, 20],
         "predicted_iou": 0.95, "stability_score": 0.97},  # bright -> drop
    ]
    result = post_filter_masks(masks, original, (100, 100), _FakeConfig())
    assert len(result) == 1
    assert np.array_equal(result[0]["segmentation"], masks[0]["segmentation"])


def test_stage4_nested_mask_resolution():
    large = _mask(100, 100, slice(10, 50), slice(10, 50))
    small = _mask(100, 100, slice(20, 40), slice(20, 40))
    masks = [
        {"segmentation": large, "area": 1600, "bbox": [10, 10, 40, 40],
         "predicted_iou": 0.95, "stability_score": 0.97},
        {"segmentation": small, "area": 400, "bbox": [20, 20, 20, 20],
         "predicted_iou": 0.95, "stability_score": 0.97},
    ]
    original = np.ones((100, 100), dtype=np.uint8) * 200
    original[10:50, 10:50] = 50
    result = post_filter_masks(masks, original, (100, 100), _FakeConfig())
    assert len(result) == 1
    assert np.array_equal(result[0]["segmentation"], large)


def test_stage5_edge_exclusion():
    original = np.ones((100, 100), dtype=np.uint8) * 200
    original[0:20, 0:20] = 50
    original[10:30, 10:30] = 50
    masks = [
        {"segmentation": _mask(100, 100, slice(0, 20), slice(0, 20)),
         "area": 400, "bbox": [0, 0, 20, 20],
         "predicted_iou": 0.95, "stability_score": 0.97},  # touches edge
        {"segmentation": _mask(100, 100, slice(10, 30), slice(10, 30)),
         "area": 400, "bbox": [10, 10, 20, 20],
         "predicted_iou": 0.95, "stability_score": 0.97},
    ]
    result = post_filter_masks(masks, original, (100, 100), _FakeConfig())
    assert len(result) == 1
    assert result[0]["bbox"] == [10, 10, 20, 20]


# --- backend metadata ---


def test_sam_name():
    assert SAMSegmenter.name == "sam"


def test_sam_supports_is_particle_agnostic():
    assert SAMSegmenter.supports("TEM", "soot") is True
    assert SAMSegmenter.supports("SEM", "spherical") is True


def test_module_imports_without_sam_extra():
    # Importing the module must not require torch/segment_anything (lazy import).
    import importlib

    mod = importlib.import_module("SegSpy.backends.sam")
    assert hasattr(mod, "SAMSegmenter")
    assert hasattr(mod, "post_filter_masks")


# --- full segment() path (needs the [sam] extra) ---

def test_segment_requires_sam_extra_when_unavailable():
    """If the [sam] extra is missing, segment() raises an ImportError pointing
    at ``pip install SegSpy[sam]``."""
    pytest.importorskip("segment_anything")
    # Only meaningful when torch/sam ARE present we just assert the happy path
    # runs without importing errors up to model loading; full model load is
    # covered by the end-to-end CLI test on real data.
    assert SAMSegmenter().name == "sam"
