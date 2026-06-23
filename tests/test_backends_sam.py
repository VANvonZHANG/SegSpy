"""Tests for SegSpy.backends.sam post_filter (runs without torch)."""
import numpy as np

from SegSpy.backends.sam import _is_nested, post_filter_masks


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


def test_is_nested_true():
    large = _mask(100, 100, slice(10, 50), slice(10, 50))
    small = _mask(100, 100, slice(20, 40), slice(20, 40))
    assert _is_nested(small, large, threshold=0.8) is True


def test_is_nested_false():
    a = _mask(100, 100, slice(10, 30), slice(10, 30))
    b = _mask(100, 100, slice(50, 70), slice(50, 70))
    assert _is_nested(a, b, threshold=0.8) is False


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
