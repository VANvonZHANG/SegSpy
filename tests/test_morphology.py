"""Tests for SegSpy.morphology.measure_morphology."""
import pytest
import cv2
import numpy as np

from SegSpy.core import ParticleObject
from SegSpy.morphology import measure_morphology


REQUIRED_KEYS = {
    "area_px",
    "equivalent_diameter_px",
    "equivalent_diameter_nm",
    "feret_min_px",
    "feret_max_px",
    "feret_max_nm",
    "perimeter_px",
    "aspect_ratio",
    "convexity",
    "roundness",
    "solidity",
    "centroid_yx",
}


def _obj_from_mask(mask):
    return ParticleObject(particle_id=1, image_roi=None, mask_roi=mask)


def _circle_mask(size, center, radius):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _rect_mask(h, w, y0, x0, bh, bw):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0 : y0 + bh, x0 : x0 + bw] = 255
    return mask


def test_returns_dict_with_all_required_keys():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    assert set(metrics.keys()) == REQUIRED_KEYS


def test_writes_metrics_onto_object():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    measure_morphology(obj)
    assert REQUIRED_KEYS.issubset(obj.metrics.keys())


def test_circle_area_matches_pi_r_squared():
    radius = 20
    mask = _circle_mask(100, (50, 50), radius)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    # Discrete circle is slightly smaller than the ideal; allow ~5% slack.
    assert metrics["area_px"] == pytest.approx(np.pi * radius ** 2, rel=0.05)


def test_circle_equivalent_diameter_matches_2r():
    radius = 20
    mask = _circle_mask(100, (50, 50), radius)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    expected = 2 * np.sqrt(metrics["area_px"] / np.pi)
    assert metrics["equivalent_diameter_px"] == expected
    # equivalent_diameter_px should be ~2r = 40.
    assert abs(metrics["equivalent_diameter_px"] - 2 * radius) < 1.5


def test_circle_roundness_near_one():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    assert abs(metrics["roundness"] - 1.0) < 0.1


def test_circle_solidity_near_one():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    assert abs(metrics["solidity"] - 1.0) < 0.05


def test_circle_convexity_near_one():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    assert metrics["convexity"] <= 1.0 + 1e-6
    # Discrete (pixel-staircase) circle contour is slightly longer than its
    # convex hull, so convexity is a touch under 1.0 (~0.95); still well above
    # the ~0.7-0.8 of a genuinely concave shape.
    assert metrics["convexity"] > 0.9


def test_rectangle_aspect_ratio_greater_than_one():
    # 40 tall x 10 wide -> aspect_ratio ~ 4.
    mask = _rect_mask(80, 80, 20, 35, 40, 10)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    assert metrics["aspect_ratio"] > 1.5
    assert metrics["feret_max_px"] > metrics["feret_min_px"]


def test_nm_variants_equal_px_times_scale():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj, scale_nm=2.5)
    assert metrics["equivalent_diameter_nm"] == pytest.approx(
        metrics["equivalent_diameter_px"] * 2.5
    )
    assert metrics["feret_max_nm"] == pytest.approx(
        metrics["feret_max_px"] * 2.5
    )


def test_default_scale_nm_one_means_nm_equals_px():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)  # scale_nm defaults to 1.0
    assert metrics["equivalent_diameter_nm"] == pytest.approx(
        metrics["equivalent_diameter_px"]
    )


def test_centroid_within_mask_bounds():
    mask = _circle_mask(100, (50, 50), 20)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    cy, cx = metrics["centroid_yx"]
    assert 30 <= cy <= 70
    assert 30 <= cx <= 70


def test_aspect_ratio_is_feret_max_over_min():
    mask = _rect_mask(80, 80, 20, 35, 40, 10)
    obj = _obj_from_mask(mask)
    metrics = measure_morphology(obj)
    expected = metrics["feret_max_px"] / metrics["feret_min_px"]
    assert metrics["aspect_ratio"] == pytest.approx(expected)
