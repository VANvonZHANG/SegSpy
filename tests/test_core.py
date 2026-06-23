"""Tests for SegSpy.core.ParticleObject."""

import numpy as np

from SegSpy.core import ParticleObject


def test_construct_from_ndarray_roi_and_mask_round_trips():
    roi = np.zeros((40, 60), dtype=np.float32)
    mask = np.zeros((40, 60), dtype=np.uint8)
    obj = ParticleObject(
        particle_id=7,
        image_roi=roi,
        mask_roi=mask,
        offset=(10, 20),
        bbox=(20, 10, 60, 40),
        area_px=1500.0,
    )
    assert obj.particle_id == 7
    assert obj.offset == (10, 20)
    assert obj.bbox == (20, 10, 60, 40)
    assert obj.area_px == 1500.0
    assert obj.image_roi is roi
    assert obj.mask_roi is mask


def test_metrics_defaults_to_empty_dict_and_is_per_instance():
    obj_a = ParticleObject(1, np.zeros((4, 4)), np.zeros((4, 4)))
    obj_b = ParticleObject(2, np.zeros((4, 4)), np.zeros((4, 4)))
    assert obj_a.metrics == {}
    assert obj_b.metrics == {}
    obj_a.add_metric("area", 1.0)
    assert obj_a.metrics == {"area": 1.0}
    # Default must be per-instance, not a shared mutable.
    assert obj_b.metrics == {}


def test_get_physical_scale_nm_returns_one_for_plain_ndarray():
    obj = ParticleObject(1, np.zeros((10, 10)), np.zeros((10, 10)))
    assert obj.get_physical_scale_nm() == 1.0


def test_get_physical_scale_nm_reads_hyperspy_calibration():
    import hyperspy.api as hs

    sig = hs.signals.Signal2D(np.zeros((10, 10)))
    sig.axes_manager[0].scale = 0.5
    sig.axes_manager[0].units = "um"
    obj = ParticleObject(1, sig, np.zeros((10, 10), dtype=np.uint8))
    # 0.5 um/pixel -> 500 nm/pixel
    assert obj.get_physical_scale_nm() == 500.0
