"""Tests for SegSpy.backends.base.SegmentationBackend / extract_objects."""

import numpy as np
import pytest

from SegSpy.backends.base import SegmentationBackend
from SegSpy.config import SegConfig
from SegSpy.core import ParticleObject


def test_segmentation_backend_is_abstract():
    with pytest.raises(TypeError):
        SegmentationBackend()


class _StaticBackend(SegmentationBackend):
    """Backend that returns a fixed mask, ignoring the input image."""

    name = "static"

    def __init__(self, mask):
        self._mask = mask

    def segment(self, signal, config):
        return self._mask


def _make_mask_with_two_blobs():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[10:30, 10:30] = 255  # blob A, 20x20, area 400
    mask[60:90, 80:110] = 255  # blob B, 30x30, area 900
    return mask


def test_extract_objects_finds_two_objects_with_correct_bbox_and_offset():
    mask = _make_mask_with_two_blobs()
    image = np.zeros((100, 120), dtype=np.uint8)
    backend = _StaticBackend(mask)
    objects = backend.extract_objects(image, mask, SegConfig(min_area=50))

    assert len(objects) == 2
    for obj in objects:
        assert isinstance(obj, ParticleObject)
        # bbox is (x, y, w, h); offset is (y, x); they must agree.
        bx, by, bw, bh = obj.bbox
        assert obj.offset == (by, bx)
        assert bw > 0 and bh > 0
        # mask_roi shape matches the bbox height/width.
        assert obj.mask_roi.shape == (bh, bw)


def test_extract_objects_min_area_filtering():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[5:10, 5:10] = 255  # tiny blob, area 25
    mask[20:60, 20:60] = 255  # big blob, area 1600
    image = np.zeros((80, 80), dtype=np.uint8)
    backend = _StaticBackend(mask)

    objects = backend.extract_objects(image, mask, SegConfig(min_area=100))
    assert len(objects) == 1
    assert objects[0].area_px >= 100


def test_extract_objects_returns_contour_area_in_area_px():
    import cv2

    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:40, 10:40] = 255  # filled 30x30 square
    image = np.zeros((60, 60), dtype=np.uint8)
    backend = _StaticBackend(mask)
    objects = backend.extract_objects(image, mask, SegConfig(min_area=10))
    # area_px is cv2.contourArea of the blob (interior region of the contour).
    (contour,) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    assert len(objects) == 1
    assert objects[0].area_px == pytest.approx(cv2.contourArea(contour))


def test_extract_objects_particle_ids_are_one_indexed_and_sequential():
    mask = _make_mask_with_two_blobs()
    image = np.zeros((100, 120), dtype=np.uint8)
    backend = _StaticBackend(mask)
    objects = backend.extract_objects(image, mask, SegConfig(min_area=50))
    ids = [o.particle_id for o in objects]
    assert ids == [1, 2]
