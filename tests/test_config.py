"""Tests for SegSpy.config.SegConfig."""

import hyperspy.api as hs
import numpy as np

from SegSpy.config import SegConfig


def test_defaults_match_spec():
    c = SegConfig()
    assert c.microscope_type == "TEM"
    assert c.clahe_clip == 3.0
    assert c.clahe_tile == (8, 8)
    assert c.background_kernel == 25
    assert c.filter_type == "bilateral"
    assert c.backend == "traditional_cv"
    assert c.threshold_method == "adaptive"
    assert c.min_area == 100
    assert c.use_grabcut_refinement is True
    assert c.sam_model_type == "vit_b"
    assert c.sam_pred_iou_thresh == 0.88
    assert c.sam_stability_score_thresh == 0.95
    assert c.sam_min_area_ratio == 0.0005
    assert c.sam_max_area_ratio == 0.60
    assert c.sam_intensity_ratio == 0.85
    assert c.sam_edge_margin == 5
    assert c.compute_morphology is True
    assert c.output_dir == "output"


def test_field_overrides():
    c = SegConfig(min_area=500, backend="sam", threshold_method="otsu")
    assert c.min_area == 500
    assert c.backend == "sam"
    assert c.threshold_method == "otsu"


def test_from_signal_detects_tem_by_default():
    s = hs.signals.Signal2D(np.zeros((10, 10)))
    c = SegConfig.from_signal(s)
    assert c.microscope_type == "TEM"


def test_from_signal_detects_sem_from_metadata():
    s = hs.signals.Signal2D(np.zeros((10, 10)))
    s.metadata.set_item("Acquisition_instrument.SEM.beam_energy", 5.0)
    c = SegConfig.from_signal(s)
    assert c.microscope_type == "SEM"


def test_from_signal_detects_tem_from_metadata():
    s = hs.signals.Signal2D(np.zeros((10, 10)))
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    c = SegConfig.from_signal(s)
    assert c.microscope_type == "TEM"


def test_from_signal_applies_overrides():
    s = hs.signals.Signal2D(np.zeros((10, 10)))
    c = SegConfig.from_signal(s, min_area=777, backend="sam")
    assert c.microscope_type == "TEM"
    assert c.min_area == 777
    assert c.backend == "sam"
