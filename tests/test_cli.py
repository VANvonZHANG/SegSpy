"""Tests for SegSpy.cli (run subcommand) and the core run_segmentation helper."""
import csv

import hyperspy.api as hs
import numpy as np

from SegSpy.cli import main, run_segmentation, write_results
from SegSpy.config import SegConfig


def _blob_signal(size=200):
    """Bright image with a large high-contrast dark square blob in the centre."""
    data = np.ones((size, size), dtype=np.float32) * 200.0
    half = size // 4
    data[size // 2 - half : size // 2 + half, size // 2 - half : size // 2 + half] = 40.0
    s = hs.signals.Signal2D(data)
    s.axes_manager[0].scale = 1.0
    s.axes_manager[0].units = "nm"
    s.axes_manager[1].scale = 1.0
    s.axes_manager[1].units = "nm"
    return s


# --- core helper: run_segmentation ---


def test_run_segmentation_returns_objects_with_morphology():
    signal = _blob_signal()
    config = SegConfig(min_area=50, use_grabcut_refinement=False)
    objects = run_segmentation(signal, config)
    assert len(objects) >= 1
    for obj in objects:
        assert "area_px" in obj.metrics
        assert "equivalent_diameter_px" in obj.metrics


def test_run_segmentation_respects_min_area():
    signal = _blob_signal()
    config = SegConfig(min_area=10_000, use_grabcut_refinement=False)
    objects = run_segmentation(signal, config)
    for obj in objects:
        assert obj.area_px >= 10_000


# --- write_results ---


def test_write_results_creates_csv_and_png(tmp_path):
    signal = _blob_signal()
    config = SegConfig(min_area=50, use_grabcut_refinement=False)
    objects = run_segmentation(signal, config)
    csv_path, png_path = write_results(
        objects, str(tmp_path), "demo", signal.data.shape[:2]
    )
    assert csv_path.exists()
    assert png_path.exists()
    assert csv_path.name == "demo_particles.csv"
    assert png_path.name == "demo_labeled.png"
    # CSV has a header + at least one data row.
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 2


# --- end-to-end CLI via main() ---


def _save_blob_tif(path):
    signal = _blob_signal()
    signal.save(str(path))
    return path


def test_main_run_writes_csv_and_png(tmp_path):
    tif = tmp_path / "blob.tif"
    _save_blob_tif(tif)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "run",
            str(tif),
            "--backend",
            "traditional_cv",
            "--output",
            str(out_dir),
            "--no-grabcut",
        ]
    )
    assert rc == 0
    assert (out_dir / "blob_particles.csv").exists()
    assert (out_dir / "blob_labeled.png").exists()
    with open(out_dir / "blob_particles.csv") as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 2  # header + >=1 particle


def test_main_run_unknown_backend_returns_nonzero(tmp_path):
    tif = tmp_path / "blob.tif"
    _save_blob_tif(tif)
    rc = main(
        [
            "run",
            str(tif),
            "--backend",
            "does_not_exist",
            "--output",
            str(tmp_path / "out"),
            "--no-grabcut",
        ]
    )
    assert rc != 0


def test_main_no_subcommand_prints_usage():
    rc = main([])
    assert rc != 0
