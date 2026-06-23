"""Command-line interface: ``SegSpy run <file> [--backend] [--output]``.

Loads an EM image via HyperSpy, runs a segmentation backend, extracts particle
objects, measures generic morphology, and writes a per-particle metrics CSV and
a labelled-mask PNG. The core logic is split into :func:`run_segmentation`
(signal in, objects out) and :func:`write_results` so both are reusable and
unit-testable without going through ``argparse``/file I/O.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import hyperspy.api as hs
import numpy as np

from SegSpy.backends import SegmentationRegistry
from SegSpy.config import SegConfig
from SegSpy.morphology import measure_morphology

# Distinct BGR colours for labelling successive particles on the PNG.
_LABEL_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
]


def run_segmentation(signal, config: SegConfig):
    """Segment ``signal``, extract particles, and measure morphology.

    Returns the list of :class:`~SegSpy.core.ParticleObject`. Morphology is
    measured only when ``config.compute_morphology`` is true.
    """
    backend = SegmentationRegistry.get(config.backend)
    mask = backend.segment(signal, config)
    objects = backend.extract_objects(signal, mask, config)
    if config.compute_morphology:
        for obj in objects:
            measure_morphology(obj, scale_nm=1.0)
    return objects


def write_results(objects, output_dir, stem, image_shape):
    """Write ``{stem}_particles.csv`` and ``{stem}_labeled.png`` to ``output_dir``.

    ``image_shape`` is the ``(H, W)`` of the source image, used to size the
    labelled-mask canvas. Returns ``(csv_path, png_path)`` as :class:`Path`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{stem}_particles.csv"
    png_path = out / f"{stem}_labeled.png"

    rows = [_flatten_metrics(obj) for obj in objects]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_labeled_png(objects, png_path, image_shape)
    return csv_path, png_path


def _flatten_metrics(obj):
    """Flatten a ParticleObject into a CSV row (id + scalar metrics)."""
    row = {"particle_id": obj.particle_id, "area_px": obj.area_px}
    for key, value in obj.metrics.items():
        if isinstance(value, tuple | list | np.ndarray):
            value = ";".join(f"{float(v):.4f}" for v in np.asarray(value).ravel())
        row[key] = value
    return row


def _write_labeled_png(objects, png_path, image_shape):
    h, w = image_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i, obj in enumerate(objects):
        color = _LABEL_COLORS[i % len(_LABEL_COLORS)]
        y, x = obj.offset
        mh, mw = obj.mask_roi.shape[:2]
        y2, x2 = min(y + mh, h), min(x + mw, w)
        region = obj.mask_roi[: y2 - y, : x2 - x] > 0
        canvas[y:y2, x:x2][region] = color
    cv2.imwrite(str(png_path), canvas)


def _build_config(args, signal) -> SegConfig:
    return SegConfig.from_signal(
        signal,
        backend=args.backend,
        min_area=args.min_area,
        use_grabcut_refinement=not args.no_grabcut,
    )


def _cmd_run(args) -> int:
    signal = hs.load(args.file)
    if isinstance(signal, list):
        signal = signal[0]
    config = _build_config(args, signal)
    try:
        objects = run_segmentation(signal, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stem = Path(args.file).stem
    write_results(objects, args.output, stem, signal.data.shape[:2])
    print(
        f"Segmented {len(objects)} particle(s) -> {Path(args.output) / (stem + '_particles.csv')}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="SegSpy",
        description="Headless EM image segmentation for HyperSpy.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Segment an image file and export results.")
    run_p.add_argument("file", help="Path to an EM image readable by HyperSpy.")
    run_p.add_argument(
        "--backend",
        default="traditional_cv",
        help="Segmentation backend name (default: traditional_cv).",
    )
    run_p.add_argument("--output", default="output", help="Output directory (default: output).")
    run_p.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum particle contour area in pixels (default: 100).",
    )
    run_p.add_argument(
        "--no-grabcut",
        action="store_true",
        help="Disable GrabCut refinement (traditional backend).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
