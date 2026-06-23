"""Minimal example: use SegSpy as a library (no CLI) on a DM4 image.

Run with the path to any HyperSpy-readable EM image::

    python examples/segment_dm4.py /path/to/image.dm4
"""
import sys

import hyperspy.api as hs

from SegSpy import SegConfig, SegmentationRegistry, get_scale_nm, measure_morphology


def main(path: str) -> None:
    signal = hs.load(path)
    if isinstance(signal, list):
        signal = signal[0]

    config = SegConfig.from_signal(signal, min_area=100, use_grabcut_refinement=False)
    scale_nm = get_scale_nm(signal)

    backend = SegmentationRegistry.get(config.backend)
    mask = backend.segment(signal, config)
    objects = backend.extract_objects(signal, mask, config)

    for obj in objects:
        measure_morphology(obj, scale_nm=scale_nm)

    print(f"{len(objects)} particle(s) found in {path}")
    for obj in objects[:5]:
        m = obj.metrics
        print(
            f"  #{obj.particle_id}: area={m['area_px']:.0f}px "
            f"equiv_diam={m['equivalent_diameter_nm']:.1f}nm "
            f"roundness={m['roundness']:.2f} solidity={m['solidity']:.2f}"
        )
    if len(objects) > 5:
        print(f"  ... ({len(objects) - 5} more)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <image>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
