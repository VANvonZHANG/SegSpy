"""SegSpy — headless, backend-pluggable EM image segmentation library for HyperSpy.

Public API:

    from SegSpy import SegConfig, SegmentationRegistry, measure_morphology

    backend = SegmentationRegistry.get("traditional_cv")
    mask = backend.segment(signal, SegConfig())
    objects = backend.extract_objects(signal, mask, SegConfig())
    for obj in objects:
        measure_morphology(obj, scale_nm=1.0)
"""

from SegSpy.backends import (
    SAMSegmenter,
    SegmentationBackend,
    SegmentationRegistry,
    TraditionalCVSegmenter,
)
from SegSpy.config import SegConfig
from SegSpy.core import ParticleObject
from SegSpy.io import get_scale_nm, to_uint8
from SegSpy.morphology import measure_morphology

__version__ = "0.1.0"

__all__ = [
    "SegConfig",
    "ParticleObject",
    "to_uint8",
    "get_scale_nm",
    "measure_morphology",
    "SegmentationBackend",
    "SegmentationRegistry",
    "TraditionalCVSegmenter",
    "SAMSegmenter",
]
