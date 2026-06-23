"""Segmentation backends and the central registry.

Built-in backends are registered here at import time:

- :class:`~SegSpy.backends.traditional.TraditionalCVSegmenter` → ``traditional_cv``
- :class:`~SegSpy.backends.sam.SAMSegmenter` → ``sam``
"""

from SegSpy.backends.base import SegmentationBackend
from SegSpy.backends.registry import SegmentationRegistry
from SegSpy.backends.sam import SAMSegmenter
from SegSpy.backends.traditional import TraditionalCVSegmenter

SegmentationRegistry.register(TraditionalCVSegmenter)
SegmentationRegistry.register(SAMSegmenter)

__all__ = [
    "SegmentationBackend",
    "SegmentationRegistry",
    "TraditionalCVSegmenter",
    "SAMSegmenter",
]
