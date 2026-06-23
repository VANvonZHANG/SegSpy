"""Abstract base class for segmentation backends."""
from abc import ABC, abstractmethod

import numpy as np

from SegSpy.config import SegConfig


class SegmentationBackend(ABC):
    """Abstract base class for all segmentation algorithms."""

    name: str = "abstract"

    @abstractmethod
    def segment(self, signal, config: SegConfig) -> np.ndarray:
        """Segment a signal and return a binary mask (0/255).

        Args:
            signal: HyperSpy ``Signal2D`` or numpy array containing the image.
            config: SegSpy configuration with segmentation parameters.

        Returns:
            Binary mask as a ``uint8`` array with values 0 or 255.
        """
        ...

    @classmethod
    def supports(cls, microscope_type: str, particle_type: str) -> bool:
        """Check if this backend supports the given microscope/particle combination.

        Returns ``False`` by default. Subclasses override to opt-in. SegSpy's
        built-in backends are particle-agnostic, so they return ``True``.
        """
        return False
