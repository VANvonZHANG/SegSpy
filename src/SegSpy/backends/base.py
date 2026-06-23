"""Abstract base class for segmentation backends."""
from abc import ABC, abstractmethod

import numpy as np

from SegSpy.config import SegConfig
from SegSpy.core import ParticleObject


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

    def extract_objects(
        self, original_signal, mask: np.ndarray, config: SegConfig
    ) -> list[ParticleObject]:
        """Extract individual particle objects from a binary mask.

        Performs connected-component analysis and returns one
        :class:`~SegSpy.core.ParticleObject` per detected contour that clears
        ``config.min_area``. Each object's ROI/mask is sliced out of the
        original signal; when the input is a HyperSpy ``Signal2D`` the ROI is
        itself a calibrated signal (via ``isig``), so downstream calibration
        works transparently.
        """
        import cv2

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        objects: list[ParticleObject] = []
        h, w = mask.shape

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < config.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            full = (
                original_signal
                if isinstance(original_signal, np.ndarray)
                else original_signal.data
            )
            roi_data = full[y : y + bh, x : x + bw]
            mask_roi = mask[y : y + bh, x : x + bw].copy()

            local_cnt = cnt - [x, y]
            temp_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(temp_mask, [local_cnt], -1, 255, -1)
            mask_roi = cv2.bitwise_and(mask_roi, temp_mask)

            try:
                roi_signal = original_signal.isig[x : x + bw, y : y + bh]
            except (ImportError, AttributeError):
                roi_signal = roi_data

            obj = ParticleObject(
                particle_id=i + 1,
                image_roi=roi_signal,
                mask_roi=mask_roi,
                offset=(y, x),
                bbox=(x, y, bw, bh),
                area_px=area,
            )
            objects.append(obj)

        return objects
