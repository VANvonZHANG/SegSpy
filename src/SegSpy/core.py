"""Core data model: :class:`ParticleObject`."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParticleObject:
    """A single particle extracted from an EM image.

    Carries the ROI image, its binary mask, bounding box, and a ``metrics`` dict
    populated by :func:`SegSpy.morphology.measure_morphology` (generic) and/or by
    downstream consumers (domain-specific, e.g. soot fractal dimension).
    """

    particle_id: int
    image_roi: Any
    mask_roi: Any
    offset: tuple[int, int] = (0, 0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: float = 0.0
    metrics: dict = field(default_factory=dict)

    def add_metric(self, name: str, value: Any) -> None:
        """Record a named metric on this particle."""
        self.metrics[name] = value

    def get_physical_scale_nm(self) -> float:
        """Return nm/pixel from the ROI's calibration if available.

        Reads the HyperSpy axis calibration when ``image_roi`` is a
        ``Signal2D``; otherwise returns ``1.0`` (uncalibrated / numpy array).
        """
        try:
            import hyperspy.api as hs

            if isinstance(self.image_roi, hs.signals.Signal2D):
                scale = self.image_roi.axes_manager[0].scale
                units = self.image_roi.axes_manager[0].units
                if units in ("um", "µm", "micrometer"):
                    return scale * 1000.0
                if units in ("nm", "nanometer"):
                    return scale
                if units in ("mm",):
                    return scale * 1e6
                return 1.0
        except ImportError:
            pass
        return 1.0

    def __repr__(self) -> str:
        return f"<ParticleObject ID={self.particle_id} metrics={list(self.metrics.keys())}>"
