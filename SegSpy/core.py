"""Core data model: :class:`ParticleObject`."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParticleObject:
    """A single particle extracted from an EM image.

    Carries the ROI image, its binary mask, bounding box, and a ``metrics`` dict
    populated by downstream measurement code (e.g. :mod:`SegSpy.morphology`).
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

    def __repr__(self) -> str:
        return f"<ParticleObject ID={self.particle_id} metrics={list(self.metrics.keys())}>"
