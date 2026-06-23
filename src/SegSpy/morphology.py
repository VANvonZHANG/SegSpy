"""Generic (particle-type-agnostic) morphology measurements.

:func:`measure_morphology` computes size/shape metrics for a single
:class:`~SegSpy.core.ParticleObject` from its ``mask_roi`` using OpenCV contour
operations. It is deliberately free of any domain-specific (e.g. soot fractal)
logic — that belongs in downstream consumers.
"""

import cv2
import numpy as np

from SegSpy.core import ParticleObject


def measure_morphology(obj: ParticleObject, scale_nm: float = 1.0) -> dict:
    """Compute particle-type-agnostic morphology metrics from ``obj.mask_roi``.

    The largest external contour of the mask is used. Results are written into
    ``obj.metrics`` and also returned. Length metrics are reported both in pixels
    (``_px``) and, where applicable, in nanometres (``_nm`` = px × ``scale_nm``).

    Keys returned:
        area_px, equivalent_diameter_px, equivalent_diameter_nm,
        feret_min_px, feret_max_px, feret_max_nm, perimeter_px,
        aspect_ratio, convexity, roundness, solidity, centroid_yx
    """
    mask = np.asarray(obj.mask_roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metrics = _empty_metrics()
    if not contours:
        obj.metrics.update(metrics)
        return metrics

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    # Equivalent (equal-area-circle) diameter.
    equivalent_diameter_px = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0

    # Min / max Feret diameters via the minimum-area bounding rectangle.
    feret_min_px, feret_max_px = _feret_diameters(cnt)
    aspect_ratio = feret_max_px / feret_min_px if feret_min_px > 0 else 0.0

    # Convex-hull based shape factors.
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    roundness = 4.0 * area / (np.pi * feret_max_px**2) if feret_max_px > 0 else 0.0

    # Centroid (moment-based), in (y, x) order to match mask indexing.
    m = cv2.moments(cnt)
    if m["m00"] != 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
    else:
        cx = cy = 0.0

    metrics = {
        "area_px": area,
        "equivalent_diameter_px": float(equivalent_diameter_px),
        "equivalent_diameter_nm": float(equivalent_diameter_px * scale_nm),
        "feret_min_px": float(feret_min_px),
        "feret_max_px": float(feret_max_px),
        "feret_max_nm": float(feret_max_px * scale_nm),
        "perimeter_px": perimeter,
        "aspect_ratio": float(aspect_ratio),
        "convexity": float(convexity),
        "roundness": float(roundness),
        "solidity": float(solidity),
        "centroid_yx": (float(cy), float(cx)),
    }
    obj.metrics.update(metrics)
    return metrics


def _feret_diameters(cnt) -> tuple[float, float]:
    """Approximate min/max Feret diameters from the minimum-area bounding rect.

    ``cv2.minAreaRect`` returns the smallest rotated rectangle enclosing the
    contour; its two side lengths approximate the minimum and maximum caliper
    (Feret) diameters.
    """
    (_cx, _cy), (w, h), _angle = cv2.minAreaRect(cnt)
    w = float(w)
    h = float(h)
    return (min(w, h), max(w, h))


def _empty_metrics() -> dict:
    return {
        "area_px": 0.0,
        "equivalent_diameter_px": 0.0,
        "equivalent_diameter_nm": 0.0,
        "feret_min_px": 0.0,
        "feret_max_px": 0.0,
        "feret_max_nm": 0.0,
        "perimeter_px": 0.0,
        "aspect_ratio": 0.0,
        "convexity": 0.0,
        "roundness": 0.0,
        "solidity": 0.0,
        "centroid_yx": (0.0, 0.0),
    }
