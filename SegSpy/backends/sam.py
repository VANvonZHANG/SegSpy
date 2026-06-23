"""SAM Auto-Mask 5-stage post-filter.

:func:`post_filter_masks` applies five filtering stages to SAM Auto-Mask
Generator output (confidence, area, intensity, nested-dedup, edge exclusion).
It is dependency-light (numpy only) and is SegSpy's standout feature. The
:class:`SAMSegmenter` backend that consumes it lands in a later commit.
"""
import numpy as np


def _is_nested(small_mask, large_mask, threshold=0.8):
    """Return True if ``small_mask`` is mostly contained within ``large_mask``."""
    intersection = np.logical_and(small_mask, large_mask).sum()
    containment = intersection / (small_mask.sum() + 1e-8)
    return bool(containment > threshold)


def post_filter_masks(masks, original_gray, image_shape, config):
    """Apply 5-stage post-filtering to SAM AMG output.

    Stages:
        1. Confidence — drop low ``predicted_iou`` / ``stability_score``.
        2. Area — keep masks within [min_area_ratio, max_area_ratio] of the image.
        3. Intensity — keep only masks darker than ``intensity_ratio`` * global mean.
        4. Nested resolution — drop masks contained within an already-kept mask.
        5. Edge exclusion — drop masks touching the image border.

    Args:
        masks: list of AMG dicts (``segmentation`` bool HxW, ``area``, ``bbox``
            ``[x, y, w, h]``, ``predicted_iou``, ``stability_score``).
        original_gray: original grayscale image (H, W) uint8, before enhancement.
        image_shape: ``(H, W)`` of the image.
        config: :class:`SegConfig` (or compatible) with SAM filter params.

    Returns:
        The filtered list of mask dicts.
    """
    h, w = image_shape
    image_area = h * w

    # Stage 1: Confidence filtering
    filtered = [
        m for m in masks
        if m.get("predicted_iou", 0) >= config.sam_pred_iou_thresh
        and m.get("stability_score", 0) >= config.sam_stability_score_thresh
    ]

    # Stage 2: Area filtering
    min_area = image_area * config.sam_min_area_ratio
    max_area = image_area * config.sam_max_area_ratio
    filtered = [m for m in filtered if min_area <= m["area"] <= max_area]

    # Stage 3: Intensity filtering (keep dark particles)
    global_mean = original_gray.mean()
    filtered = [
        m for m in filtered
        if original_gray[m["segmentation"]].mean() < global_mean * config.sam_intensity_ratio
    ]

    # Stage 4: Nested mask resolution (largest first, drop contained duplicates)
    sorted_by_area = sorted(filtered, key=lambda m: m["area"], reverse=True)
    selected = []
    for m in sorted_by_area:
        seg = m["segmentation"]
        if any(_is_nested(seg, sel["segmentation"]) for sel in selected):
            continue
        selected.append(m)
    filtered = selected

    # Stage 5: Edge exclusion
    edge_margin = config.sam_edge_margin
    filtered = [
        m for m in filtered
        if not (
            m["bbox"][0] < edge_margin
            or m["bbox"][1] < edge_margin
            or (m["bbox"][0] + m["bbox"][2]) > (w - edge_margin)
            or (m["bbox"][1] + m["bbox"][3]) > (h - edge_margin)
        )
    ]

    return filtered
