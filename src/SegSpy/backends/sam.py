"""SAM Auto-Mask segmentation backend with 5-stage post-filtering.

Migrated from ``a_em.segmentation.sam_automask``. ``segment_anything`` and
``torch`` are imported *lazily* inside :meth:`SAMSegmenter._load_model`, so this
module imports cleanly without the ``[sam]`` extra. The standout feature is
:func:`post_filter_masks` — a five-stage filter on SAM AMG output (confidence,
area, intensity, nested-dedup, edge exclusion) that runs without any heavy deps.
"""
import logging
import os

import numpy as np

from SegSpy.backends.base import SegmentationBackend
from SegSpy.config import SegConfig
from SegSpy.io import to_uint8

logger = logging.getLogger(__name__)

_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

_DEFAULT_CHECKPOINT_NAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


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


class SAMSegmenter(SegmentationBackend):
    """SAM Auto-Mask Generator backend with 5-stage post-filtering."""

    name: str = "sam"

    def __init__(self):
        self._sam = None
        self._mask_generator = None

    @classmethod
    def supports(cls, microscope_type: str, particle_type: str) -> bool:
        # SegSpy backends are microscope/particle agnostic.
        return True

    def _load_model(self, config: SegConfig):
        if self._sam is not None:
            return

        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as e:
            raise ImportError(
                "segment-anything is required. Install with: pip install SegSpy[sam]"
            ) from e

        device = config.sam_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = config.sam_checkpoint_path
        if not checkpoint_path:
            checkpoint_path = _DEFAULT_CHECKPOINT_NAMES.get(
                config.sam_model_type, "sam_vit_b_01ec64.pth"
            )

        if not os.path.exists(checkpoint_path):
            url = _CHECKPOINT_URLS.get(config.sam_model_type, _CHECKPOINT_URLS["vit_b"])
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. Download from: {url}"
            )

        logger.info("Loading SAM model %s on %s", config.sam_model_type, device)
        sam = sam_model_registry[config.sam_model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        self._sam = sam
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=config.sam_points_per_side,
            pred_iou_thresh=config.sam_pred_iou_thresh,
            stability_score_thresh=config.sam_stability_score_thresh,
            min_mask_region_area=100,
        )

    def segment(self, signal, config: SegConfig) -> np.ndarray:
        self._load_model(config)

        uint8_signal = to_uint8(signal)
        gray = uint8_signal.data
        h, w = gray.shape[:2]

        if gray.ndim == 2:
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif gray.ndim == 3 and gray.shape[2] == 1:
            rgb = np.repeat(gray, 3, axis=-1)
        else:
            rgb = gray

        logger.info("Running SAM AMG on image %s", rgb.shape[:2])
        masks = self._mask_generator.generate(rgb)
        logger.info("SAM generated %d candidate masks", len(masks))

        filtered = post_filter_masks(
            masks, original_gray=gray, image_shape=(h, w), config=config,
        )
        logger.info("Post-filtering retained %d masks", len(filtered))

        if not filtered:
            logger.warning("No masks passed post-filtering; returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)

        binary_mask = np.zeros((h, w), dtype=np.uint8)
        for m in filtered:
            binary_mask[m["segmentation"]] = 255

        return binary_mask
