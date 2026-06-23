"""Configuration dataclass for segmentation + generic morphology.

Holds every knob the built-in backends and the morphology module read.
Domain-specific (e.g. soot analysis) and reporting fields are intentionally NOT
here — those live in downstream consumers such as ``a_em``.
"""
from dataclasses import dataclass


@dataclass
class SegConfig:
    """Segmentation + morphology configuration.

    All values default to sensible EM-image defaults; override via the constructor.
    """

    microscope_type: str = "TEM"

    # --- Preprocessing ---
    clahe_clip: float = 3.0
    clahe_tile: tuple[int, int] = (8, 8)
    background_kernel: int = 25
    filter_type: str = "bilateral"

    # --- Segmentation ---
    backend: str = "traditional_cv"
    threshold_method: str = "adaptive"
    min_area: int = 100
    use_grabcut_refinement: bool = True

    # --- SAM backend ---
    sam_model_type: str = "vit_b"
    sam_checkpoint_path: str = ""
    sam_device: str = "auto"
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    sam_min_area_ratio: float = 0.0005
    sam_max_area_ratio: float = 0.60
    sam_intensity_ratio: float = 0.85
    sam_edge_margin: int = 5

    # --- Morphology ---
    compute_morphology: bool = True

    # --- Output ---
    output_dir: str = "output"
