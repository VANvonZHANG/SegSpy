# Configuration

All segmentation and morphology knobs live in a single dataclass,
{class}`~SegSpy.SegConfig`. Construct it directly or via the signal-aware factory:

```python
from SegSpy import SegConfig
config = SegConfig.from_signal(signal, min_area=200, threshold_method="otsu")
```

## Auto-detecting TEM/SEM

`from_signal` inspects HyperSpy metadata: if an `Acquisition_instrument.SEM`
block is present the microscope type is set to `SEM`, otherwise `TEM` (the
default). This selects sensible preprocessing defaults; override
`microscope_type` explicitly if your metadata is absent or non-standard.

## Key tuning fields

| Field | Default | Purpose |
|-------|---------|---------|
| `min_area` | `100` | Drop particles whose contour area (px) is below this |
| `threshold_method` | `"adaptive"` | `"adaptive"` or `"otsu"` binarisation |
| `clahe_clip` / `clahe_tile` | `3.0` / `(8, 8)` | CLAHE contrast strength / tile grid |
| `background_kernel` | `25` | Black-hat morphological kernel size (px) |
| `filter_type` | `"bilateral"` | Pre-segmentation noise filter |
| `use_grabcut_refinement` | `True` | Refine masks with GrabCut (traditional backend) |

### SAM-only fields

When `backend = "sam"`, these control the Auto-Mask Generator and post-filter:

| Field | Default | Purpose |
|-------|---------|---------|
| `sam_model_type` | `"vit_b"` | SAM model variant |
| `sam_checkpoint_path` | `""` | Path to the SAM checkpoint |
| `sam_device` | `"auto"` | Device (`auto`/`cuda`/`cpu`) |
| `sam_points_per_side` | `32` | Grid density of prompt points |
| `sam_pred_iou_thresh` | `0.88` | Minimum predicted mask quality |
| `sam_stability_score_thresh` | `0.95` | Mask stability threshold |
| `sam_min_area_ratio` / `sam_max_area_ratio` | `0.0005` / `0.60` | Drop masks outside this fraction of the image |
| `sam_intensity_ratio` | `0.85` | Foreground-intensity consistency gate |
| `sam_edge_margin` | `5` | Drop masks touching the image border (px) |

For the full field list, see the {doc}`API reference <../api-reference/index>`.
