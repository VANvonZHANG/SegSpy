# Command-line interface

The `SegSpy` CLI segments an image file end-to-end and writes results to disk:

```bash
SegSpy run image.dm4 --backend traditional_cv --output ./out
```

This produces:

- `out/image_particles.csv` — one row per particle with all morphology metrics.
- `out/image_labeled.png` — the source-sized canvas with each particle's mask
  drawn in a distinct colour.

## Options

| Option | Default | Purpose |
|--------|---------|---------|
| `file` | *(required)* | Path to an EM image readable by HyperSpy (`.dm4`, `.tif`, …) |
| `--backend` | `traditional_cv` | Segmentation backend name (`traditional_cv`, `sam`) |
| `--output` | `output` | Output directory (created if missing) |
| `--min-area` | `100` | Minimum particle contour area in pixels |
| `--no-grabcut` | *(flag)* | Disable GrabCut refinement (traditional backend) |

The CLI auto-detects TEM/SEM from the file's metadata (same logic as
{class}`~SegSpy.config.SegConfig.from_signal`). For anything beyond these flags — custom
SAM thresholds, CLAHE tuning — use the library API from the {doc}`quickstart <quickstart>`.
