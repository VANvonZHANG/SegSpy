# Tutorial: segment a DM4 image

This tutorial runs SegSpy as a library on a single DM4 image and prints a
summary of the detected particles. It is the example shipped in the repository:

```{literalinclude} ../../examples/segment_dm4.py
:language: python
:linenos:
```

## Run it

```bash
python examples/segment_dm4.py /path/to/image.dm4
```

You'll see a per-particle summary like:

```text
12 particle(s) found in /path/to/image.dm4
  #1: area=312px equiv_diam=18.4nm roundness=0.71 solidity=0.94
  #2: area=205px equiv_diam=15.1nm roundness=0.83 solidity=0.97
  ...
```

## Reading the metrics

`measure_morphology` writes these keys into each object's `metrics` dict (length
metrics are reported in pixels `_px` and nanometres `_nm`):

| Key | Meaning |
|-----|---------|
| `area_px` | Contour area (px) |
| `equivalent_diameter_px` / `equivalent_diameter_nm` | Diameter of the equal-area circle |
| `feret_min_px` / `feret_max_px` / `feret_max_nm` | Min/max caliper (Feret) diameters |
| `perimeter_px` | Contour perimeter (px) |
| `aspect_ratio` | `feret_max / feret_min` |
| `convexity` | Convex-hull perimeter / contour perimeter |
| `roundness` | `4·area / (π·feret_max²)` (1 = circular) |
| `solidity` | `area / convex-hull area` (1 = convex) |
| `centroid_yx` | Moment-based centroid in (row, col) |

To export every particle to CSV instead of printing, use the {doc}`CLI <../user-guide/cli>`,
which writes the same metrics to `<stem>_particles.csv`.
