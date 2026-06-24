# SegSpy

SegSpy is a headless, backend-pluggable segmentation library for electron-microscopy
images in the HyperSpy ecosystem. It turns a `Signal2D` into a list of particle
objects with calibrated morphology metrics — microscope-generic (TEM/SEM) and
particle-type-agnostic.

## Features

- **Pluggable backends** — swap segmentation algorithms behind one registry:
  `traditional_cv` (CLAHE → black-hat → threshold → morphology → GrabCut) and `sam`
  (Meta SAM Auto-Mask Generator with a 5-stage post-filter).
- **Calibration-aware** — reads HyperSpy axis scales, so lengths come out in nanometres.
- **Generic morphology** — area, equivalent diameter, Feret diameters, perimeter,
  aspect ratio, convexity, roundness, solidity, centroid — no domain-specific assumptions.
- **Headless** — a clean Python API and a `SegSpy run` CLI; no GUI required.

## Quick navigation

```{toctree}
:maxdepth: 2

user-guide/index
api-reference/index
```
