# SegSpy

Headless, backend-pluggable EM image segmentation library for HyperSpy.

SegSpy turns a HyperSpy `Signal2D` into a list of particle objects. It is
microscope-generic (TEM/SEM) and particle-type-agnostic, shipping multiple
interchangeable segmentation backends behind a uniform registry:

- `traditional_cv` — CLAHE → black-hat → threshold → morphology → GrabCut refinement
- `sam` — Meta SAM Auto-Mask Generator with a 5-stage post-filter

## Quick start

```python
import hyperspy.api as hs
from SegSpy import SegConfig, SegmentationRegistry

signal = hs.load("image.dm4")
backend = SegmentationRegistry.get("traditional_cv")
mask = backend.segment(signal, SegConfig())
objects = backend.extract_objects(signal, mask, SegConfig())
```

## CLI

```bash
SegSpy run image.dm4 --backend traditional_cv --output ./out
```

Status: **v0.1** — pure library + CLI, no GUI.
