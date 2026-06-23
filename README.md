# SegSpy

[![CI](https://github.com/VANvonZHANG/SegSpy/actions/workflows/ci.yml/badge.svg)](https://github.com/VANvonZHANG/SegSpy/actions/workflows/ci.yml)
[![Lint](https://github.com/VANvonZHANG/SegSpy/actions/workflows/lint.yml/badge.svg)](https://github.com/VANvonZHANG/SegSpy/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/VANvonZHANG/SegSpy/blob/main/LICENSE)

Headless, backend-pluggable EM image segmentation library for HyperSpy.

SegSpy turns a HyperSpy `Signal2D` into a list of particle objects. It is
microscope-generic (TEM/SEM) and particle-type-agnostic, shipping multiple
interchangeable segmentation backends behind a uniform registry:

- `traditional_cv` — CLAHE → black-hat → threshold → morphology → GrabCut refinement
- `sam` — Meta SAM Auto-Mask Generator with a 5-stage post-filter

## Installation

```bash
pip install SegSpy
```

For the SAM backend, install the optional extra:

```bash
pip install "SegSpy[sam]"   # adds segment-anything + torch
```

For development:

```bash
git clone https://github.com/VANvonZHANG/SegSpy.git
cd SegSpy
pip install -e ".[dev]"
pre-commit install
```

## Quick start

```python
import hyperspy.api as hs
from SegSpy import SegConfig, SegmentationRegistry, measure_morphology

signal = hs.load("image.dm4")
backend = SegmentationRegistry.get("traditional_cv")
mask = backend.segment(signal, SegConfig())
objects = backend.extract_objects(signal, mask, SegConfig())
for obj in objects:
    measure_morphology(obj, scale_nm=1.0)
```

## CLI

```bash
SegSpy run image.dm4 --backend traditional_cv --output ./out
```

Writes `<stem>_particles.csv` (per-particle morphology metrics) and
`<stem>_labeled.png` (a labelled mask) to the output directory.

## Status

**v0.1** — pure library + CLI, no GUI (deferred to a later release). See the
[CHANGELOG](CHANGELOG.md) for details. Contributions welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).
