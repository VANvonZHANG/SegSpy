# Installation

SegSpy requires **Python ≥ 3.10**.

## Install the library

```bash
pip install SegSpy
```

## SAM backend (optional)

The `sam` backend uses Meta's Segment Anything Model. Install the extra to add
`segment-anything` and `torch`:

```bash
pip install "SegSpy[sam]"
```

SegSpy imports and runs fine without this extra — only `SAMSegmenter.segment`
needs it. The `traditional_cv` backend works out of the box.

## Development install

```bash
git clone https://github.com/VANvonZHANG/SegSpy.git
cd SegSpy
pip install -e ".[dev]"
pre-commit install
```

The `[dev]` extra adds `pytest`, `pytest-cov`, `ruff`, `pre-commit`, and
`tifffile` (used by the EM TIFF test fixtures).
