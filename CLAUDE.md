# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

SegSpy — headless, backend-pluggable EM image segmentation library for HyperSpy.
Turns a `Signal2D` into a list of `ParticleObject`s with calibrated morphology
metrics. Microscope-generic (TEM/SEM) and particle-agnostic. Python 3.10+, `src/`
layout. Migrated from a prior `a_em` project; domain-specific logic (e.g. soot
fractal dimension) was **intentionally excluded**.

## Commands

```bash
# Development install (core only)
pip install -e ".[dev]"
pre-commit install                  # ruff --fix + ruff-format on commit

# Optional SAM backend (segment-anything + torch) — not required to import/extend the lib
pip install -e ".[sam]"

# Tests — coverage is ON by default (pytest addopts)
pytest                              # -v --cov=SegSpy --cov-report=term-missing
pytest --no-cov                     # fast, no coverage

# Lint / format
ruff check src tests
ruff format src tests               # `ruff format --check src tests` in CI

# Docs — builds with -W (warnings-as-errors)
cd docs && sphinx-build -W -b html . _build/html

# Build / CLI
python -m build                     # dist (publish.yml runs on GitHub release)
SegSpy run image.dm4 --backend traditional_cv --output ./out
```

## Architecture

The 9 public symbols are exported from `src/SegSpy/__init__.py`. Flow:

`Signal2D` → `SegConfig.from_signal` → `SegmentationRegistry.get(name).segment(signal)` →
uint8 mask → `.extract_objects` → `ParticleObject[]` → `measure_morphology`.

- **`backends/`** — `SegmentationBackend` ABC (`base.py`); class-level
  `SegmentationRegistry` (`registry.py`); two implementations:
  `TraditionalCVSegmenter` (`traditional.py`, `name="traditional_cv"`) and
  `SAMSegmenter` (`sam.py`, `name="sam"`). Built-ins are **registered at import time**
  in `backends/__init__.py`.
- **`config.py`** — `SegConfig` dataclass (every backend/morphology knob);
  `from_signal` auto-detects TEM/SEM from HyperSpy metadata.
- **`core.py`** — `ParticleObject` (ROI + mask + bbox + `metrics` dict).
- **`io.py`** — `to_uint8`, `get_scale_nm` (HyperSpy-native helpers).
- **`morphology.py`** — `measure_morphology` (generic particle metrics via OpenCV).
- **`cli.py`** — `SegSpy run` entry point. Logic split into `run_segmentation` +
  `write_results` so both are unit-testable without argparse/IO.

## Gotchas

- **`SegmentationRegistry` is module-level global state.** Tests that touch it
  must snapshot/restore — see the `isolated_registry` fixture in
  `tests/test_backends_registry.py`. Clearing without restoring unregisters the
  built-ins for the rest of the session.
- **SAM deps import lazily** (`SAMSegmenter._load_model`). `backends/sam.py`
  imports cleanly without `[sam]`; `segment()` raises `ImportError` only when
  called. `post_filter_masks` is pure-numpy and fully tested without the extra.
- **Mask convention is `uint8` `0`/`255`** (not `0`/`1`) — the contract of `segment()`.
- **Centroids are `(y, x)`** order (matches mask indexing).
- **Two different scale fallbacks:** `io.get_scale_nm` raises `ValueError` on
  unknown units; `ParticleObject.get_physical_scale_nm` silently returns `1.0`
  when uncalibrated.
- **`*.dm4` is gitignored;** test fixtures use TIFF (`tifffile` is in `[dev]`).
- **ruff `line-length` is 100 but `E501` is ignored;** also ignores `E741`,
  `N806`, `N817`, `N999`. `__init__.py` ignores `F401` for re-exports.
- **`0.1.0` is not yet released (no git tags);** `publish.yml` triggers on GitHub
  release and fills release notes from CHANGELOG.

## Workflow

- New algorithm → subclass `SegmentationBackend`, set a unique `name`, register in
  `backends/__init__.py`. Nothing else changes.
- Log changes under `[Unreleased]` in `CHANGELOG.md`.
- PR gate: `pytest` + `ruff check src tests && ruff format --check src tests` pass;
  CI runs on 3.10/3.11/3.12.
