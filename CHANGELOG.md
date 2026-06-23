# Changelog

## [Unreleased]

## [0.1.0] - 2026-06-23

### Added
- Initial release of SegSpy, a headless, backend-pluggable EM image
  segmentation library for HyperSpy.
- `ParticleObject` data model with calibration-aware `get_physical_scale_nm`.
- HyperSpy-native I/O helpers: `to_uint8`, `get_scale_nm`.
- `SegConfig` dataclass with `from_signal` TEM/SEM auto-detection.
- Segmentation backends behind a uniform `SegmentationBackend` ABC + registry:
  - `TraditionalCVSegmenter` (CLAHE → black-hat → threshold → morphology → GrabCut)
  - `SAMSegmenter` (Meta SAM Auto-Mask Generator + 5-stage `post_filter_masks`)
- Generic morphology measurements: area, equivalent diameter, Feret (min/max),
  perimeter, aspect ratio, convexity, roundness, solidity, centroid (px + nm).
- `SegSpy run` CLI writing per-particle CSV + labelled PNG.
- `src/` layout, MIT license, 72 unit tests.
