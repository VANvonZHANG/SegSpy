# Quickstart

This walks through the core library flow: load an image, segment it, extract
particle objects, and measure morphology. The same flow underlies the
{doc}`CLI <cli>`.

```python
import hyperspy.api as hs
from SegSpy import SegConfig, SegmentationRegistry, get_scale_nm, measure_morphology

signal = hs.load("image.dm4")

# 1. Configure (TEM/SEM is auto-detected from the signal metadata)
config = SegConfig.from_signal(signal, min_area=100)

# 2. Pick a backend through the registry and segment -> binary mask
backend = SegmentationRegistry.get(config.backend)
mask = backend.segment(signal, config)

# 3. Extract one ParticleObject per detected particle
objects = backend.extract_objects(signal, mask, config)

# 4. Measure morphology (calibrated to nanometres via the signal's scale)
scale_nm = get_scale_nm(signal)
for obj in objects:
    measure_morphology(obj, scale_nm=scale_nm)

print(f"{len(objects)} particle(s) found")
for obj in objects[:5]:
    m = obj.metrics
    print(f"  #{obj.particle_id}: area={m['area_px']:.0f}px "
          f"equiv_diam={m['equivalent_diameter_nm']:.1f}nm "
          f"roundness={m['roundness']:.2f}")
```

## How the steps fit together

1. **`SegConfig.from_signal`** reads HyperSpy metadata to set `microscope_type`
   (a `SEM` block under `Acquisition_instrument` → SEM, otherwise TEM). Pass
   overrides like `min_area` directly.
2. **`SegmentationRegistry.get`** returns a backend by name — `traditional_cv`
   by default, or `sam` if you installed the extra and set `backend="sam"`.
3. **`segment`** returns a binary mask (uint8, 0/255).
4. **`extract_objects`** runs connected-component analysis on the mask and
   returns one {class}`~SegSpy.core.ParticleObject` per contour above `min_area`.
   Each object carries its image ROI, mask ROI, and offset.
5. **`measure_morphology`** computes size/shape metrics from each mask and stores
   them in `obj.metrics`. Lengths are reported in both pixels (`_px`) and
   nanometres (`_nm`).

To swap backends, change one line — set `config.backend = "sam"` (with the extra
installed). Everything downstream stays the same.
