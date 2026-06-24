# Architecture overview

SegSpy is a **microscope-generic, backend-pluggable** segmentation library for
HyperSpy. It converts a `Signal2D` into a list of particle objects, then measures
generic (particle-type-agnostic) morphology on each. The diagram below shows the
end-to-end flow.

```{mermaid}
flowchart LR
    IMG["HyperSpy Signal2D<br/>.dm4 / .tif / ndarray"]

    subgraph REG["Backend registry  (pluggable)"]
        direction TB
        TC["traditional_cv<br/>CLAHE → black-hat → threshold → morphology"]
        SAM["sam  (optional)<br/>SAM Auto-Mask + 5-stage filter"]
    end

    CFG["SegConfig<br/>from_signal → TEM/SEM"] --> REG
    IMG --> REG
    REG -->|segment| MASK["binary mask"]
    MASK --> EXTRACT["extract_objects<br/>(connected components)"]
    IMG --> EXTRACT
    EXTRACT --> OBJ["ParticleObject × N<br/>ROI + mask + calibration"]
    OBJ --> MORPH["measure_morphology<br/>area · Feret · perimeter · aspect · …"]
    MORPH --> OUT["per-particle metrics<br/>CSV + labelled PNG"]
```

Three ideas drive the design:

1. **Segmentation is swappable.** Every backend implements the same
   {class}`~SegSpy.backends.base.SegmentationBackend` interface and is reached
   through the {class}`~SegSpy.backends.registry.SegmentationRegistry`.
   `traditional_cv` ships with the core install; `sam` is an optional extra. A
   new algorithm is just a new registry entry — nothing else changes.
2. **Extract once, measure often.** `extract_objects` produces
   {class}`~SegSpy.core.ParticleObject` instances carrying the image ROI, mask
   ROI, and offset. `measure_morphology` runs on those objects, so you can
   re-measure or change metrics without re-segmenting.
3. **Calibration flows through.** When the input is a HyperSpy `Signal2D`, each
   object's ROI is itself a calibrated signal, and `get_scale_nm` reads the axis
   units so length metrics come out in nanometres.

> A deeper design-philosophy write-up, per-layer diagrams, and a "write your own
> backend" guide are planned for a future release.
