# API reference

SegSpy's public API is re-exported from `SegSpy.__init__`. The nine symbols
below are all available via `from SegSpy import ...`.

## Configuration

```{eval-rst}
.. autoclass:: SegSpy.config.SegConfig
   :members:
   :undoc-members:
```

## Data model

```{eval-rst}
.. autoclass:: SegSpy.core.ParticleObject
   :members:
   :undoc-members:
```

## I/O helpers

```{eval-rst}
.. autofunction:: SegSpy.io.to_uint8
.. autofunction:: SegSpy.io.get_scale_nm
```

## Morphology

```{eval-rst}
.. autofunction:: SegSpy.morphology.measure_morphology
```

## Backends

```{eval-rst}
.. autoclass:: SegSpy.backends.base.SegmentationBackend
   :members:

.. autoclass:: SegSpy.backends.registry.SegmentationRegistry
   :members:

.. autoclass:: SegSpy.backends.traditional.TraditionalCVSegmenter
   :members:

.. autoclass:: SegSpy.backends.sam.SAMSegmenter
   :members:
```
