"""HyperSpy-native I/O helpers.

SegSpy consumes :class:`hyperspy.signals.Signal2D` objects; loading EM files is
the caller's job (use ``hyperspy.api.load``). The two helpers here are
dependency-light utilities shared across backends and the CLI.
"""

import numpy as np


def to_uint8(signal):
    """Normalize a signal's data to ``uint8`` (0–255) without mutating the input.

    - Already-uint8 signals are deep-copied unchanged.
    - Constant-valued signals map to all zeros (avoids divide-by-zero).
    - Everything else is linearly stretched to ``[0, 255]``.

    Returns a new signal (deepcopy) whose ``.data`` is ``uint8``.
    """
    data = signal.data
    if data.dtype == np.uint8:
        return signal.deepcopy()
    dmin, dmax = data.min(), data.max()
    if dmax == dmin:
        normalized = np.zeros_like(data, dtype=np.uint8)
    else:
        normalized = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    result = signal.deepcopy()
    result.data = normalized
    return result


def get_scale_nm(signal) -> float:
    """Return the nm/pixel scale from a signal's axis calibration.

    Converts the axis-0 ``scale`` × ``units`` into nanometers:
    ``um``→×1000, ``nm``→×1, ``mm``→×1e6, ``pm``→×1e-3.

    Raises:
        ValueError: if the units string is not a recognised length unit.
    """
    scale = signal.axes_manager[0].scale
    units = signal.axes_manager[0].units
    if units in ("um", "µm", "micrometer"):
        return scale * 1000.0
    if units in ("nm", "nanometer"):
        return scale
    if units in ("mm",):
        return scale * 1e6
    if units in ("pm", "picometer"):
        return scale * 1e-3
    raise ValueError(f"Unknown units for scale conversion: '{units}'")
