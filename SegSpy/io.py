"""HyperSpy-native I/O helpers.

SegSpy consumes :class:`hyperspy.signals.Signal2D` objects; loading EM files is
the caller's job (use ``hyperspy.api.load``). The helpers here are
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
