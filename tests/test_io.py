"""Tests for SegSpy.io helpers."""
import numpy as np
import hyperspy.api as hs

from SegSpy.io import to_uint8


class TestToUint8:
    def test_float_array_normalized_to_uint8_spanning_0_255(self):
        s = hs.signals.Signal2D(
            np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        )
        out = to_uint8(s)
        assert out.data.dtype == np.uint8
        assert out.data[0, 0] == 0
        assert out.data[-1, -1] == 255

    def test_constant_array_yields_zeros(self):
        s = hs.signals.Signal2D(np.full((4, 4), 128.0, dtype=np.float32))
        out = to_uint8(s)
        assert out.data.dtype == np.uint8
        assert np.all(out.data == 0)

    def test_already_uint8_returned_unchanged_values(self):
        arr = np.array([[0, 128], [128, 255]], dtype=np.uint8)
        s = hs.signals.Signal2D(arr)
        out = to_uint8(s)
        assert out.data.dtype == np.uint8
        np.testing.assert_array_equal(out.data, arr)

    def test_does_not_mutate_input(self):
        s = hs.signals.Signal2D(np.array([[0.0, 300.0]], dtype=np.float32))
        original = s.data.copy()
        to_uint8(s)
        np.testing.assert_array_equal(s.data, original)
