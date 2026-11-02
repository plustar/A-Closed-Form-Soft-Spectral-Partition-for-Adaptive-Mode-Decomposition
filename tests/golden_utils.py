"""Access to the reference data of the baselines (see tests/golden/make_golden.py)."""
import os

import numpy as np

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')

# Floating-point results differ in the last bits between platforms (FFT and BLAS
# builds, compilers), so float arrays are compared to a relative tolerance far below
# any printed digit rather than bit for bit.
RTOL = 1e-10


def load(name):
    return np.load(os.path.join(GOLDEN, name + '.npz'), allow_pickle=False)


def assert_same(g, key, value):
    """Compare with the stored array: exact for integers, to RTOL (relative to the
    array's largest magnitude) for floats."""
    value = np.asarray(value)
    ref = g[key]
    assert ref.shape == value.shape, f'{key}: shape {value.shape} != {ref.shape}'
    assert ref.dtype == value.dtype, f'{key}: dtype {value.dtype} != {ref.dtype}'
    if ref.dtype.kind in 'fc':
        scale = max(float(np.nanmax(np.abs(ref))), np.finfo(np.float64).tiny)
        err = float(np.nanmax(np.abs(ref - value))) if ref.size else 0.0
        assert np.array_equal(np.isnan(ref), np.isnan(value)), f'{key}: NaN pattern differs'
        assert err <= RTOL * scale, f'{key}: max |diff| = {err:.3g} (scale {scale:.3g})'
    else:
        assert np.array_equal(ref, value), f'{key}: values differ'
