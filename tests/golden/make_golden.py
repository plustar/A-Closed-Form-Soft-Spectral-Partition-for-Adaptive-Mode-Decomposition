"""Provenance of ``baselines.npz``: reference VMD / STVMD outputs.

The ADMM kernels of ``softdamd.baselines`` follow the reference VMD implementation;
this file pins their output so that any change to them is detected by
``tests/test_baselines.py``.  Regenerate only when a change is intended:

    python tests/golden/make_golden.py
"""
import os

import numpy as np

from softdamd.baselines import VMD, STVMD

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(7)
    t = np.arange(1024) / 1000
    x = np.sin(2 * np.pi * 80 * t) + np.sin(2 * np.pi * 200 * t) + 0.1 * rng.standard_normal(1024)
    out = {'vmd/x': x}
    r = VMD(K=3, alpha=1000).decompose(x, fs=1000.0)
    out['vmd/modes'], out['vmd/omega'] = np.asarray(r.modes), np.asarray(r.center_freqs)
    out['vmd/n_iter'] = np.int64(r.n_iterations)
    omega0 = np.array([0.05, 0.15, 0.3])
    r = VMD(K=3, alpha=2000, tau=1e-5, tol=1e-9, maxiters=500).decompose(
        x, fs=1000.0, omega_init=omega0)
    out['vmd_init/omega_init'] = omega0
    out['vmd_init/modes'], out['vmd_init/omega'] = np.asarray(r.modes), np.asarray(r.center_freqs)
    xs = x[:256]
    out['stvmd/x'] = xs
    for dyn in (False, True):
        r = STVMD(K=3, n_fft=128, alpha=50, tau=1e-5, tol=1e-9, maxiters=60).decompose(
            xs, fs=1000.0, dynamic=dyn)
        k = f'stvmd/{"dynamic" if dyn else "static"}'
        out[k + '/modes'], out[k + '/omega'] = np.asarray(r.modes), np.asarray(r.center_freqs)
        out[k + '/residual'] = np.float64(r.residual)
    np.savez_compressed(os.path.join(HERE, 'baselines.npz'), **out)


if __name__ == '__main__':
    main()
