"""VMD / STVMD: agreement with the stored reference outputs (to golden_utils.RTOL)."""
import numpy as np
import pytest

from softdamd.baselines import VMD, STVMD
from golden_utils import load, assert_same

G = load('baselines')


def test_vmd_default():
    x = G['vmd/x']
    r = VMD(K=3, alpha=1000).decompose(x, fs=1000.0)
    assert_same(G, 'vmd/modes', r.modes)
    assert_same(G, 'vmd/omega', r.center_freqs)
    assert r.n_iterations == int(G['vmd/n_iter'])


def test_vmd_omega_init():
    x = G['vmd/x']
    r = VMD(K=3, alpha=2000, tau=1e-5, tol=1e-9, maxiters=500).decompose(
        x, fs=1000.0, omega_init=G['vmd_init/omega_init'])
    assert_same(G, 'vmd_init/modes', r.modes)
    assert_same(G, 'vmd_init/omega', r.center_freqs)


@pytest.mark.parametrize('dynamic', [False, True])
def test_stvmd(dynamic):
    k = f'stvmd/{"dynamic" if dynamic else "static"}'
    r = STVMD(K=3, n_fft=128, alpha=50, tau=1e-5, tol=1e-9, maxiters=60).decompose(
        G['stvmd/x'], fs=1000.0, dynamic=dynamic)
    assert_same(G, k + '/modes', r.modes)
    assert_same(G, k + '/omega', r.center_freqs)
    assert_same(G, k + '/residual', np.float64(r.residual))


def test_rejects_2d():
    with pytest.raises(ValueError):
        VMD(K=2).decompose(np.zeros((2, 64)))
