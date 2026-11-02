"""Detection (Step 1)."""
import numpy as np
import pytest

from softdamd import MeanShift, Partition, STFT, SSQSTFT
from softdamd.detect import midpoint_bands

FS = 1000.0


def two_tones(n=1024, f=(100.0, 300.0)):
    t = np.arange(n) / FS
    return sum(np.cos(2 * np.pi * fi * t) for fi in f)


@pytest.mark.parametrize('bw,adapt', [('unit', 'none'), ('unit', 'curvature'),
                                      ('percentile', 'curvature'), ('percentile', 'none')])
@pytest.mark.parametrize('tf', [STFT(n_fft=256, hop=16), SSQSTFT(n_fft=256, hop=1)])
def test_two_tones_found(bw, adapt, tf):
    m = tf.forward(two_tones(), FS)
    p = MeanShift(bandwidth=bw, adapt=adapt).detect(m.power(), m.freqs)
    t0 = p.T // 2
    c = p.centers_at(t0) * FS
    strong = np.sort(c[np.argsort(p.kde[p.frame(t0)])[-2:]])
    np.testing.assert_allclose(strong, [100, 300], atol=FS / 256)


def test_partition_structure():
    m = STFT(n_fft=128, hop=8).forward(two_tones(), FS)
    p = MeanShift().detect(m.power(), m.freqs)
    assert p.T == m.n_frames and p.N == p.n_modes.sum()
    for t in range(p.T):
        c = p.centers_at(t)
        assert np.all(np.diff(c) > 0)
        b = p.band[p.frame(t)]
        assert b[0, 0] == 0 and b[-1, 1] == m.n_freqs and np.all(b[1:, 0] == b[:-1, 1])


def test_midpoint_bands_cover_axis():
    freqs = np.linspace(0, 0.5, 101)
    band = midpoint_bands(np.array([0.1, 0.2, 0.4]), freqs)
    assert band[0, 0] == 0 and band[-1, 1] == 101
    assert np.all(band[1:, 0] == band[:-1, 1])


def test_from_centers():
    freqs = np.linspace(0, 0.5, 11)
    p = Partition.from_centers([[0.3, 0.1], [], [0.25]], freqs)
    assert list(p.n_modes) == [2, 0, 1]
    assert np.allclose(p.centers_at(0), [0.1, 0.3])
    assert p.frame(1) == slice(2, 2)


def test_bad_options():
    with pytest.raises(ValueError):
        MeanShift(bandwidth='silverman')
    with pytest.raises(ValueError):
        MeanShift(adapt='abramson')
