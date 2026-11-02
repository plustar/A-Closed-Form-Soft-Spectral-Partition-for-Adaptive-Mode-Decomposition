"""Extraction (Step 2): the properties of the hard and soft masks that the paper states."""
import numpy as np
import pytest

from softdamd import (MeanShift, Partition, SoftDAMD, STFT, SSQSTFT, fwhm_alpha,
                      lorentzian_weights, voronoi_weights)

def test_partition_of_unity_and_limits():
    freqs = np.linspace(0, 0.5, 257)
    c = np.array([0.05, 0.12, 0.3])
    for a in (1.0, 50.0, 2000.0, 1e7):
        np.testing.assert_allclose(lorentzian_weights(freqs, c, a).sum(axis=0), 1.0, atol=1e-14)
    H = voronoi_weights(freqs, c)
    np.testing.assert_allclose(H.sum(axis=0), 1.0)
    # alpha -> infinity: the inverse-square allocation d_k^-2 / sum_j d_j^-2, not the
    # Voronoi mask (Section II / Corollary of the paper)
    L = lorentzian_weights(freqs, c, 1e12)
    d = np.abs(freqs[:, None] - c[None, :])
    ok = d.min(axis=1) > 1e-3
    inv = (d[ok] ** -2 / (d[ok] ** -2).sum(axis=1, keepdims=True)).T
    np.testing.assert_allclose(L[:, ok], inv, rtol=1e-4)


def test_voronoi_ties_shared():
    freqs = np.array([0.0, 0.1, 0.2, 0.3])
    H = voronoi_weights(freqs, np.array([0.0, 0.2]))
    np.testing.assert_allclose(H[:, 1], [0.5, 0.5])         # equidistant bin
    np.testing.assert_allclose(H[:, [0, 2, 3]], [[1, 0, 0], [0, 1, 1]])


def test_voronoi_matches_midpoint_bands_off_ties():
    rng = np.random.default_rng(1)
    freqs = np.linspace(0, 0.5, 513)
    for _ in range(20):
        c = np.sort(rng.uniform(0, 0.5, 4))
        p = Partition.from_centers([c], freqs)
        lib = np.zeros((4, freqs.size))
        for k, (b0, b1) in enumerate(p.band):
            lib[k, b0:b1] = 1
        assert np.array_equal(lib, voronoi_weights(freqs, c))


def test_fwhm_on_lorentzian():
    freqs = np.linspace(0, 0.5, 4097)
    for alpha in (2e3, 2e4, 2e5):
        P = 1 / (1 + 2 * alpha * (freqs - 0.2) ** 2)
        a = fwhm_alpha(P, freqs, np.array([[0, freqs.size]]))[0]
        assert abs(a / alpha - 1) < 0.1


def test_reconstruction_stft():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(600)
    res = SoftDAMD(STFT(n_fft=128, hop=1)).run(x, 1000.0)
    for mode, alpha in (('soft', 2000.0), ('hard', None), ('soft', 'fwhm')):
        np.testing.assert_allclose(res.reconstruct(mode, alpha), x, atol=1e-10)


def test_fixed_centres_decoupled_from_detection():
    fs = 1000.0
    t = np.arange(800) / fs
    x = np.cos(2 * np.pi * 100 * t) + np.cos(2 * np.pi * 130 * t)
    m = STFT(n_fft=128, hop=8).forward(x, fs)
    P = Partition.from_centers([np.array([100, 130]) / fs] * m.n_frames, m.freqs)
    from softdamd import extract
    U = extract(m, P, 10, 'soft', 2000.0)
    assert U.shape == (2, m.n_freqs)
    np.testing.assert_allclose(U.sum(axis=0), m.coef[:, 10], atol=1e-12)


def test_mode_energy_partitions_frame_energy():
    x = np.random.default_rng(3).standard_normal(800)
    res = SoftDAMD(SSQSTFT(n_fft=128)).run(x, 1000.0)
    for t in (100, 400):
        P = np.abs(res.tfmap.coef[:, t]) ** 2
        for mode in ('soft', 'hard'):
            np.testing.assert_allclose(res.energy(t, mode).sum(), P.sum(), rtol=1e-12)
