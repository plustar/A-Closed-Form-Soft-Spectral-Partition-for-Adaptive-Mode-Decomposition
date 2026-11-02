"""Transforms: exactness, localisation, reconstruction, and agreement with ssqueezepy."""
import numpy as np
import pytest

from softdamd.tf import CWT, SSQCWT, SSQSTFT, STFT

FS = 1000.0
N = 1024
T = np.arange(N) / FS
TONE = np.cos(2 * np.pi * 100.3 * T)
CHIRP = np.cos(2 * np.pi * (50 * T + 100 * T ** 2))
NOISE = np.random.default_rng(0).standard_normal(N)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize('n_fft,hop', [(256, 1), (128, 1), (125, 1), (64, 4), (125, 3)])
def test_stft_perfect_reconstruction(n_fft, hop):
    m = STFT(n_fft=n_fft, hop=hop).forward(NOISE, FS)
    assert m.coef.shape == (n_fft // 2 + 1, -(-N // hop))
    assert rel(m.inverse(), NOISE) < 1e-13


@pytest.mark.parametrize('n_fft', [128, 125])
def test_stft_grid_is_dft_grid(n_fft):
    m = STFT(n_fft=n_fft).forward(NOISE, FS)
    np.testing.assert_allclose(m.freqs_hz, np.arange(n_fft // 2 + 1) * FS / n_fft)


def test_stft_frame_centred_on_sample():
    x = np.zeros(N)
    x[500] = 1.0
    e = (np.abs(STFT(n_fft=128).forward(x, FS).coef) ** 2).sum(axis=0)
    assert e.argmax() == 500
    assert np.nonzero(e > 0)[0].tolist() == list(range(437, 564))


def test_modulated_phase_tone_is_stationary():
    """Frame-centred phase: for a tone, S[k, t+1]/S[k, t] = exp(j2πf/fs)."""
    S = STFT(n_fft=128).forward(TONE, FS).coef
    k = int(round(100.3 / FS * 128))
    ratio = S[k, 201:800] / S[k, 200:799]
    # (the residual wobble is leakage from the negative-frequency image of the cosine)
    np.testing.assert_allclose(np.angle(ratio), 2 * np.pi * 100.3 / FS, atol=1e-4)


@pytest.mark.parametrize('n_fft', [128, 125])
def test_ssq_stft_tone_and_chirp(n_fft):
    for x in (TONE, CHIRP):
        m = SSQSTFT(n_fft=n_fft).forward(x, FS)
        assert rel(m.inverse(), x) < 1e-3
    m = SSQSTFT(n_fft=n_fft).forward(TONE, FS)
    row = (np.abs(m.coef) ** 2).sum(axis=1).argmax()
    assert abs(m.freqs_hz[row] - 100.3) <= FS / n_fft / 2


def test_ssq_stft_noise_reconstruction():
    assert rel(SSQSTFT(n_fft=128).forward(NOISE, FS).inverse(), NOISE) < 0.05


def test_ssq_stft_matches_ssqueezepy():
    sq = pytest.importorskip('ssqueezepy')
    Tx, Sx, *_ = sq.ssq_stft(CHIRP, window='hann', n_fft=128, win_len=128, hop_len=1,
                             fs=FS, astensor=False)
    S = STFT(n_fft=128).forward(CHIRP, FS).coef
    T_ = SSQSTFT(n_fft=128).forward(CHIRP, FS).coef
    assert np.corrcoef(np.abs(S).ravel(), np.abs(Sx).ravel())[0, 1] > 1 - 1e-9
    assert np.corrcoef(np.abs(T_).ravel(), np.abs(Tx).ravel())[0, 1] > 1 - 1e-6


@pytest.mark.parametrize('cls', [CWT, SSQCWT])
def test_wavelet_reconstruction(cls):
    for x, tol in ((TONE, 0.01), (CHIRP, 0.01)):
        assert rel(cls().forward(x, FS).inverse(), x) < tol


def test_wavelet_tone_row():
    for cls in (CWT, SSQCWT):
        m = cls().forward(TONE, FS)
        row = (np.abs(m.coef) ** 2).sum(axis=1).argmax()
        ratio = m.freqs_hz[1] / m.freqs_hz[0]
        assert abs(np.log(m.freqs_hz[row] / 100.3)) <= np.log(ratio) / 2 + 1e-12


def test_rejects_multichannel():
    with pytest.raises(ValueError, match='single-channel'):
        STFT().forward(np.zeros((1, 512)))
    with pytest.raises(ValueError):
        STFT().forward(np.zeros((2, 512)))
