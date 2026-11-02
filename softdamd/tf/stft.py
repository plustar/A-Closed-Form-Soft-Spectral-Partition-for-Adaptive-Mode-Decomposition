"""Short-time Fourier transform and its synchrosqueezed version.

Conventions (one signal x[n], n = 0 … N−1, window w of length M = n_fft)

    frame t is centred on sample t·hop:  x_t[m] = x[t·hop + m − ⌊M/2⌋],  m = 0 … M−1,
    with reflect padding at the ends;

    STFT   S[k, t] = Σ_m x_t[m] w[m] e^{−j2πk(m − ⌊M/2⌋)/M},     k = 0 … ⌊M/2⌋.

The phase is referenced to the frame centre (the "modulated" form), which is what
synchrosqueezing needs; |S| is the ordinary spectrogram.  Row k has frequency
f_k = k·fs/M (the DFT grid, including for odd M).
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np

from .base import TFMap, as_signal


# ── windows ──────────────────────────────────────────────────────────────────

def make_window(name: Union[str, np.ndarray], n: int) -> np.ndarray:
    """Periodic window of length ``n`` ('hann', 'hamming', 'blackman', 'rect', or any
    name scipy.signal.get_window accepts); an array is returned unchanged."""
    if not isinstance(name, str):
        return np.asarray(name, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    key = name.lower()
    if key in ('hann', 'hanning'):
        return 0.5 * (1.0 - np.cos(2 * math.pi * t / n))
    if key in ('rect', 'rectangular', 'boxcar', 'ones'):
        return np.ones(n)
    import scipy.signal
    return np.asarray(scipy.signal.get_window(name, n, fftbins=True), dtype=np.float64)


def derivative_window(window: np.ndarray) -> np.ndarray:
    """dw/dm of a periodic window, in cycles⁻¹ per sample, computed exactly in the
    frequency domain (Nyquist term zeroed for even length)."""
    M = len(window)
    xi = 2 * np.pi * np.fft.fftfreq(M)
    if M % 2 == 0:
        xi[M // 2] = 0.0
    return np.fft.ifft(np.fft.fft(window) * 1j * xi).real


# ── core ─────────────────────────────────────────────────────────────────────

def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(T, n_fft) frames centred on samples 0, hop, 2·hop, …; T = ⌊(N − 1)/hop⌋ + 1."""
    c = n_fft // 2
    xp = np.pad(x, (c, n_fft - 1 - c), mode='reflect')
    return np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop]


def stft_coef(x: np.ndarray, window: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """S[k, t] of the module docstring, shape (⌊n_fft/2⌋ + 1, T)."""
    fr = _frames(x, n_fft, hop) * window[np.newaxis, :]
    # phase referenced to the frame centre: rotate the centre sample to index 0
    return np.fft.rfft(np.roll(fr, -(n_fft // 2), axis=1), axis=1).T


def istft(coef: np.ndarray, window: np.ndarray, n_fft: int, hop: int,
          length: int) -> np.ndarray:
    """Least-squares inverse of :func:`stft_coef` (weighted overlap-add, normalised by
    Σ w²); exact for any hop at which the shifted windows cover every sample."""
    T = coef.shape[1]
    c = n_fft // 2
    frames = np.roll(np.fft.irfft(coef.T, n=n_fft, axis=1), c, axis=1)       # (T, n_fft)
    out_len = (T - 1) * hop + n_fft
    idx = (np.arange(T)[:, None] * hop + np.arange(n_fft)[None, :]).ravel()
    num = np.bincount(idx, weights=(frames * window).ravel(), minlength=out_len)
    den = np.bincount(idx, weights=np.tile(window ** 2, T), minlength=out_len)
    return num[c:c + length] / np.maximum(den[c:c + length], 1e-12)


class STFT:
    """Short-time Fourier transform (Hann window by default).

    Parameters
    ----------
    n_fft : window and FFT length
    hop : frame hop in samples (1: one frame per sample)
    window : window name or array of length n_fft
    """
    is_linear = True
    is_invertible = True

    def __init__(self, n_fft: int = 256, hop: int = 1, window='hann') -> None:
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.window = make_window(window, self.n_fft)

    def forward(self, x, fs: float = 1.0) -> TFMap:
        x = as_signal(x)
        coef = stft_coef(x, self.window, self.n_fft, self.hop)
        freqs = np.arange(coef.shape[0]) / self.n_fft
        return TFMap(coef=coef, freqs=freqs, freqs_hz=freqs * fs, fs=float(fs),
                     hop=self.hop, n_samples=x.size, scale='linear', transform=self)

    def inverse(self, coef: np.ndarray, tfmap: TFMap) -> np.ndarray:
        return istft(np.asarray(coef), self.window, self.n_fft, self.hop, tfmap.n_samples)

    def __repr__(self) -> str:
        return f'STFT(n_fft={self.n_fft}, hop={self.hop})'


# ── synchrosqueezing ─────────────────────────────────────────────────────────

def _squeeze(coef: np.ndarray, target: np.ndarray, ok: np.ndarray, n_rows: int) -> np.ndarray:
    """T[l, t] = Σ_{k : target[k,t] = l, ok[k,t]} coef[k, t]."""
    F, T = coef.shape
    cols = np.broadcast_to(np.arange(T)[np.newaxis, :], (F, T))
    out = np.zeros((n_rows, T), dtype=coef.dtype)
    np.add.at(out, (target[ok], cols[ok]), coef[ok])
    return out


class SSQSTFT:
    """Synchrosqueezed STFT (Fourier synchrosqueezed transform, Oberlin et al. 2014).

    Every coefficient S[k, t] is moved to the DFT bin nearest its instantaneous
    frequency  f̂[k, t] = f_k − Im(S'[k, t] / S[k, t]) / (2π),  S' being the STFT with
    the derivative window.  Coefficients with |S| < gamma are dropped.  The rows are the
    DFT grid of :class:`STFT`.

    With hop = 1 the transform inverts as  x[t] = Re Σ_l c_l T[l, t] / (n_fft · w[⌊n_fft/2⌋]),
    c_0 = 1, c_l = 2 otherwise (c = 1 also at the Nyquist row for even n_fft); the
    only loss is the coefficients below gamma and those squeezed across the DC or
    Nyquist row.
    """
    is_linear = False
    is_invertible = True

    def __init__(self, n_fft: int = 256, hop: int = 1, window='hann', gamma=None) -> None:
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.window = make_window(window, self.n_fft)
        self.dwindow = derivative_window(self.window)
        self.gamma = gamma

    def forward(self, x, fs: float = 1.0) -> TFMap:
        x = as_signal(x)
        S = stft_coef(x, self.window, self.n_fft, self.hop)
        dS = stft_coef(x, self.dwindow, self.n_fft, self.hop)
        F = S.shape[0]
        gamma = self.gamma if self.gamma is not None else 10 * np.finfo(np.float64).eps
        ok = np.abs(S) >= gamma
        k = np.arange(F)[:, np.newaxis]
        # instantaneous frequency in bins; dS/S = −j·2π·(f̂ − f_k)/M for a pure tone
        f_hat = np.full(S.shape, np.inf)
        f_hat[ok] = (k + np.zeros_like(S.real))[ok] - self.n_fft * np.imag(dS[ok] / S[ok]) / (2 * np.pi)
        target = np.rint(np.where(ok, f_hat, 0.0)).astype(np.int64)
        ok &= (target >= 0) & (target < F)
        Tx = _squeeze(S, target, ok, F)
        freqs = np.arange(F) / self.n_fft
        return TFMap(coef=Tx, freqs=freqs, freqs_hz=freqs * fs, fs=float(fs), hop=self.hop,
                     n_samples=x.size, scale='linear', transform=self)

    def inverse(self, coef: np.ndarray, tfmap: TFMap) -> np.ndarray:
        if self.hop != 1:
            raise ValueError('the SSQ-STFT inverse requires hop = 1')
        coef = np.asarray(coef)
        c = np.full(coef.shape[0], 2.0)
        c[0] = 1.0
        if self.n_fft % 2 == 0:
            c[-1] = 1.0
        x = (c[:, np.newaxis] * coef).real.sum(axis=0) / (self.n_fft * self.window[self.n_fft // 2])
        return x[:tfmap.n_samples]

    def __repr__(self) -> str:
        return f'SSQSTFT(n_fft={self.n_fft}, hop={self.hop})'
