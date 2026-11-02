"""Continuous wavelet transform (analytic Morlet) and its synchrosqueezed version.

Conventions (Daubechies, Lu & Wu 2011; ssqueezepy)

    W(a, b) = Σ_ξ X̂(ξ) Ψ̂*(aξ) e^{jξb} / N          (L1 normalisation, ξ in rad/sample)
    Ψ̂(ξ)  = π^{-1/4} [exp(−(ξ − μ)²/2) − exp(−(ξ² + μ²)/2)],  ξ > 0,  zero for ξ ≤ 0

(the analytic Morlet wavelet with the admissibility correction, Ψ̂(0) = 0).

Scales are log-spaced, a_j = a_min·r^j.  Scale a has centre frequency μ/(2πa) cycles per
sample.  With C_ψ = ∫_0^∞ Ψ̂*(u) du/u the single-integral reconstruction is

    x(b) = 2 Re Σ_j W(a_j, b) Δlog a / C_ψ,

which holds for the CWT and, since squeezing only moves terms of that sum between
rows, for the synchrosqueezed transform T as well (x = 2 Re Σ_l T[l, b] / C_ψ).
Rows are ordered by increasing frequency.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

from .base import TFMap, as_signal
from .stft import _squeeze


def morlet_hat(xi: np.ndarray, mu: float = 6.0) -> np.ndarray:
    """Analytic, admissible Morlet wavelet in frequency (zero for ξ ≤ 0)."""
    xi = np.asarray(xi, dtype=np.float64)
    val = math.pi ** (-0.25) * (np.exp(-0.5 * (xi - mu) ** 2) - np.exp(-0.5 * (xi ** 2 + mu ** 2)))
    return np.where(xi > 0, val, 0.0)


@lru_cache(maxsize=None)
def admissibility(mu: float = 6.0) -> float:
    """C_ψ = ∫_0^∞ Ψ̂(u)/u du (the Morlet Ψ̂ is real)."""
    from scipy.integrate import quad
    val, _ = quad(lambda u: morlet_hat(u, mu) / u, 0.0, mu + 40.0, points=[mu], limit=200)
    return float(val)


def log_scales(n_samples: int, n_scales: int = 64, a_min: float = 2.0) -> np.ndarray:
    """n_scales log-spaced scales from a_min to N/4 (ascending)."""
    return np.geomspace(a_min, max(n_samples / 4.0, 2 * a_min), n_scales)


def _cwt(x: np.ndarray, scales: np.ndarray, mu: float, derivative: bool):
    """W (and ∂_b W) for every scale; rows in ascending scale."""
    N = x.size
    N_up = 1 << (max(N, 2) - 1).bit_length()                 # reflect-pad to 2^p
    pad_l = (N_up - N) // 2
    xp = np.pad(x, (pad_l, N_up - N - pad_l), mode='reflect') if N_up > N else x
    X = np.fft.fft(xp)
    xi = 2 * np.pi * np.fft.fftfreq(N_up)
    Psi = morlet_hat(scales[:, np.newaxis] * xi[np.newaxis, :], mu)   # real, so Ψ̂* = Ψ̂
    W = np.fft.ifft(X[np.newaxis, :] * Psi, axis=1)[:, pad_l:pad_l + N]
    dW = (np.fft.ifft(X[np.newaxis, :] * (1j * xi)[np.newaxis, :] * Psi, axis=1)[:, pad_l:pad_l + N]
          if derivative else None)
    return W, dW


class CWT:
    """Continuous wavelet transform with the analytic Morlet wavelet.

    Parameters
    ----------
    n_scales : number of log-spaced scales (rows)
    mu : Morlet centre frequency in rad/sample at unit scale
    """
    is_linear = True
    is_invertible = True

    def __init__(self, n_scales: int = 64, mu: float = 6.0) -> None:
        self.n_scales = int(n_scales)
        self.mu = float(mu)

    def _axis(self, n):
        scales = log_scales(n, self.n_scales)[::-1]            # descending scale
        return scales, self.mu / (2 * np.pi * scales)            # ascending frequency

    def forward(self, x, fs: float = 1.0) -> TFMap:
        x = as_signal(x)
        scales, freqs = self._axis(x.size)
        W, _ = _cwt(x, scales, self.mu, derivative=False)
        return TFMap(coef=W, freqs=freqs, freqs_hz=freqs * fs, fs=float(fs), hop=1,
                     n_samples=x.size, scale='nonlinear', transform=self,
                     extra={'scales': scales})

    def inverse(self, coef: np.ndarray, tfmap: TFMap) -> np.ndarray:
        dlog = abs(np.log(tfmap.extra['scales'][0] / tfmap.extra['scales'][1]))
        return 2.0 * np.real(np.asarray(coef).sum(axis=0)) * dlog / admissibility(self.mu)

    def __repr__(self) -> str:
        return f'CWT(n_scales={self.n_scales}, mu={self.mu:g})'


class SSQCWT(CWT):
    """Synchrosqueezed CWT.

    Each W(a, b)·Δlog a is moved to the row whose frequency is nearest (on the log
    axis) to its instantaneous frequency  f̂(a, b) = Im(∂_b W / W)/(2π)  cycles/sample;
    coefficients with |W| < gamma, or with f̂ outside the row range by more than half
    a row, are dropped.
    """
    is_linear = False

    def __init__(self, n_scales: int = 64, mu: float = 6.0, gamma: Optional[float] = None) -> None:
        super().__init__(n_scales, mu)
        self.gamma = gamma

    def forward(self, x, fs: float = 1.0) -> TFMap:
        x = as_signal(x)
        scales, freqs = self._axis(x.size)
        W, dW = _cwt(x, scales, self.mu, derivative=True)
        gamma = self.gamma if self.gamma is not None else 10 * np.finfo(np.float64).eps
        ok = np.abs(W) >= gamma
        f_hat = np.full(W.shape, np.nan)
        f_hat[ok] = np.imag(dW[ok] / W[ok]) / (2 * np.pi)
        ok &= f_hat > 0
        dlog = abs(np.log(scales[0] / scales[1]))
        # nearest row on the log-frequency axis
        pos = np.full(W.shape, -1.0)
        pos[ok] = (np.log(f_hat[ok]) - np.log(freqs[0])) / dlog
        target = np.rint(pos).astype(np.int64)
        ok &= (target >= 0) & (target < len(freqs))
        Tx = _squeeze(W * dlog, target, ok, len(freqs))
        return TFMap(coef=Tx, freqs=freqs, freqs_hz=freqs * fs, fs=float(fs), hop=1,
                     n_samples=x.size, scale='nonlinear', transform=self,
                     extra={'scales': scales})

    def inverse(self, coef: np.ndarray, tfmap: TFMap) -> np.ndarray:
        return 2.0 * np.real(np.asarray(coef).sum(axis=0)) / admissibility(self.mu)

    def __repr__(self) -> str:
        return f'SSQCWT(n_scales={self.n_scales}, mu={self.mu:g})'
