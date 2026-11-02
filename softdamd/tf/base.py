"""Time-frequency map container and the transform interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


class Transform(Protocol):
    """A time-frequency transform 𝒯: signal (N,) -> TFMap with coefficients (F, T)."""

    def forward(self, x: np.ndarray, fs: float = 1.0) -> 'TFMap': ...

    def inverse(self, coef: np.ndarray, tfmap: 'TFMap') -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class TFMap:
    """Coefficients 𝒯(ω, t) of one single-channel signal.

    Attributes
    ----------
    coef : (F, T) complex
    freqs : (F,) normalised frequency ω of each row, in cycles/sample, [0, 0.5]
    freqs_hz : (F,) the same axis in Hz (``freqs * fs`` up to rounding)
    fs : sampling rate in Hz
    hop : frame hop in samples
    n_samples : length of the analysed signal
    scale : 'linear' (STFT family) or 'nonlinear' (wavelet family, log-spaced rows)
    transform : the transform that produced the map (used by :meth:`inverse`)
    extra : transform-specific data needed by the inverse (e.g. CWT scales)
    """
    coef: np.ndarray
    freqs: np.ndarray
    freqs_hz: np.ndarray
    fs: float
    hop: int
    n_samples: int
    scale: str
    transform: Transform
    extra: dict = field(default_factory=dict)

    @property
    def n_freqs(self) -> int:
        return self.coef.shape[0]

    @property
    def n_frames(self) -> int:
        return self.coef.shape[1]

    def power(self) -> np.ndarray:
        """|𝒯(ω, t)|², shape (F, T)."""
        c = self.coef
        return c.real ** 2 + c.imag ** 2 if np.iscomplexobj(c) else c ** 2

    def times(self) -> np.ndarray:
        """Frame times in seconds."""
        return np.arange(self.n_frames) * self.hop / self.fs

    def inverse(self, coef: Optional[np.ndarray] = None) -> np.ndarray:
        """Signal (N,) from ``coef`` (default: this map's own coefficients)."""
        return self.transform.inverse(self.coef if coef is None else coef, self)


def as_signal(x) -> np.ndarray:
    """Validate a single-channel real signal and return it as float64 (N,)."""
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == 1:
        raise ValueError('softdamd is single-channel: pass a 1-D signal of shape (N,), '
                         f'not {x.shape}')
    if x.ndim != 1:
        raise ValueError(f'expected a 1-D signal of shape (N,), got shape {x.shape}')
    if x.size < 2:
        raise ValueError(f'signal too short: N = {x.size}')
    if np.iscomplexobj(x):
        raise ValueError('expected a real-valued signal')
    x = x.astype(np.float64, copy=False)
    if not np.all(np.isfinite(x)):
        raise ValueError('signal contains NaN or Inf')
    return x
