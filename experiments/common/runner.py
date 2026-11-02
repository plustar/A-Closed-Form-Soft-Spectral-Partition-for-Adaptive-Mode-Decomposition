"""Pipeline runner for the experiments.

``run_damd(x, fs, ...)`` runs Algorithm 1 with the detection settings of the
experiments written out as keyword defaults, and returns ``(Decomposition, seconds)``.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from softdamd import Decomposition, MeanShift, SoftDAMD, make_transform


def run_damd(x, fs: float, *, transform: str = 'stft', n_fft: Optional[int] = None,
             hop: int = 1, bandwidth: str = 'unit', adapt: str = 'curvature',
             seed_spacing: int = 5, alpha=2000.0) -> Tuple[Decomposition, float]:
    """Run transform → detection on ``x``.

    STFT-family transforms use ``n_fft`` (default max(fs // 2, 64)) and ``hop``;
    detection uses mean-shift with the given bandwidth rule, 1000 iterations,
    tolerance 1e-5 and bandwidth floor 1e-6.
    """
    if transform in ('stft', 'ssq_stft'):
        tf = make_transform(transform, n_fft=int(n_fft or max(int(fs) // 2, 64)), hop=hop)
    else:
        tf = make_transform(transform)
    det = MeanShift(bandwidth=bandwidth, adapt=adapt, seed_spacing=seed_spacing)
    t0 = time.perf_counter()
    res = SoftDAMD(tf, det, alpha=alpha).run(np.asarray(x, dtype=np.float64), fs)
    return res, time.perf_counter() - t0


def freq_axis(result: Decomposition, fs: float) -> np.ndarray:
    """Frequency axis in Hz."""
    return np.asarray(result.freqs) * fs


def time_axis(result: Decomposition, fs: float) -> np.ndarray:
    """Frame times in seconds."""
    return np.arange(result.tfmap.n_frames) * result.tfmap.hop / fs
