"""Mean-shift kernel bandwidth: h = base × adaptation, clipped to [h0, h_max].

Base h0
    'unit'        median bin spacing Δf (per-bin spacing on non-uniform grids)
    'percentile'  10th percentile of the spacings of the frequency grid
Adaptation λ_i ≥ 1
    'none'        λ_i = 1
    'curvature'   λ_i ∝ |P''(ω_i)|^(-1/2), normalised so that min λ = 1

h_max is Silverman's rule of thumb 0.9 σ N^(-1/5) (never below h0).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

EPS = 1e-10

BASES = ('unit', 'percentile')
ADAPTATIONS = ('none', 'curvature')


def local_spacing(freqs: np.ndarray):
    """Per-bin spacing (N,), its median, and whether the grid is non-uniform
    (max/min spacing > 2)."""
    N = len(freqs)
    if N <= 1:
        return np.array([0.01]), 0.01, False
    diffs = np.diff(freqs)
    df_local = np.empty(N, dtype=np.float64)
    df_local[0] = diffs[0]
    df_local[-1] = diffs[-1]
    df_local[1:-1] = (diffs[:-1] + diffs[1:]) / 2
    df_scalar = float(np.median(df_local))
    is_nonuniform = (df_local.max() / max(df_local.min(), 1e-12)) > 2.0
    return df_local, df_scalar, is_nonuniform


def weighted_std(freqs: np.ndarray, weights: np.ndarray) -> float:
    total = np.sum(weights)
    if total < EPS:
        return 0.0
    weights_norm = weights / (total + EPS)
    mu = np.sum(freqs * weights_norm)
    variance = np.sum(weights_norm * (freqs - mu) ** 2)
    return float(np.sqrt(variance))


def silverman(freqs: np.ndarray, weights: np.ndarray) -> float:
    """0.9 σ N^(-1/5), floored at 1e-3 (0.01 for an all-zero spectrum)."""
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    if weights.sum() < EPS:
        return 0.01
    bw = 0.9 * weighted_std(freqs, weights) * len(freqs) ** (-0.2)
    return max(bw, 1e-3)


def percentile_spacing(freqs: np.ndarray, percentile: float = 10) -> float:
    if len(freqs) < 2:
        return 0.01
    diffs = np.abs(np.diff(np.sort(freqs)))
    valid = diffs[diffs > EPS]
    if len(valid) == 0:
        return float(np.median(diffs)) if len(diffs) > 0 else 0.01
    return float(np.percentile(valid, percentile))


def curvature_lambda(weights: np.ndarray, df: float) -> np.ndarray:
    """λ_i = |P''_i|^(-1/2) / min_j |P''_j|^(-1/2), second differences with spacing df."""
    N = len(weights)
    w = weights + EPS
    df2 = df * df
    d2 = np.zeros(N, dtype=np.float64)
    d2[1:-1] = (w[2:] - 2 * w[1:-1] + w[:-2]) / df2
    d2[0] = (w[2] - 2 * w[1] + w[0]) / df2
    d2[-1] = (w[-1] - 2 * w[-2] + w[-3]) / df2
    raw = np.sqrt(1.0 / np.maximum(np.abs(d2), EPS))
    return raw / raw.min()


def bandwidth(freqs: np.ndarray, weights: np.ndarray, base: str = 'unit',
              adapt: str = 'none', h_min: float = 1e-6
              ) -> Tuple[Union[float, np.ndarray], float]:
    """Kernel bandwidth for one frame.

    Returns ``(h, h_scalar)``: ``h`` is a scalar, or a per-bin (N,) array when the
    grid is non-uniform or an adaptation is used; ``h_scalar`` is its mean (used for
    the de-duplication threshold and the KDE value).
    """
    df_local, df_scalar, nonuniform = local_spacing(freqs)

    h0_perbin = None
    if base == 'unit':
        h0 = df_scalar
        h0_perbin = df_local if nonuniform else None
    elif base == 'percentile':
        h0 = percentile_spacing(freqs, 10)
        if nonuniform:
            h0_perbin = df_local * (h0 / max(df_scalar, EPS))
    else:
        raise ValueError(f'bandwidth base must be one of {BASES}, got {base!r}')
    h0 = max(h0, h_min)

    if adapt == 'none':
        if h0_perbin is not None:
            h_max = max(silverman(freqs, weights), h0)
            per_bin = np.clip(h0_perbin, h_min, h_max)
            return per_bin, float(np.mean(per_bin))
        h = max(h0, h_min)
        return h, h
    if adapt != 'curvature':
        raise ValueError(f'bandwidth adaptation must be one of {ADAPTATIONS}, got {adapt!r}')

    lam = curvature_lambda(weights, df_scalar)
    h_max = max(silverman(freqs, weights), h0)
    if h0_perbin is not None:
        per_bin = np.clip(h0_perbin * lam, h0_perbin, h_max)
    else:
        per_bin = np.clip(h0 * lam, h0, h_max)
    return per_bin, float(np.mean(per_bin))
