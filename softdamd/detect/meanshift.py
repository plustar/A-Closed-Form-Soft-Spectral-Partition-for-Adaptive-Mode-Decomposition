"""Step 1 of Algorithm 1: per-frame density-based mode detection (DAMD).

For every frame t the power spectrum P(ω, t), normalised to unit maximum, is treated
as a weighted sample on the frequency grid.  Mean-shift is run from seeds placed
every ``seed_spacing`` bins; converged seeds closer than the frame's bandwidth are
merged (gap split, mean of each group).  The merged points are the centres
ω*_{k,t}; their hard bands follow from the midpoints between neighbours.

Per-frame bandwidth, seed placement, iteration and merging are those of the
published DAMD detector.  The iteration loop is compiled with Numba; it is the only
implementation, so results do not depend on the environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .bandwidth import ADAPTATIONS, BASES, EPS, bandwidth
from .partition import Partition


@dataclass(frozen=True)
class MeanShift:
    """Mean-shift mode detector.

    Parameters
    ----------
    bandwidth : 'unit' | 'percentile'   base kernel bandwidth h0
    adapt : 'none' | 'curvature'        bandwidth adaptation
    seed_spacing : one seed every this many frequency bins
    max_iter : mean-shift iterations per frame (upper bound)
    tol : convergence threshold on the shift, normalised frequency
    h_min : bandwidth floor
    """
    bandwidth: str = 'unit'
    adapt: str = 'none'
    seed_spacing: int = 5
    max_iter: int = 1000
    tol: float = 1e-5
    h_min: float = 1e-6

    def __post_init__(self):
        if self.bandwidth not in BASES:
            raise ValueError(f'bandwidth must be one of {BASES}, got {self.bandwidth!r}')
        if self.adapt not in ADAPTATIONS:
            raise ValueError(f'adapt must be one of {ADAPTATIONS}, got {self.adapt!r}')

    def detect(self, power: np.ndarray, freqs: np.ndarray) -> Partition:
        """Detect the centres of every frame of ``power`` (F, T) on the axis ``freqs`` (F,)."""
        power = np.asarray(power, dtype=np.float64)
        freqs = np.asarray(freqs, dtype=np.float64)
        F, T = power.shape
        seed_idx = np.arange(0, F, max(1, self.seed_spacing), dtype=np.int64)
        k_max = len(seed_idx)                  # at most one centre per seed
        seeds = freqs[seed_idx]
        spectra = power.T                                           # (T, F)
        spectra = spectra / np.maximum(spectra.max(axis=1, keepdims=True), 1e-30)

        raw = np.empty((T, len(seed_idx)))
        frame_bw = np.empty(T)
        for t in range(T):
            h, h_scalar = bandwidth(freqs, spectra[t], self.bandwidth, self.adapt, self.h_min)
            raw[t] = shift_seeds(seeds, freqs, spectra[t], h, self.max_iter, self.tol, seed_idx)
            frame_bw[t] = h_scalar

        centers, n_modes = merge_seeds(raw, frame_bw, k_max)
        kde = kde_at(freqs, spectra, centers, frame_bw)
        return _to_partition(centers, n_modes, kde, frame_bw, freqs, k_max)


def shift_seeds(seeds, freqs, weights, h, max_iter, tol, seed_idx, iters=None):
    """Run mean-shift from every seed; returns the converged positions.

    Each seed keeps the bandwidth of its starting bin (per-bin ``h``) or the scalar
    ``h``.  A seed stops when its shift falls below ``tol``, when its kernel mass
    vanishes, or after ``max_iter`` iterations.  If ``iters`` (int array, one entry
    per seed) is given, it receives the number of iterations of every seed.
    """
    seeds = np.ascontiguousarray(seeds, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if np.ndim(h) > 0 and len(h) > 1:
        seed_h = np.ascontiguousarray(np.asarray(h, dtype=np.float64)[seed_idx])
    else:
        seed_h = np.full(len(seeds), float(np.asarray(h).ravel()[0]))
    centers = np.empty_like(seeds)
    n_it = np.zeros(len(seeds), dtype=np.int64)
    _shift_kernel(seeds, freqs, weights, seed_h, int(max_iter), float(tol), centers, n_it)
    if iters is not None:
        iters[:] = n_it
    return centers


@njit(cache=True, nogil=True)
def _shift_kernel(seeds, freqs, weights, seed_h, max_iter, tol, centers, n_it):
    """Compiled mean-shift loop (the only implementation; see ``shift_seeds``).

    One pass over the grid per iteration:  c <- sum_f k_f w_f f / sum_f k_f w_f,
    k_f = exp(-(c - f)^2 / (2 h^2)).
    """
    F = freqs.shape[0]
    for s in range(seeds.shape[0]):
        c = seeds[s]
        h = seed_h[s]
        it = 0
        while it < max_iter:
            it += 1
            num = 0.0
            den = 0.0
            for f in range(F):
                d = (c - freqs[f]) / h
                k = np.exp(-0.5 * d * d) * weights[f]
                den += k
                num += k * freqs[f]
            if den <= EPS:                   # no kernel mass: stay put and stop
                break
            new_c = num / den
            shift = abs(new_c - c)
            c = new_c
            if shift < tol:
                break
        centers[s] = c
        n_it[s] = it


def merge_seeds(raw: np.ndarray, frame_bw: np.ndarray, k_max: int):
    """Sort each frame's converged seeds, split where the gap exceeds the frame
    bandwidth, and average each group.  Returns NaN-padded centres (T, k_max) and
    K_t (T,); k_max ≥ number of seeds, so no group is ever dropped."""
    T, S = raw.shape
    c_sorted = np.sort(raw, axis=1)
    is_break = (c_sorted[:, 1:] - c_sorted[:, :-1]) > frame_bw[:, None]
    labels = np.concatenate([np.zeros((T, 1), dtype=np.int64),
                             np.cumsum(is_break.astype(np.int64), axis=1)], axis=1)

    sums = np.zeros((T, k_max))
    counts = np.zeros((T, k_max))
    flat_idx = np.repeat(np.arange(T, dtype=np.int64), S) * k_max + labels.ravel()
    np.add.at(sums.ravel(), flat_idx, c_sorted.ravel())
    np.add.at(counts.ravel(), flat_idx, 1.0)
    with np.errstate(invalid='ignore'):
        centers = np.where(counts > 0, sums / counts, np.nan)

    n_modes = (labels.max(axis=1) + 1).astype(np.int64)
    centers[np.arange(k_max)[None, :] >= n_modes[:, None]] = np.nan
    return centers, n_modes


def kde_at(freqs, spectra, points, frame_bw):
    """Σ_f exp(−(p − ω_f)²/(2(h_t + ε)²)) P̃(ω_f, t) at every point p (T, K)."""
    T, K = points.shape
    dist = points.reshape(T, K, 1) - freqs.reshape(1, 1, -1)
    kern = np.exp(-0.5 * (dist / (frame_bw.reshape(T, 1, 1) + EPS)) ** 2)
    return np.sum(kern * spectra.reshape(T, 1, -1), axis=2)


def _to_partition(centers, n_modes, kde, frame_bw, freqs, k_max) -> Partition:
    """NaN-padded (T, k_max) arrays -> CSR Partition with midpoint bands."""
    T, F = len(n_modes), len(freqs)
    # hard bands from the midpoints between neighbouring centres
    bin_idx = np.searchsorted(freqs, np.clip((centers[:, :-1] + centers[:, 1:]) / 2, 0, 0.5))
    starts = np.zeros((T, k_max), dtype=np.int64)
    ends = np.zeros((T, k_max), dtype=np.int64)
    kb = min(bin_idx.shape[1], k_max - 1)
    starts[:, 1:kb + 1] = bin_idx[:, :kb]
    ends[:, :kb] = bin_idx[:, :kb]
    k_idx = np.arange(k_max)[None, :]
    ends = np.where((k_idx == (n_modes - 1)[:, None]) & (n_modes[:, None] > 0), F, ends)

    frame_ids = np.repeat(np.arange(T, dtype=np.int64), n_modes)
    offsets = np.concatenate([[0], np.cumsum(n_modes)]).astype(np.int64)
    mode_ids = np.arange(int(n_modes.sum()), dtype=np.int64) - offsets[frame_ids]
    return Partition(
        n_modes=n_modes,
        centers=centers[frame_ids, mode_ids].astype(np.float64),
        band=np.stack([starts[frame_ids, mode_ids], ends[frame_ids, mode_ids]], axis=1),
        kde=kde[frame_ids, mode_ids].astype(np.float64),
        bandwidth=np.asarray(frame_bw, dtype=np.float64),
        freqs=freqs,
    )
