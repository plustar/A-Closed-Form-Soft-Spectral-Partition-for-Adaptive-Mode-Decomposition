"""Per-frame detection result: centres ω*_{k,t} with a variable number K_t per frame.

Stored in CSR layout: the N = Σ_t K_t detections are concatenated frame by frame;
``offsets[t]:offsets[t+1]`` indexes frame t.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np


def midpoint_bands(centers: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Hard bands [start, end) of sorted centres from the midpoints between neighbours:
    bin f belongs to mode k when mid_{k-1} <= freqs[f] < mid_k (searchsorted, left)."""
    K, F = len(centers), len(freqs)
    band = np.zeros((K, 2), dtype=np.int64)
    if K == 0:
        return band
    mids = (centers[:-1] + centers[1:]) / 2
    idx = np.searchsorted(freqs, np.clip(mids, 0, 0.5))
    band[1:, 0] = idx
    band[:-1, 1] = idx
    band[-1, 1] = F
    return band


@dataclass(frozen=True, eq=False)
class Partition:
    """Detected mode centres of every frame.

    Attributes
    ----------
    n_modes : (T,) int     K_t
    centers : (N,) float   ω*_{k,t}, normalised frequency, ascending within a frame
    band : (N, 2) int      hard band [start, end) of each centre (midpoint boundaries)
    kde : (N,) float       kernel density at each centre (detector output)
    bandwidth : (T,) float kernel bandwidth of each frame (detector output)
    freqs : (F,) float     frequency axis of the time-frequency map
    """
    n_modes: np.ndarray
    centers: np.ndarray
    band: np.ndarray
    kde: np.ndarray
    bandwidth: np.ndarray
    freqs: np.ndarray

    @property
    def T(self) -> int:
        return int(len(self.n_modes))

    @property
    def N(self) -> int:
        return int(len(self.centers))

    @cached_property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(np.asarray(self.n_modes, dtype=np.int64))]).astype(np.int64)

    @cached_property
    def frame_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.T, dtype=np.int64), self.n_modes)

    def frame(self, t: int) -> slice:
        """Slice of frame t in the flat arrays."""
        return slice(int(self.offsets[t]), int(self.offsets[t + 1]))

    def centers_at(self, t: int) -> np.ndarray:
        return self.centers[self.frame(t)]

    @classmethod
    def from_centers(cls, centers: Sequence[np.ndarray], freqs: np.ndarray) -> 'Partition':
        """Partition from given centres (one array per frame, normalised frequency),
        e.g. oracle or perturbed centres.  ``kde`` and ``bandwidth`` are zero."""
        freqs = np.asarray(freqs, dtype=np.float64)
        per_frame = [np.sort(np.asarray(c, dtype=np.float64).ravel()) for c in centers]
        n_modes = np.array([len(c) for c in per_frame], dtype=np.int64)
        flat = np.concatenate(per_frame) if per_frame else np.zeros(0)
        band = (np.concatenate([midpoint_bands(c, freqs) for c in per_frame])
                if per_frame else np.zeros((0, 2), dtype=np.int64))
        return cls(n_modes=n_modes, centers=flat, band=band, kde=np.zeros(len(flat)),
                   bandwidth=np.zeros(len(per_frame)), freqs=freqs)

    def __repr__(self) -> str:
        nm = np.asarray(self.n_modes)
        k = f'K_t in [{nm.min()}, {nm.max()}]' if self.N else 'no detections'
        return f'Partition(T={self.T}, N={self.N}, {k})'
