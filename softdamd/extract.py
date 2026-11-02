"""Step 2 of Algorithm 1: mode extraction from given centres.

U_k(ω, t) = s_k(ω) 𝒯(ω, t) with

    hard  s_k(ω) = 1_{B_k}(ω), B_k the Voronoi cells of the centres (Eq. 7), ties
          (bins equidistant from m centres, to 1e-12) shared equally, s_k = 1/m;
    soft  s_k(ω) = [1 + 2α_k(ω − ω_k)²]⁻¹ / Σ_j [1 + 2α_j(ω − ω_j)²]⁻¹   (Eq. 12),
          α_k given, or estimated as α̂_k = 2/FWHM_k² on the hard band (Corollary).

Every function here takes centres from any source (detector, oracle, perturbed),
so extraction can be studied with detection held fixed.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from .detect.partition import Partition
from .tf.base import TFMap

Alpha = Union[float, np.ndarray, str]
TIE_TOL = 1e-12


def lorentzian_weights(freqs: np.ndarray, centers: np.ndarray, alpha) -> np.ndarray:
    """Soft weights s_k(ω), Eq. (12); ``alpha`` scalar or one value per centre.
    Returns (K, F); columns sum to 1."""
    freqs = np.asarray(freqs, dtype=np.float64)
    w_k = np.asarray(centers, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    a = np.full(len(w_k), float(a)) if a.ndim == 0 else a
    D = freqs[:, None] - w_k[None, :]
    W = 1.0 / (1.0 + 2.0 * a[None, :] * D ** 2)
    s = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30)
    return s.T


def voronoi_weights(freqs: np.ndarray, centers: np.ndarray, tol: float = TIE_TOL) -> np.ndarray:
    """Hard weights 1_{B_k}(ω), Eq. (7), ties shared equally.  Returns (K, F)."""
    d = np.abs(np.asarray(freqs, dtype=np.float64)[:, None]
               - np.asarray(centers, dtype=np.float64)[None, :])
    near = np.isclose(d, d.min(axis=1, keepdims=True), rtol=0, atol=tol)
    return (near / near.sum(axis=1, keepdims=True)).T


def fwhm_alpha(power: np.ndarray, freqs: np.ndarray, band: np.ndarray) -> np.ndarray:
    """α̂_k = 2/FWHM_k² for each hard band [b0, b1) of one frame (Corollary).

    FWHM_k = n_k · Δω_k, with n_k the number of bins of the band whose power is at
    least half the band maximum and Δω_k the bin spacing at the band start.  A band
    that is empty or carries no power is given the one-bin value n_k = 1.
    """
    power = np.asarray(power, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    F = len(freqs)
    out = np.empty(len(band))
    for k, (b0, b1) in enumerate(np.asarray(band, dtype=np.int64)):
        j = min(int(b0), F - 2)
        dw = freqs[j + 1] - freqs[j]
        P = power[b0:b1]
        n_half = int((P >= P.max() / 2.0).sum()) if P.size and P.max() > 0 else 1
        out[k] = 2.0 / (n_half * dw) ** 2
    return out


def frame_weights(tfmap: TFMap, partition: Partition, t: int, mode: str = 'soft',
                  alpha: Alpha = 2000.0) -> np.ndarray:
    """s_k(ω) of frame t, (K_t, F).  ``alpha='fwhm'`` estimates α̂_k per mode."""
    sl = partition.frame(t)
    c = partition.centers[sl]
    if mode == 'hard':
        return voronoi_weights(tfmap.freqs, c)
    if mode != 'soft':
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
    if isinstance(alpha, str):
        if alpha != 'fwhm':
            raise ValueError(f"alpha must be a number, an array or 'fwhm', got {alpha!r}")
        col = tfmap.coef[:, t]
        a = fwhm_alpha(col.real ** 2 + col.imag ** 2, tfmap.freqs, partition.band[sl])
    else:
        a = alpha
    return lorentzian_weights(tfmap.freqs, c, a)


def extract(tfmap: TFMap, partition: Partition, t: int, mode: str = 'soft',
            alpha: Alpha = 2000.0) -> np.ndarray:
    """Mode coefficients U_k(ω, t) = s_k(ω) 𝒯(ω, t) of frame t, (K_t, F) complex."""
    return frame_weights(tfmap, partition, t, mode, alpha) * tfmap.coef[:, t][None, :]


def mode_energy(tfmap: TFMap, partition: Partition, t: int, mode: str = 'soft',
                alpha: Alpha = 2000.0) -> np.ndarray:
    """Energy allocated to each mode of frame t, E_k = Σ_ω s_k(ω) |𝒯(ω, t)|², (K_t,).
    Σ_k E_k is the frame energy for either mask."""
    col = tfmap.coef[:, t]
    return frame_weights(tfmap, partition, t, mode, alpha) @ (col.real ** 2 + col.imag ** 2)


def reconstruct(tfmap: TFMap, partition: Partition, mode: str = 'soft',
                alpha: Alpha = 2000.0) -> np.ndarray:
    """Inverse transform of Σ_k U_k: equals the signal up to the transform's own
    round-trip error, because Σ_k s_k(ω) = 1 at every ω of every non-empty frame."""
    total = np.zeros_like(tfmap.coef)
    for t in range(partition.T):
        if partition.n_modes[t]:
            total[:, t] = extract(tfmap, partition, t, mode, alpha).sum(axis=0)
    return tfmap.inverse(total)
