"""Frequency and α conventions used throughout softdamd.

Frequency
    ω = f / fs, normalised frequency in cycles per sample, ω ∈ [0, 0.5]
    (Nyquist at 0.5).  Hertz = ω · fs.  ``TFMap.freqs`` and every centre ω_k
    are in this unit.

α (manuscript convention)
    Lorentzian weight denominator 1 + 2α(ω − ω_k)², ω in normalised frequency, so
    α has units of (cycles/sample)⁻².  The mode half-bandwidth is β = 1/√(2α).

α of the VMD baselines (reference-implementation convention)
    The VMD/STVMD code follows Dragomiretskiy & Zosso: denominator 1 + α_impl(ω' − ω'_k)²
    on the axis ω' of that code.  :func:`alpha_impl_from_paper` converts.
"""
from __future__ import annotations

import numpy as np


def hz_to_norm(f_hz, fs: float) -> np.ndarray:
    """Hertz -> normalised frequency ω = f/fs (not 2f/fs)."""
    return np.asarray(f_hz, dtype=np.float64) / float(fs)


def norm_to_hz(w, fs: float) -> np.ndarray:
    return np.asarray(w, dtype=np.float64) * float(fs)


def alpha_impl_from_paper(alpha_paper: float) -> float:
    """Manuscript α (1 + 2α d²) -> implementation α (1 + α d²)."""
    return 2.0 * float(alpha_paper)


def alpha_paper_from_impl(alpha_impl: float) -> float:
    return 0.5 * float(alpha_impl)


def beta_from_alpha(alpha_paper: float) -> float:
    """Lorentzian half-bandwidth β = 1/√(2α), normalised frequency."""
    return 1.0 / np.sqrt(2.0 * float(alpha_paper))


def alpha_from_beta(beta: float) -> float:
    return 1.0 / (2.0 * float(beta) ** 2)
