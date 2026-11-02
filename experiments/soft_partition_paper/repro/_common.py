"""Shared conventions and verification utilities for the reproducibility suite.

This module fixes the three conventions on which every script in the suite relies:

  1. FREQUENCY CONVENTION.  Every script here uses the *normalised* frequency
         w = f / fs  in  [0, 0.5]
     ("cycles per sample", Nyquist at 0.5).  This is the convention of
     ``softdamd.TFMap.freqs`` and of ``experiments/common/runner.freq_axis``
     (which recovers hertz as ``freqs * fs``).  Never use ``f/fs*2``.

  2. ALPHA CONVENTION.  The paper writes the VMD/Lorentzian denominator as
         1 + 2*alpha_paper*(w - w_k)**2
     while the reference VMD implementations (Dragomiretskiy's MATLAB code and
     ``softdamd/baselines/vmd.py``) use
         1 + alpha_impl*(w - w_k)**2 .
     Hence  alpha_impl = 2 * alpha_paper.  Every script states which one it uses and
     converts explicitly via :func:`alpha_impl_from_paper` / :func:`alpha_paper_from_impl`.
     Because alpha carries units of (normalised frequency)^-2, a bare number such as
     "alpha = 2000" is meaningless without both statements above.

  3. INDEPENDENT GROUND TRUTH.  Closed-form claims are checked against a generic
     numerical solver (:func:`simplex_qp_numeric`), never against a second copy of the
     same analytic expression.  This is what makes the checks non-circular.

Nothing here depends on the ``softdamd`` package, so the checks can be run standalone.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import numpy as np

__all__ = [
    "alpha_impl_from_paper", "alpha_paper_from_impl",
    "norm_freqs", "hz_to_norm", "norm_to_hz",
    "lorentzian_weights", "inverse_square_weights", "voronoi_weights",
    "weighted_lorentzian_paper", "weighted_lorentzian_sigma_only",
    "simplex_qp_numeric", "beta_from_alpha", "alpha_from_beta",
    "rel_l2", "env_report", "Section",
]


# ----------------------------------------------------------------------------------
# 1. Conventions
# ----------------------------------------------------------------------------------

def alpha_impl_from_paper(alpha_paper: float) -> float:
    """Paper alpha (denominator 1 + 2*a*d^2) -> implementation alpha (1 + a*d^2)."""
    return 2.0 * float(alpha_paper)


def alpha_paper_from_impl(alpha_impl: float) -> float:
    """Implementation alpha (denominator 1 + a*d^2) -> paper alpha (1 + 2*a*d^2)."""
    return 0.5 * float(alpha_impl)


def norm_freqs(n_fft: int) -> np.ndarray:
    """One-sided normalised frequency axis w = f/fs in [0, 0.5], length n_fft//2 + 1."""
    return np.linspace(0.0, 0.5, n_fft // 2 + 1)


def hz_to_norm(f_hz, fs: float) -> np.ndarray:
    """Hertz -> normalised frequency in [0, 0.5].  NOT f/fs*2."""
    return np.asarray(f_hz, dtype=np.float64) / float(fs)


def norm_to_hz(w, fs: float) -> np.ndarray:
    return np.asarray(w, dtype=np.float64) * float(fs)


def beta_from_alpha(alpha_paper: float) -> float:
    """Lorentzian half-bandwidth beta = 1/sqrt(2*alpha), in normalised frequency."""
    return 1.0 / np.sqrt(2.0 * float(alpha_paper))


def alpha_from_beta(beta: float) -> float:
    """alpha = 1/(2 beta^2), paper convention."""
    return 1.0 / (2.0 * float(beta) ** 2)


# ----------------------------------------------------------------------------------
# 2. Allocation rules (paper convention throughout)
# ----------------------------------------------------------------------------------

def lorentzian_weights(w: np.ndarray, centers: np.ndarray, alpha_paper: float) -> np.ndarray:
    """Equal-bandwidth Lorentzian soft partition, Eq. (12) of the paper (Theorem IV.3).

    s_k(w) = [1 + 2a(w-w_k)^2]^-1 / sum_j [1 + 2a(w-w_j)^2]^-1        -> (K, F)
    """
    d2 = (np.asarray(w)[:, None] - np.asarray(centers)[None, :]) ** 2
    a = 1.0 / (1.0 + 2.0 * float(alpha_paper) * d2)
    return (a / a.sum(axis=1, keepdims=True)).T


def inverse_square_weights(w: np.ndarray, centers: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Normalised inverse-square allocation, the alpha->inf limit (Corollary IV.9, Eq. (16))."""
    d = np.abs(np.asarray(w)[:, None] - np.asarray(centers)[None, :])
    d = np.maximum(d, eps)
    a = d ** -2.0
    return (a / a.sum(axis=1, keepdims=True)).T


def voronoi_weights(w: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Hard Voronoi indicator, Eq. (7) of the paper.  Ties (bins equidistant from m
    centres, to 1e-12) are shared equally, s_k = 1/m, as stated after Eq. (7);
    this is also the rule of ``softdamd.voronoi_weights``."""
    d = np.abs(np.asarray(w)[:, None] - np.asarray(centers)[None, :])
    near = np.isclose(d, d.min(axis=1, keepdims=True), rtol=0, atol=1e-12)
    return (near / near.sum(axis=1, keepdims=True)).T


def weighted_lorentzian_paper(w, centers, alpha_k, sigma2_k) -> np.ndarray:
    """Weighted Lorentzian partition, Eq. (15) of the paper (Corollary IV.7).

    s_k ∝ sigma_k^2 [1 + 2 a_k (w-w_k)^2]^-1

    Exact minimiser of the weighted cost of Eq. (14),
    (|w-w_k|^2 + beta_k^2) / (sigma_k^2 beta_k^2), and equal to the low-noise MMSE
    weights of Theorem V.2.
    """
    w = np.asarray(w, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    a_k = np.asarray(alpha_k, dtype=np.float64)
    s2 = np.asarray(sigma2_k, dtype=np.float64)
    d2 = (w[:, None] - centers[None, :]) ** 2
    num = s2[None, :] / (1.0 + 2.0 * a_k[None, :] * d2)
    return (num / num.sum(axis=1, keepdims=True)).T


def weighted_lorentzian_sigma_only(w, centers, alpha_k, sigma2_k) -> np.ndarray:
    """Minimiser of the weighted cost normalised by sigma_k^2 alone (Remark IV.8).

    Cost  c_k = (|w-w_k|^2 + 1/(2 a_k)) / sigma_k^2 ,  s_k ∝ 1/c_k, giving

        s_k ∝ sigma_k^2 * a_k * [1 + 2 a_k (w-w_k)^2]^-1

    i.e. Eq. (15) times an extra a_k.  Coincides with Eq. (15) iff a_k is constant
    in k.
    """
    w = np.asarray(w, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    a_k = np.asarray(alpha_k, dtype=np.float64)
    s2 = np.asarray(sigma2_k, dtype=np.float64)
    d2 = (w[:, None] - centers[None, :]) ** 2
    c = (d2 + 1.0 / (2.0 * a_k[None, :])) / s2[None, :]
    inv = 1.0 / c
    return (inv / inv.sum(axis=1, keepdims=True)).T


# ----------------------------------------------------------------------------------
# 3. Independent numerical ground truth
# ----------------------------------------------------------------------------------

def simplex_qp_numeric(c: np.ndarray, n_restarts: int = 3, seed: int = 0) -> np.ndarray:
    """Minimise  sum_k c_k s_k^2  over the probability simplex, numerically.

    Uses SLSQP from a generic optimiser -- deliberately NOT the closed form -- so that
    agreement with a closed form is evidence rather than a tautology.

    Parameters
    ----------
    c : (K,) strictly positive costs.

    Returns
    -------
    s : (K,) minimiser.
    """
    from scipy.optimize import minimize

    c = np.asarray(c, dtype=np.float64)
    K = c.size
    cons = ({"type": "eq", "fun": lambda s: s.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * K
    rng = np.random.default_rng(seed)

    best, best_val = None, np.inf
    starts = [np.full(K, 1.0 / K)] + [rng.dirichlet(np.ones(K)) for _ in range(n_restarts - 1)]
    for s0 in starts:
        res = minimize(lambda s: float(c @ (s ** 2)), s0,
                       jac=lambda s: 2.0 * c * s,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-14, "maxiter": 500})
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return np.asarray(best, dtype=np.float64)


# ----------------------------------------------------------------------------------
# 4. Reporting helpers
# ----------------------------------------------------------------------------------

def rel_l2(a, b) -> float:
    """||a - b|| / ||b||, guarded against a zero denominator."""
    a = np.asarray(a); b = np.asarray(b)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")


def env_report() -> dict:
    """Software/hardware provenance, printed at the start of every script."""
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        info["scipy"] = "not installed"
    try:
        import os
        info["cpu_count"] = str(os.cpu_count())
    except Exception:
        pass
    return info


@dataclass
class Section:
    """Tiny console formatter so every script prints a self-describing report."""
    title: str

    def __enter__(self):
        print("\n" + "=" * 78)
        print(self.title)
        print("=" * 78)
        return self

    def __exit__(self, *exc):
        return False


def print_env():
    print("environment:", ", ".join(f"{k}={v}" for k, v in env_report().items()))
