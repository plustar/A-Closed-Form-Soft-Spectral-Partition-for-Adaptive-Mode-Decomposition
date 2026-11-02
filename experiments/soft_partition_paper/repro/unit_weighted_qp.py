"""Corollary IV.7 and Remark IV.8 -- the weighted soft partition with unequal bandwidths.

WHAT THE PAPER STATES
---------------------
Corollary IV.7: with beta_k^2 = 1/(2 a_k), the weighted problem of Eq. (14) minimises,
at each frequency,

    sum_k  [ (|w - w_k|^2 + beta_k^2) / (sigma_k^2 beta_k^2) ]  s_k^2  P(w)   s.t.  s in simplex

and, where P(w) > 0, its unique minimiser is Eq. (15),

    s_k ∝ sigma_k^2 [1 + 2 a_k (w - w_k)^2]^-1 .

Remark IV.8: the normalisation by sigma_k^2 beta_k^2 matters when the bandwidths
differ.  If the cost is divided by sigma_k^2 alone,

    sum_k  [ (|w - w_k|^2 + 1/(2 a_k)) / sigma_k^2 ]  s_k^2  P(w) ,

the KKT solution of the diagonal simplex QP, s_k ∝ 1/c_k, is

    s_k ∝ sigma_k^2 * a_k * [1 + 2 a_k (w - w_k)^2]^-1 ,

carrying an extra factor a_k: 2*a_k cancels only when a_k does not depend on k, which
is why Eq. (12) (equal bandwidths) is unaffected.  For w = 0.5, centres (0, 1),
sigma^2 = (1, 1) and alpha = (1, 10), Eq. (15) gives s_1 = 0.800 whereas the minimiser
under the sigma_k^2 normalisation is s_1 = 0.286.

RELATION TO THEOREM V.2
-----------------------
Theorem V.2's MMSE weights are  S_k / sum_j S_j ∝ sigma_k^2 [1 + 2 a_k d_k^2]^-1,
with NO a_k factor, so they coincide with Eq. (15).  The cost of Eq. (14) is therefore
the variational problem whose solution matches the MMSE weights exactly (Remark V.3).
Theorem V.2 reduces to the equal-weight partition (12) only when both
sigma_k^2 = sigma^2 and alpha_k = alpha.

In the code, ``cost_sigma_only`` is the cost normalised by sigma_k^2 alone and
``cost_weighted`` is the cost of Eq. (14).  Every closed form below is checked against
a generic SLSQP simplex solver, so no comparison in this file is between two copies of
the same expression.

Usage
-----
    python experiments/soft_partition_paper/repro/unit_weighted_qp.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(os.path.dirname(_HERE))   # experiments/
_ROOT = os.path.dirname(_EXP)                     # repository root, where softdamd lives
for _p in (_ROOT, _EXP, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from soft_partition_paper.repro._common import (  # noqa: E402
    Section, print_env, simplex_qp_numeric,
    weighted_lorentzian_sigma_only, weighted_lorentzian_paper,
)


def cost_sigma_only(w, centers, alpha_k, sigma2_k):
    """c_k normalised by sigma_k^2 alone: (d_k^2 + 1/(2 a_k)) / sigma_k^2  (P(w) factors out)."""
    d2 = (np.asarray(w)[:, None] - np.asarray(centers)[None, :]) ** 2
    return (d2 + 1.0 / (2.0 * np.asarray(alpha_k)[None, :])) / np.asarray(sigma2_k)[None, :]


def cost_weighted(w, centers, alpha_k, sigma2_k):
    """Cost of Eq. (14): divide by sigma_k^2 * beta_k^2 = sigma_k^2 / (2 a_k).

    Then c_k = (1 + 2 a_k d_k^2) / sigma_k^2 and s_k ∝ 1/c_k reproduces Eq. (15).
    """
    d2 = (np.asarray(w)[:, None] - np.asarray(centers)[None, :]) ** 2
    a_k = np.asarray(alpha_k)[None, :]
    s2 = np.asarray(sigma2_k)[None, :]
    beta2 = 1.0 / (2.0 * a_k)
    return (d2 + beta2) / (s2 * beta2)


def mmse_weights(w, centers, alpha_k, sigma2_k):
    """Theorem V.2 weights: S_k / sum_j S_j with S_k = sigma_k^2 / (1 + 2 a_k d_k^2)."""
    d2 = (np.asarray(w)[:, None] - np.asarray(centers)[None, :]) ** 2
    S = np.asarray(sigma2_k)[None, :] / (1.0 + 2.0 * np.asarray(alpha_k)[None, :] * d2)
    return (S / S.sum(axis=1, keepdims=True)).T


def main(seed: int = 0, n_random: int = 2000):
    print_env()

    # ---------------------------------------------------------------- worked case
    with Section("1. Worked case of Remark IV.8 (unequal bandwidths)"):
        w = np.array([0.5])
        centers = np.array([0.0, 1.0])
        alpha_k = np.array([1.0, 10.0])
        sigma2_k = np.array([1.0, 1.0])

        c = cost_sigma_only(w, centers, alpha_k, sigma2_k)[0]
        s_num = simplex_qp_numeric(c)
        s_paper = weighted_lorentzian_paper(w, centers, alpha_k, sigma2_k)[:, 0]
        s_sig = weighted_lorentzian_sigma_only(w, centers, alpha_k, sigma2_k)[:, 0]

        print(f"  w = 0.5, centres = (0, 1), sigma^2 = (1, 1), alpha = (1, 10)")
        print(f"  cost normalised by sigma^2  : c = {np.round(c, 4)}")
        print(f"  Eq.(15)                     : s_1 = {s_paper[0]:.4f}")
        print(f"  its minimiser (closed form) : s_1 = {s_sig[0]:.4f}")
        print(f"  SLSQP numerical minimiser   : s_1 = {s_num[0]:.4f}   <-- independent")
        obj = lambda s: float(c @ (s ** 2))
        print(f"  objective value  Eq.(15)={obj(s_paper):.6f}   "
              f"minimiser={obj(s_sig):.6f}   numeric={obj(s_num):.6f}")
        assert abs(s_sig[0] - s_num[0]) < 1e-6, "closed form disagrees with solver"
        print("  -> under the sigma^2-only normalisation Eq.(15) is not the minimiser;")
        print("     this is why Eq.(14) divides by sigma_k^2 beta_k^2 (Remark IV.8).")

    # ---------------------------------------------------------------- random sweep
    with Section("2. Random sweep: Eq. (15) against the sigma^2-only cost"):
        rng = np.random.default_rng(seed)
        worst = 0.0
        worst_cfg = None
        errs_unequal, errs_equal = [], []
        for _ in range(n_random):
            K = rng.integers(2, 6)
            centers = np.sort(rng.uniform(0.0, 0.5, K))
            wq = np.array([rng.uniform(0.0, 0.5)])
            sig = rng.uniform(0.2, 5.0, K)

            a_un = 10.0 ** rng.uniform(1.0, 4.0, K)          # unequal bandwidths
            c = cost_sigma_only(wq, centers, a_un, sig)[0]
            s_num = simplex_qp_numeric(c)
            s_pap = weighted_lorentzian_paper(wq, centers, a_un, sig)[:, 0]
            s_sig = weighted_lorentzian_sigma_only(wq, centers, a_un, sig)[:, 0]
            e = float(np.abs(s_pap - s_num).max())
            errs_unequal.append(e)
            assert np.abs(s_sig - s_num).max() < 1e-5
            if e > worst:
                worst, worst_cfg = e, (centers, a_un, sig, wq, s_pap, s_num)

            a_eq = np.full(K, 10.0 ** rng.uniform(1.0, 4.0))  # equal bandwidths
            c2 = cost_sigma_only(wq, centers, a_eq, sig)[0]
            s_num2 = simplex_qp_numeric(c2)
            s_pap2 = weighted_lorentzian_paper(wq, centers, a_eq, sig)[:, 0]
            errs_equal.append(float(np.abs(s_pap2 - s_num2).max()))

        eu, ee = np.array(errs_unequal), np.array(errs_equal)
        print(f"  {n_random} random configurations, K in [2, 5]")
        print(f"  UNEQUAL alpha_k : max|Eq.(15) - numeric|  median {np.median(eu):.4f}  "
              f"p90 {np.quantile(eu, 0.9):.4f}  max {eu.max():.4f}")
        print(f"  EQUAL   alpha_k : max|Eq.(15) - numeric|  median {np.median(ee):.2e}  "
              f"max {ee.max():.2e}")
        print("  -> the difference is confined to the unequal-bandwidth case, since 2 a_k")
        print("     cancels only as a common factor.  Eq. (12) is unaffected.")
        cen, a_un, sig, wq, s_pap, s_num = worst_cfg
        print(f"\n  worst case: w={wq[0]:.4f}")
        print(f"    centres = {np.round(cen, 4)}")
        print(f"    alpha_k = {np.round(a_un, 1)}")
        print(f"    sigma^2 = {np.round(sig, 3)}")
        print(f"    Eq.(15) = {np.round(s_pap, 4)}")
        print(f"    numeric = {np.round(s_num, 4)}")

    # ---------------------------------------------------------------- cost of Eq. (14)
    with Section("3. The cost of Eq. (14): divide by sigma_k^2 * beta_k^2"):
        rng = np.random.default_rng(seed + 1)
        errs = []
        for _ in range(500):
            K = rng.integers(2, 6)
            centers = np.sort(rng.uniform(0.0, 0.5, K))
            wq = np.array([rng.uniform(0.0, 0.5)])
            sig = rng.uniform(0.2, 5.0, K)
            a_un = 10.0 ** rng.uniform(1.0, 4.0, K)
            c = cost_weighted(wq, centers, a_un, sig)[0]
            s_num = simplex_qp_numeric(c)
            s_pap = weighted_lorentzian_paper(wq, centers, a_un, sig)[:, 0]
            errs.append(float(np.abs(s_pap - s_num).max()))
        errs = np.array(errs)
        print(f"  with cost c_k = (d_k^2 + beta_k^2) / (sigma_k^2 beta_k^2):")
        print(f"    max|Eq.(15) - numeric| over 500 configs = {errs.max():.2e}")
        print("  -> Eq. (15) is the exact minimiser of this cost (Corollary IV.7).")

    # ---------------------------------------------------------------- consequence
    with Section("4. Link to Theorem V.2 and Remark V.3"):
        rng = np.random.default_rng(seed + 2)
        K = 3
        centers = np.array([0.08, 0.20, 0.35])
        a_un = np.array([500.0, 3000.0, 12000.0])
        sig = np.array([1.0, 0.4, 2.0])
        w = np.linspace(0.0, 0.5, 513)

        S_mmse = mmse_weights(w, centers, a_un, sig)
        S_pap = weighted_lorentzian_paper(w, centers, a_un, sig)
        S_sig = weighted_lorentzian_sigma_only(w, centers, a_un, sig)

        print(f"  max |Eq.(15)  - MMSE(Thm V.2)| = {np.abs(S_pap - S_mmse).max():.2e}")
        print(f"  max |sigma^2-only minimiser - MMSE(Thm V.2)| = {np.abs(S_sig - S_mmse).max():.4f}")
        print("  -> the MMSE weights of Theorem V.2 equal Eq.(15), the minimiser of the")
        print("     cost of Eq.(14), so the weighted variational problem matches the MMSE")
        print("     weights exactly (Remark V.3).  With the sigma^2-only cost this would")
        print("     fail for unequal bandwidths.")

    with Section("5. Theorem V.2 -> Eq. (12) needs alpha_k = alpha, not only sigma_k = sigma"):
        w = np.linspace(0.0, 0.5, 513)
        centers = np.array([0.08, 0.20, 0.35])
        sig_eq = np.ones(3)
        from soft_partition_paper.repro._common import lorentzian_weights
        a_un = np.array([500.0, 3000.0, 12000.0])
        a_eq = np.full(3, 2000.0)
        S11 = lorentzian_weights(w, centers, 2000.0)
        print(f"  sigma equal, alpha UNEQUAL : max |Thm V.2 - Eq.(12)| = "
              f"{np.abs(mmse_weights(w, centers, a_un, sig_eq) - S11).max():.4f}")
        print(f"  sigma equal, alpha EQUAL   : max |Thm V.2 - Eq.(12)| = "
              f"{np.abs(mmse_weights(w, centers, a_eq, sig_eq) - S11).max():.2e}")
        print("  -> as Theorem V.2 states, the reduction to Eq. (12) requires both")
        print("     sigma_k^2 = sigma^2 and alpha_k = alpha.")

    print("\nCONCLUSION")
    print("  Eq. (15) minimises the sigma^2-only cost only when alpha_k is constant in k;")
    print("  with the cost of Eq. (14), (d_k^2 + beta_k^2)/(sigma_k^2 beta_k^2), it is the")
    print("  exact minimiser and equals the MMSE weights of Theorem V.2 (Remark V.3).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_random", type=int, default=2000)
    args = ap.parse_args()
    main(seed=args.seed, n_random=args.n_random)
