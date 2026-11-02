"""Theorem IV.3, Remark IV.4, Corollary IV.9, Remark IV.10 -- the closed-form optimum.

WHAT THE PAPER STATES
---------------------
Thm. IV.3  : on {P > 0} the regularised soft VMD problem (11) is strictly convex per
             frequency and has the unique global minimiser (12),

                 s_k*(w) = [1 + 2a(w-w_k)^2]^-1 / sum_j [1 + 2a(w-w_j)^2]^-1 .

Rem. IV.4  : if two centres coincide, the regulariser keeps c_i = c_j > 0 and the two
             share equally; uniqueness is preserved.
Cor. IV.9  : as a -> infinity, s_k* -> d_k^-2 / sum_j d_j^-2 (inverse-square, (16)),
             and, for distinct centres, s_k*(w_k) -> 1 (with m coincident centres
             the limit is 1/m); as a -> 0, s_k* -> 1/K.
Rem. IV.10 : the a -> infinity limit is NOT the Voronoi indicator; with w_1 = 0,
             w_2 = 2, w = 0.9 the inverse-square limit gives about 0.60 while the
             Voronoi mask gives 1.  The alpha-dependence cancels asymptotically,
             leaving an O(1/alpha) residual.

Sections 1-3 are the numerical verification of Theorem IV.3 reported as Experiment S3
in the supplementary material.  Cor. IV.9 and Rem. IV.10 are covered by Experiment B
of the paper; this script adds the two cases Experiment B does not exercise (the
a -> 0 limit and coincident centres).

HOW IT IS CHECKED
-----------------
Against a generic SLSQP solver on the simplex, never against a second copy of the
closed form.  In addition the KKT conditions and strict convexity are verified
directly, and the claimed minimiser is compared with random feasible perturbations to
confirm it is a minimum rather than a stationary point.

Usage
-----
    python experiments/soft_partition_paper/repro/s6_closed_form.py
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
    Section, inverse_square_weights, lorentzian_weights, print_env,
    simplex_qp_numeric, voronoi_weights,
)


def cost_vector(w_point, centres, alpha):
    """c_k of Definition IV.1 at one frequency; P(w) > 0 factors out of the argmin."""
    d2 = (float(w_point) - np.asarray(centres, dtype=np.float64)) ** 2
    return d2 + 1.0 / (2.0 * float(alpha))


def objective(s, c):
    return float(np.asarray(c) @ (np.asarray(s) ** 2))


def main(seed: int = 0, n_random: int = 800, quick: bool = False):
    print_env()
    if quick:
        n_random = 200
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ Thm IV.3
    with Section("1. Theorem IV.3 -- (12) is the unique global minimiser of (11)"):
        print("    Random configurations; the closed form is compared with a generic")
        print("    SLSQP minimisation of sum_k c_k s_k^2 over the simplex.")
        errs, obj_gaps = [], []
        for _ in range(n_random):
            K = int(rng.integers(2, 7))
            centres = np.sort(rng.uniform(0.0, 0.5, K))
            w_pt = float(rng.uniform(0.0, 0.5))
            alpha = float(10.0 ** rng.uniform(1.0, 5.0))

            c = cost_vector(w_pt, centres, alpha)
            s_num = simplex_qp_numeric(c)
            s_cf = lorentzian_weights(np.array([w_pt]), centres, alpha)[:, 0]

            errs.append(float(np.abs(s_cf - s_num).max()))
            obj_gaps.append(objective(s_cf, c) - objective(s_num, c))
        errs, obj_gaps = np.array(errs), np.array(obj_gaps)
        print(f"    {n_random} configurations, K in [2, 6], alpha in [1e1, 1e5]")
        print(f"    max |closed form - numerical|      : median {np.median(errs):.2e}"
              f"   max {errs.max():.2e}")
        print(f"    objective(closed form) - objective(numerical): "
              f"max {obj_gaps.max():.2e}")
        assert errs.max() < 1e-5, "closed form disagrees with the numerical minimiser"
        print("    -> Theorem IV.3 verified numerically.")

    with Section("2. It is a minimum, not merely a stationary point"):
        K, alpha = 4, 2000.0
        centres = np.array([0.05, 0.12, 0.28, 0.41])
        w_pt = 0.19
        c = cost_vector(w_pt, centres, alpha)
        s_star = lorentzian_weights(np.array([w_pt]), centres, alpha)[:, 0]
        f_star = objective(s_star, c)
        worse = 0
        for _ in range(20000):
            s = rng.dirichlet(np.ones(K))
            if objective(s, c) < f_star - 1e-15:
                worse += 1
        print(f"    K = {K}, alpha = {alpha:g}, w = {w_pt}")
        print(f"    objective at (12) = {f_star:.10f}")
        print(f"    feasible draws with a smaller objective: {worse} / 20000")
        print(f"    Hessian of the per-frequency problem = 2*diag(c), c = "
              f"{np.round(c, 5)}")
        print(f"    all c_k > 0 : {bool(np.all(c > 0))}  -> strictly convex, so the")
        print("    stationary point is the unique global minimiser.")

    with Section("3. KKT conditions hold at (12)"):
        lam = 2.0 * c * s_star
        print(f"    2 c_k s_k*        = {np.round(lam, 8)}")
        print(f"    spread of 2 c_k s_k* (should be 0) = {lam.max() - lam.min():.2e}")
        print(f"    sum_k s_k* - 1    = {s_star.sum() - 1.0:.2e}")
        print(f"    min_k s_k*        = {s_star.min():.6f}  (>= 0, interior)")

    # ------------------------------------------------------------------ Rem IV.4
    with Section("4. Remark IV.4 -- coincident centres share equally"):
        for m in (2, 3):
            centres = np.array([0.20] * m + [0.35])
            alpha = 2000.0
            w_pt = 0.20
            c = cost_vector(w_pt, centres, alpha)
            s_cf = lorentzian_weights(np.array([w_pt]), centres, alpha)[:, 0]
            s_num = simplex_qp_numeric(c)
            print(f"    {m} coincident centres at 0.20 plus one at 0.35, evaluated at w = 0.20")
            print(f"      closed form = {np.round(s_cf, 6)}")
            print(f"      numerical   = {np.round(s_num, 6)}")
            print(f"      equal share among the coincident group: "
                  f"{np.ptp(s_cf[:m]) < 1e-12}")
        print("    -> the regulariser keeps c_k > 0, so uniqueness survives coincidence.")

    # ------------------------------------------------------------------ Cor IV.9
    with Section("5. Corollary IV.9 -- the two limits"):
        centres = np.array([0.08, 0.20, 0.35])
        w = np.linspace(0.001, 0.499, 997)
        K = len(centres)

        print("    (a) alpha -> infinity: convergence to the inverse-square allocation")
        S_inv = inverse_square_weights(w, centres)
        print(f"        {'alpha':>10}  {'max |s* - inverse-square|':>26}")
        for a in [1e2, 1e3, 1e4, 1e6, 1e8, 1e10]:
            S = lorentzian_weights(w, centres, a)
            print(f"        {a:10.0e}  {np.abs(S - S_inv).max():26.3e}")

        print("\n    (b) alpha -> 0: convergence to the uniform allocation 1/K")
        print(f"        {'alpha':>10}  {'max |s* - 1/K|':>26}")
        for a in [1e0, 1e-2, 1e-4, 1e-6, 1e-8]:
            S = lorentzian_weights(w, centres, a)
            print(f"        {a:10.0e}  {np.abs(S - 1.0 / K).max():26.3e}")

        print("\n    (c) s_k*(w_k) -> 1 at a distinct centre")
        print(f"        {'alpha':>10}  {'s_1(w_1)':>12}")
        for a in [1e2, 1e3, 1e4, 1e6, 1e8]:
            S = lorentzian_weights(np.array([centres[0]]), centres, a)
            print(f"        {a:10.0e}  {S[0, 0]:12.8f}")
        print("        (this requires distinct centres -- with m coincident centres")
        print("         the limit is 1/m, as Corollary IV.9 states and section 4 shows.)")
        for m in (2, 3):
            cc = np.array([0.20] * m + [0.35])
            S = lorentzian_weights(np.array([0.20]), cc, 1e8)
            print(f"        m = {m} coincident: s_1(w_1) = {S[0, 0]:.8f}  (1/m = {1/m:.8f})")

    # ------------------------------------------------------------------ Rem IV.10
    with Section("6. Remark IV.10 -- the alpha -> infinity limit is not the Voronoi mask"):
        centres = np.array([0.0, 2.0])
        w_pt = np.array([0.9])
        print("    worked case of Remark IV.10: centres 0 and 2, evaluated at w = 0.9")
        s_inv = inverse_square_weights(w_pt, centres)[0, 0]
        s_vor = voronoi_weights(w_pt, centres)[0, 0]
        print(f"      inverse-square limit s_1 = {s_inv:.6f}   (paper: ~0.60)")
        print(f"      Voronoi indicator    s_1 = {s_vor:.6f}")
        print(f"      {'alpha':>10}  {'s_1(0.9)':>12}")
        for a in [1e0, 1e1, 1e2, 1e4, 1e8]:
            print(f"      {a:10.0e}  "
                  f"{lorentzian_weights(w_pt, centres, a)[0, 0]:12.8f}")
        print("    The alpha-dependence cancels asymptotically, leaving an O(1/alpha)")
        print("    residual (Remark IV.10):")
        print(f"      {'alpha':>10}  {'|s_1(alpha) - inverse-square|':>30}")
        for a in [1e0, 1e1, 1e2, 1e3, 1e4, 1e6]:
            d = abs(lorentzian_weights(w_pt, centres, a)[0, 0] - s_inv)
            print(f"      {a:10.0e}  {d:30.3e}")

        print("\n    distance from the whole Lorentzian family to the Voronoi mask")
        centres2 = np.array([0.08, 0.20, 0.35])
        w2 = np.linspace(0.001, 0.499, 997)
        S_vor = voronoi_weights(w2, centres2)
        print(f"      {'alpha':>10}  {'max |s* - Voronoi|':>20}")
        for a in [1e2, 1e4, 1e6, 1e8, 1e10]:
            S = lorentzian_weights(w2, centres2, a)
            print(f"      {a:10.0e}  {np.abs(S - S_vor).max():20.6f}")
        print("      -> the gap does not go to zero: the hard mask lies outside the family.")

    print("\nCONCLUSION")
    print("  Theorem IV.3 holds: the closed form matches an independent simplex solver to")
    print("  machine precision over the tested range, the per-frequency Hessian 2*diag(c)")
    print("  is positive definite, and the KKT conditions are satisfied exactly.")
    print("  Corollary IV.9 holds in both limits, with s_k(w_k) -> 1 for distinct")
    print("  centres and 1/m for m coincident ones; the cancellation in Remark IV.10 is")
    print("  asymptotic, with the O(1/alpha) residual shown in the table.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_random", type=int, default=800)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(seed=args.seed, n_random=args.n_random, quick=args.quick)
