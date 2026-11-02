"""Theorem III.1, Corollary III.2, Remark VI.4 -- the algebraic foundation.

WHAT THE PAPER STATES
---------------------
Thm. III.1 : for an allocation fixed independently of the signal, the identity
             |w-w_k|^2 s_k^2 == |w-w_k|^2 s_k holds for all k and almost all w
             **iff** s_k(w) in {0,1}; equivalently J_VMD = J_Kmeans for every
             signal X iff the allocation is binary.
Cor. III.2 : if s_k(w) in (0,1) on a set of positive measure where P(w) > 0 and
             |w - w_k| > 0, then J_VMD < J_Kmeans strictly.
Sec. III-C : the equivalence applies at the vertices of the feasible set (the
             simplex); for K >= 3 the boundary also contains non-binary points.
Rem. VI.4  : minimising the *unregularised* J_VMD over the simplex gives the
             inverse-square allocation (31), not the Lorentzian.

WHAT IS CHECKED
---------------
Sections 1-2: both directions of Theorem III.1 and the strict inequality of
    Corollary III.2, on random signals, centres and allocations.
Section 3 (quantifier over signals): the converse needs equality *for every signal*.
    Equality for one particular X does not force a binary allocation -- section 3
    exhibits a non-binary allocation and a signal for which the two objectives agree
    exactly.
Section 4 (vertices): for K >= 3 the simplex boundary contains non-binary points.
    Section 4 evaluates one and shows the strict inequality holds there, so the
    equivalence is confined to the K vertices.
Section 5 (Rem. VI.4): the Lorentzian shape comes from the regulariser, not from the
    VMD objective itself.

Usage
-----
    python experiments/soft_partition_paper/repro/s5_algebraic.py
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


def J_vmd(S, w, centres, P):
    """sum_k int |w-w_k|^2 s_k(w)^2 P(w) dw, on the discrete grid."""
    d2 = (w[None, :] - np.asarray(centres)[:, None]) ** 2
    return float(np.sum(d2 * S ** 2 * P[None, :]))


def J_kmeans(S, w, centres, P):
    """sum_k int |w-w_k|^2 s_k(w) P(w) dw."""
    d2 = (w[None, :] - np.asarray(centres)[:, None]) ** 2
    return float(np.sum(d2 * S * P[None, :]))


def random_simplex_rows(rng, K, F):
    """A random feasible allocation: (K, F), columns on the simplex."""
    S = rng.dirichlet(np.ones(K), size=F).T
    return S


def main(seed: int = 0, n_random: int = 500):
    print_env()
    rng = np.random.default_rng(seed)
    w = np.linspace(0.0, 0.5, 1024)
    centres = np.array([0.08, 0.20, 0.35])
    K = len(centres)

    # ------------------------------------------------------------------ forward
    with Section("1. Theorem III.1, forward direction: binary => the two objectives agree"):
        print("    Random signals and random centre configurations, hard Voronoi mask.")
        gaps = []
        for _ in range(n_random):
            Kx = int(rng.integers(2, 7))
            c = np.sort(rng.uniform(0.0, 0.5, Kx))
            P = rng.random(w.size) ** 2
            S = voronoi_weights(w, c)
            gaps.append(abs(J_vmd(S, w, c, P) - J_kmeans(S, w, c, P)))
        gaps = np.array(gaps)
        print(f"    {n_random} draws, K in [2, 6]")
        print(f"    |J_VMD - J_Kmeans| : median {np.median(gaps):.2e}   max {gaps.max():.2e}")
        print("    -> the identity holds exactly, as s^2 = s pointwise for binary s.")

    # ------------------------------------------------------------------ converse
    with Section("2. Corollary III.2: any interior allocation makes J_VMD strictly smaller"):
        P = np.exp(-((w - 0.20) / 0.05) ** 2) + 0.1
        print(f"    {'allocation':>28}  {'J_VMD':>12}  {'J_Kmeans':>12}  {'J_VMD < J_KM':>13}")
        cases = {
            "hard Voronoi (vertex)": voronoi_weights(w, centres),
            "Lorentzian, alpha = 2000": lorentzian_weights(w, centres, 2000.0),
            "Lorentzian, alpha = 100": lorentzian_weights(w, centres, 100.0),
            "uniform 1/K": np.full((K, w.size), 1.0 / K),
            "random interior": random_simplex_rows(rng, K, w.size),
        }
        for name, S in cases.items():
            jv, jk = J_vmd(S, w, centres, P), J_kmeans(S, w, centres, P)
            print(f"    {name:>28}  {jv:12.6f}  {jk:12.6f}  "
                  f"{'yes' if jv < jk - 1e-12 else 'equal':>13}")
        print("    -> strict inequality for every non-binary allocation tested; equality")
        print("       only at the vertex.")

        n_strict = 0
        for _ in range(n_random):
            S = random_simplex_rows(rng, K, w.size)
            if J_vmd(S, w, centres, P) < J_kmeans(S, w, centres, P) - 1e-12:
                n_strict += 1
        print(f"    {n_strict}/{n_random} random interior allocations satisfy the strict")
        print("    inequality (expected: all of them).")

    # ------------------------------------------------------------------ quantifiers
    with Section("3. Equality for ONE signal does not force a binary allocation"):
        print("    The converse in Theorem III.1 needs equality for EVERY signal.  Here is")
        print("    a non-binary allocation and a signal for which the two objectives are")
        print("    nevertheless exactly equal: the allocation is binary wherever P > 0 and")
        print("    non-binary only where the signal has no energy.")
        S = voronoi_weights(w, centres)
        band = (w > 0.42) & (w < 0.48)
        S[:, band] = 1.0 / K                      # non-binary, off the signal support
        P = np.zeros_like(w)
        supp = w < 0.40
        P[supp] = np.exp(-((w[supp] - 0.20) / 0.05) ** 2) + 0.1

        jv, jk = J_vmd(S, w, centres, P), J_kmeans(S, w, centres, P)
        frac = float(np.mean(np.any((S > 1e-12) & (S < 1 - 1e-12), axis=0)))
        print(f"    fraction of bins where the allocation is non-binary : {frac:.3f}")
        print(f"    J_VMD = {jv:.10f}   J_Kmeans = {jk:.10f}   difference = {jv - jk:.2e}")
        print("    -> this is why Theorem III.1 quantifies the converse over all signals")
        print("       and takes the allocation to be fixed independently of the signal.")

    # ------------------------------------------------------------------ vertices
    with Section("4. The identity holds at the vertices, not on the whole boundary"):
        print("    For K >= 3 the simplex boundary contains non-binary points.  Example at")
        print("    K = 3: s = (1/2, 1/2, 0) lies on the boundary (s_3 = 0) and is not a")
        print("    vertex.")
        P = np.exp(-((w - 0.20) / 0.05) ** 2) + 0.1
        S = np.zeros((K, w.size))
        S[0] = 0.5
        S[1] = 0.5
        S[2] = 0.0
        jv, jk = J_vmd(S, w, centres, P), J_kmeans(S, w, centres, P)
        print(f"    boundary point (1/2, 1/2, 0):  J_VMD = {jv:.6f}   "
              f"J_Kmeans = {jk:.6f}")
        print(f"    J_VMD < J_Kmeans : {jv < jk - 1e-12}  -> the identity fails on the")
        print("    boundary away from the vertices, as Section III-C states.")

        print("\n    Where on the simplex does the identity hold?  200 random draws each:")
        F = w.size

        def draw_vertex():
            S = np.zeros((K, F))
            S[rng.integers(0, K, F), np.arange(F)] = 1.0
            return S

        def draw_face():                       # s_3 = 0, s_1 + s_2 = 1, non-binary
            a = rng.uniform(0.05, 0.95, F)
            return np.vstack([a, 1.0 - a, np.zeros(F)])

        def draw_interior():
            return random_simplex_rows(rng, K, F)

        for label, gen in (("vertices (binary)", draw_vertex),
                           ("face s_3 = 0", draw_face),
                           ("interior", draw_interior)):
            eq = sum(abs(J_vmd(g := gen(), w, centres, P)
                         - J_kmeans(g, w, centres, P)) < 1e-12 for _ in range(200))
            print(f"      {label:20s}: identity holds in {eq:3d}/200 draws")
        print("      -> only the vertices.")

    # ------------------------------------------------------------------ Rem VI.4
    with Section("5. Remark VI.4: without the regulariser the minimiser is inverse-square"):
        print("    Minimising the unregularised J_VMD per frequency over the simplex, with")
        print("    a generic solver, and comparing with d_k^-2 / sum_j d_j^-2 and with the")
        print("    Lorentzian.")
        print(f"    {'w':>8}  {'max|num - inverse-sq|':>22}  {'max|num - Lorentzian(2000)|':>28}")
        for w_pt in [0.05, 0.12, 0.19, 0.27, 0.44]:
            c = (w_pt - centres) ** 2                       # no 1/(2 alpha) term
            c = np.maximum(c, 1e-18)
            s_num = simplex_qp_numeric(c)
            s_inv = inverse_square_weights(np.array([w_pt]), centres)[:, 0]
            s_lor = lorentzian_weights(np.array([w_pt]), centres, 2000.0)[:, 0]
            print(f"    {w_pt:8.3f}  {np.abs(s_num - s_inv).max():22.3e}  "
                  f"{np.abs(s_num - s_lor).max():28.3e}")
        print("    -> the Lorentzian shape is produced by the 1/(2 alpha) offset of")
        print("       Definition IV.1, not by the VMD objective.  Any strictly positive")
        print("       offset removes the singularity at w = w_k; 1/(2 alpha) is the")
        print("       particular choice whose minimiser is the Lorentzian form")
        print("       (Remark IV.2).")

    print("\nCONCLUSION")
    print("  Theorem III.1 and Corollary III.2 are confirmed numerically.  The converse")
    print("  of Theorem III.1 requires equality for every signal (section 3), and the")
    print("  equivalence holds at the simplex vertices only, not on the rest of the")
    print("  boundary (section 4).  Remark VI.4 is confirmed: without the regulariser")
    print("  the minimiser is inverse-square, which locates the origin of the")
    print("  Lorentzian shape in the regulariser.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_random", type=int, default=500)
    args = ap.parse_args()
    main(seed=args.seed, n_random=args.n_random)
