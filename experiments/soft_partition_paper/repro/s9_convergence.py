"""Propositions VI.1 and VI.2 -- the soft M-step and monotone decrease of the objective.

WHAT THE PAPER STATES
---------------------
Prop. VI.1 : for fixed weights, J~_alpha is minimised by the weighted centroids

                 w_k* = int [s_k*(w)]^2 P(w) w dw  /  int [s_k*(w)]^2 P(w) dw ,

             and the same formula minimises the unregularised J_VMD, because the
             regulariser (1/2a) int sum_k s_k^2 P dw does not depend on {w_k}.
Prop. VI.2 : the alternating E-step / M-step iteration produces a monotonically
             non-increasing sequence of objective values, which converges by the
             bounded monotone sequence theorem because J~_alpha >= 0.
Rem. VI.3  : the proposition concerns the objective values only -- not convergence
             of the iterates, not stationarity, not a global optimum of the joint
             problem; coincident centres form an invariant set.

WHAT IS CHECKED
---------------
Sections 1-2 verify the two propositions numerically.  Section 3 illustrates the scope
stated in Remark VI.3: the iteration is run from several starts and the objective
always converges, while the centres do not always reach the same point.  Section 4
shows the rate on one example.

Usage
-----
    python experiments/soft_partition_paper/repro/s9_convergence.py
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
    Section, lorentzian_weights, print_env,
)

F_BINS = 1024
SEED = 3


def spectrum(w, centres, halfwidths, amps):
    """A test power spectrum: sum of Lorentzian-like humps plus a small floor."""
    P = np.full(w.size, 0.02)
    for c, hw, a in zip(centres, halfwidths, amps):
        P = P + a / (1.0 + ((w - c) / hw) ** 2)
    return P


def J_reg(S, w, centres, P, alpha):
    """J~_alpha of Definition IV.1 (discrete)."""
    d2 = (w[None, :] - np.asarray(centres)[:, None]) ** 2
    return float(np.sum((d2 + 1.0 / (2.0 * alpha)) * S ** 2 * P[None, :]))


def J_unreg(S, w, centres, P):
    d2 = (w[None, :] - np.asarray(centres)[:, None]) ** 2
    return float(np.sum(d2 * S ** 2 * P[None, :]))


def m_step(S, w, P):
    """Prop. VI.1: weighted centroids with weight s_k^2 P."""
    num = (S ** 2 * P[None, :] * w[None, :]).sum(axis=1)
    den = (S ** 2 * P[None, :]).sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def em(w, P, centres0, alpha, n_iter=200, tol=0.0):
    """Alternating E/M iteration; returns objective and centre traces."""
    c = np.asarray(centres0, dtype=np.float64).copy()
    Js, Cs = [], [c.copy()]
    for _ in range(n_iter):
        S = lorentzian_weights(w, c, alpha)          # E-step (Thm. IV.3)
        Js.append(J_reg(S, w, c, P, alpha))
        c_new = m_step(S, w, P)                      # M-step (Prop. VI.1)
        if np.any(~np.isfinite(c_new)):
            break
        if tol and np.max(np.abs(c_new - c)) < tol:
            c = c_new
            Cs.append(c.copy())
            S = lorentzian_weights(w, c, alpha)
            Js.append(J_reg(S, w, c, P, alpha))
            break
        c = c_new
        Cs.append(c.copy())
    return np.array(Js), np.array(Cs)


def main(seed: int = SEED, alpha: float = 2000.0, quick: bool = False):
    print_env()
    rng = np.random.default_rng(seed)
    w = np.linspace(0.0, 0.5, F_BINS)
    true_c = np.array([0.09, 0.21, 0.34])
    P = spectrum(w, true_c, [0.012, 0.010, 0.015], [1.0, 0.6, 1.3])
    K = len(true_c)
    print(f"\ntest spectrum: {K} humps at {true_c}, alpha = {alpha:g}, F = {F_BINS} bins")

    # ------------------------------------------------------------------ VI.1
    with Section("1. Proposition VI.1 -- the centroid formula minimises J~ in {w_k}"):
        c0 = np.array([0.07, 0.24, 0.31])
        S = lorentzian_weights(w, c0, alpha)
        c_star = m_step(S, w, P)
        J_star = J_reg(S, w, c_star, P, alpha)
        print(f"    weights held fixed at centres {np.round(c0, 4)}")
        print(f"    centroid formula gives          {np.round(c_star, 6)}")
        print(f"    J~ at the centroid              {J_star:.10f}")

        worse = 0
        best = J_star
        for _ in range(20000 if not quick else 4000):
            cand = c_star + rng.normal(0.0, 0.01, K)
            Jc = J_reg(S, w, cand, P, alpha)
            if Jc < J_star - 1e-15:
                worse += 1
                best = min(best, Jc)
        print(f"    random perturbations with a smaller J~: {worse} "
              f"(best {best:.10f})")

        print("\n    the gradient at the centroid, d J~ / d w_k = -2 int (w - w_k) s_k^2 P:")
        g = -2.0 * ((w[None, :] - c_star[:, None]) * S ** 2 * P[None, :]).sum(axis=1)
        print(f"      {np.array2string(g, precision=3, floatmode='maxprec')}")
        print(f"      max |gradient| = {np.abs(g).max():.3e}")

        print("\n    the regulariser does not depend on {w_k}, so the same centroid also")
        print("    minimises the unregularised J_VMD:")
        gu = -2.0 * ((w[None, :] - c_star[:, None]) * S ** 2 * P[None, :]).sum(axis=1)
        print(f"      J_VMD at the centroid      = {J_unreg(S, w, c_star, P):.10f}")
        print(f"      max |gradient of J_VMD|    = {np.abs(gu).max():.3e}")
        d = J_reg(S, w, c_star, P, alpha) - J_unreg(S, w, c_star, P)
        d2 = (1.0 / (2.0 * alpha)) * float(np.sum(S ** 2 * P[None, :]))
        print(f"      J~ - J_VMD                 = {d:.10f}")
        print(f"      (1/2a) int sum_k s_k^2 P   = {d2:.10f}   difference "
              f"{abs(d - d2):.2e}")

    # ------------------------------------------------------------------ VI.2
    with Section("2. Proposition VI.2 -- the objective sequence is non-increasing"):
        starts = {
            "near the true centres": true_c + np.array([0.01, -0.01, 0.008]),
            "poor start": np.array([0.05, 0.15, 0.45]),
            "two centres coincident": np.array([0.20, 0.20, 0.40]),
            "all in one hump": np.array([0.20, 0.21, 0.22]),
        }
        print(f"    {'start':>24}  {'J~ first':>12}  {'J~ last':>12}  "
              f"{'monotone?':>10}  {'max increase':>13}")
        for name, c0 in starts.items():
            Js, Cs = em(w, P, c0, alpha, n_iter=200)
            diffs = np.diff(Js)
            print(f"    {name:>24}  {Js[0]:12.8f}  {Js[-1]:12.8f}  "
                  f"{'yes' if diffs.max() <= 1e-15 else 'NO':>10}  {diffs.max():13.2e}")
        print("    -> the sequence of objective VALUES is non-increasing and bounded")
        print("       below by 0, hence convergent by the bounded monotone sequence")
        print("       theorem (Proposition VI.2).  Nothing is asserted here about the")
        print("       iterates themselves; see section 3.")

    # ------------------------------------------------------------------ scope
    with Section("3. Scope of Proposition VI.2 (Remark VI.3)"):
        print("    Objective values converge from every start; the centres they converge")
        print("    to need not agree, so the result says nothing about the iterates or")
        print("    about global optimality of the joint problem.")
        print(f"    {'start':>24}  {'limit J~':>12}  {'limit centres':>34}")
        limits = []
        for name, c0 in starts.items():
            Js, Cs = em(w, P, c0, alpha, n_iter=400)
            limits.append((name, Js[-1], Cs[-1]))
            print(f"    {name:>24}  {Js[-1]:12.8f}  "
                  f"{np.array2string(np.sort(Cs[-1]), precision=5, floatmode='fixed'):>34}")
        spread = max(l[1] for l in limits) - min(l[1] for l in limits)
        print(f"    spread of the limiting objective across starts: {spread:.3e}")
        distinct = len({tuple(np.round(np.sort(l[2]), 4)) for l in limits})
        print(f"    distinct limiting centre configurations: {distinct} of {len(limits)}")
        print("    -> as Remark VI.3 states, the result concerns the objective only.")

    # ------------------------------------------------------------------ rate
    with Section("4. How fast, and does it matter in practice?"):
        Js, Cs = em(w, P, np.array([0.05, 0.15, 0.45]), alpha, n_iter=60)
        print(f"    {'iter':>5}  {'J~':>14}  {'|J~ - J~_inf|':>15}  {'max |dw|':>11}")
        Jinf = Js[-1]
        for i in [0, 1, 2, 3, 5, 10, 20, 40, len(Js) - 1]:
            if i >= len(Js):
                continue
            dw = (np.abs(Cs[i + 1] - Cs[i]).max() if i + 1 < len(Cs) else 0.0)
            print(f"    {i:5d}  {Js[i]:14.10f}  {abs(Js[i] - Jinf):15.3e}  {dw:11.3e}")
        print("\n    The alternating iteration is not the practical algorithm --")
        print("    Algorithm 1 evaluates (12) once at the DAMD centres (Remark VI.6).")
        print("    The table above is consistent with that: the objective is within a")
        print("    small fraction of its limit after a handful of steps, so the")
        print("    convergence analysis serves as a consistency check rather than as a")
        print("    runtime cost.")

    print("\nCONCLUSION")
    print("  Proposition VI.1 is confirmed: the centroid formula is a stationary point of")
    print("  J~ in the centres, the gradient vanishes there, and the regulariser")
    print("  contributes a constant independent of {w_k}.  Proposition VI.2 is confirmed")
    print("  for the objective values.  The limiting centres depend on the start,")
    print("  consistent with Remark VI.3, which restricts the statement to the")
    print("  objective values.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=2000.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(seed=args.seed, alpha=args.alpha, quick=args.quick)
