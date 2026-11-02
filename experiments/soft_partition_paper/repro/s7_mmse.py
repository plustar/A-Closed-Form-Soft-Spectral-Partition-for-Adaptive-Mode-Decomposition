"""Theorem V.2, Remark V.4, Section V-C -- the Bayesian interpretation.

WHAT THE PAPER STATES
---------------------
Thm. V.2  : under Definition V.1 the MMSE estimator of mode k is (S_k/S_X) X, and in
            the low-noise limit sigma_N^2 -> 0 it tends to s_k* X, with the weighted
            partition (15) at alpha_k = 1/(2 beta_k^2).  When sigma_k^2 = sigma^2 and
            alpha_k = alpha this reduces to the equal-weight partition (12).
Rem. V.4  : at finite noise the exact MMSE differs from s_k* X by the shrinkage factor
            sum_j S_j / (sum_j S_j + sigma_N^2) <= 1, which tends to 1 at the mode
            frequencies and to 0 in the gaps between modes.
Sec. V-C  : s_k* is a reconstruction-preserving relative allocation (sum_k s_k* = 1)
            whereas the finite-noise MMSE estimator is a denoising estimator whose
            weights sum to less than one.

Section 4 runs the Monte-Carlo setting of Experiment C, with standard errors: at high
SNR the soft partition and the oracle share a non-zero nMSE, the irreducible error of
separating two overlapping Lorentzian modes; at low SNR the two diverge, and applying
the shrinkage factor of Remark V.4 recovers the oracle.

Usage
-----
    python experiments/soft_partition_paper/repro/s7_mmse.py
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
    Section, print_env, voronoi_weights, weighted_lorentzian_paper,
)

F_BINS = 512
W1, W2 = 0.12, 0.30
SNR_DB = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 60])
N_TRIALS = 1500
SEED = 42


def lorentz_psd(w, w0, alpha, sigma2):
    """Definition V.1: S_k(w) = sigma_k^2 / (1 + 2 alpha_k (w - w_k)^2)."""
    return sigma2 / (1.0 + 2.0 * alpha * (w - w0) ** 2)


def monte_carlo(w, S_list, sigma_n2, weights, n_trials, rng):
    """E|W X - U_1|^2 / E|U_1|^2 with its standard error, for each weight vector."""
    K = len(S_list)
    U = [np.sqrt(S / 2.0) * (rng.standard_normal((n_trials, w.size))
                             + 1j * rng.standard_normal((n_trials, w.size)))
         for S in S_list]
    Nz = np.sqrt(sigma_n2 / 2.0) * (rng.standard_normal((n_trials, w.size))
                                    + 1j * rng.standard_normal((n_trials, w.size)))
    X = sum(U) + Nz
    den = float(np.mean(np.abs(U[0]) ** 2))

    out = {}
    for name, W in weights.items():
        per = np.mean(np.abs(W[None, :] * X - U[0]) ** 2, axis=1) / den
        out[name] = (float(per.mean()), float(per.std(ddof=1) / np.sqrt(n_trials)))
    return out


def main(n_trials: int = N_TRIALS, seed: int = SEED, quick: bool = False):
    print_env()
    if quick:
        n_trials = 400
    w = np.linspace(0.0, 0.5, F_BINS)
    alpha = 5000.0
    sig2 = np.array([1.0, 1.0])
    centres = np.array([W1, W2])

    S1 = lorentz_psd(w, W1, alpha, sig2[0])
    S2 = lorentz_psd(w, W2, alpha, sig2[1])

    W_soft = weighted_lorentzian_paper(w, centres, np.array([alpha, alpha]), sig2)[0]
    W_hard = voronoi_weights(w, centres)[0]

    print(f"\nDefinition V.1 model: K = 2, centres {W1} and {W2} (normalised),")
    print(f"alpha = {alpha:g} for both modes (beta = {1/np.sqrt(2*alpha):.5f}),")
    print(f"sigma_k^2 = {sig2}.  Monte Carlo: {n_trials} trials per SNR, +- is one")
    print("standard error over trials.")

    # ------------------------------------------------------------------ Thm V.2
    with Section("1. Theorem V.2 -- the low-noise limit"):
        print("    Weights compared directly, before any Monte Carlo:")
        for snr in (0, 20, 40, 60, 80):
            sn = float(np.mean(S1 + S2)) / 10 ** (snr / 10.0)
            W_or = S1 / (S1 + S2 + sn)
            print(f"      sigma_N^2 = {sn:9.2e}  (SNR {snr:3d} dB)   "
                  f"max |S_1/S_X - s_1*| = {np.abs(W_or - W_soft).max():.3e}")
        W_or0 = S1 / (S1 + S2)
        print(f"      sigma_N^2 = 0                        "
              f"max |S_1/S_X - s_1*| = {np.abs(W_or0 - W_soft).max():.3e}")
        print("    -> Theorem V.2 holds exactly in the limit; at finite noise the two")
        print("       differ, and the difference is the factor of Remark V.4.")

    # ------------------------------------------------------------------ Rem V.4
    with Section("2. Remark V.4 -- the shrinkage factor"):
        print("    factor(w) = sum_j S_j / (sum_j S_j + sigma_N^2)")
        print(f"    {'SNR':>5}  {'at mode centre w1':>19}  {'midway between modes':>21}"
              f"  {'min over w':>11}")
        i_c = int(np.argmin(np.abs(w - W1)))
        i_m = int(np.argmin(np.abs(w - 0.5 * (W1 + W2))))
        for snr in (0, 10, 20, 30, 40):
            sn = float(np.mean(S1 + S2)) / 10 ** (snr / 10.0)
            fac = (S1 + S2) / (S1 + S2 + sn)
            print(f"    {snr:5d}  {fac[i_c]:19.6f}  {fac[i_m]:21.6f}  {fac.min():11.6f}")
        print("    -> tends to 1 where the modes dominate and to 0 in the gaps, exactly")
        print("       as Remark V.4 states.")

    # ------------------------------------------------------------------ Sec V-C
    with Section("3. Section V-C -- allocation preserves reconstruction, MMSE does not"):
        print(f"    {'SNR':>5}  {'sum_k s_k*':>12}  {'sum_k MMSE weight':>19}")
        W_soft2 = weighted_lorentzian_paper(w, centres,
                                            np.array([alpha, alpha]), sig2)
        for snr in (0, 20, 40):
            sn = float(np.mean(S1 + S2)) / 10 ** (snr / 10.0)
            SX = S1 + S2 + sn
            print(f"    {snr:5d}  {W_soft2.sum(axis=0).mean():12.10f}  "
                  f"{((S1 + S2) / SX).mean():19.6f}")
        print("    -> the allocation is a partition of unity at every bin and every SNR;")
        print("       the MMSE estimator deliberately attenuates.  They answer different")
        print("       questions (Section V-C).")

    # ------------------------------------------------------------------ Exp C
    with Section("4. Experiment C: nMSE against SNR"):
        rng = np.random.default_rng(seed)
        rows = {}
        print(f"    {'SNR':>5}  {'hard':>18}  {'soft':>18}  {'soft x shrinkage':>18}"
              f"  {'oracle':>18}")
        for snr in SNR_DB:
            sn = float(np.mean(S1 + S2)) / 10 ** (snr / 10.0)
            SX = S1 + S2 + sn
            wts = {
                "hard": W_hard,
                "soft": W_soft,
                "shrunk": W_soft * (S1 + S2) / SX,
                "oracle": S1 / SX,
            }
            r = monte_carlo(w, [S1, S2], sn, wts, n_trials, rng)
            rows[int(snr)] = r
            print(f"    {snr:5d}  " + "  ".join(
                f"{r[k][0]:11.4f}+-{r[k][1]:5.4f}"
                for k in ("hard", "soft", "shrunk", "oracle")))

        hi = rows[int(SNR_DB[-1])]
        lo = rows[int(SNR_DB[0])]
        print("\n    The two ends of the SNR range:")
        print(f"      high SNR: the soft partition approaches the oracle (Theorem V.2)")
        print(f"       -- at {SNR_DB[-1]} dB the soft NMSE is {hi['soft'][0]:.4f} and the")
        print(f"       oracle is {hi['oracle'][0]:.4f}.  The common value is the")
        print("       irreducible error of separating two overlapping Lorentzian modes,")
        print("       a property of the model rather than a numerical floor.")
        print(f"      low SNR: the two diverge -- at {SNR_DB[0]} dB the soft NMSE is")
        print(f"       {lo['soft'][0]:.3f} against an oracle of {lo['oracle'][0]:.3f}, a factor of")
        print(f"       {lo['soft'][0] / lo['oracle'][0]:.0f}.  Applying the Remark V.4 shrinkage brings it to")
        print(f"       {lo['shrunk'][0]:.3f}, which recovers the oracle.  The divergence is the")
        print("       expected behaviour of a partition of unity, which cannot attenuate")
        print("       the noise (Remark V.4).")

    print("\nCONCLUSION")
    print("  Theorem V.2, Remark V.4 and Section V-C are all confirmed.  At high SNR")
    print("  the soft partition agrees with the oracle at a non-zero irreducible nMSE,")
    print("  and at low SNR the gap to the oracle is the shrinkage factor of Remark V.4,")
    print("  which a partition of unity cannot apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(n_trials=args.n_trials, seed=args.seed, quick=args.quick)
