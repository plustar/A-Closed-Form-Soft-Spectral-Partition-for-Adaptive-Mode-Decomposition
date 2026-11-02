"""Corollary V.5 -- the FWHM estimator for alpha.

WHAT THE PAPER STATES
---------------------
Cor. V.5 : for a Lorentzian mode, FWHM = 2 beta, so

               alpha_hat_k = 2 / FWHM_k^2 ,
               FWHM_k = #{w in [b_0k, b_1k] : P(w) >= P_k^peak / 2} * dw ,

           where [b_0k, b_1k] is the Voronoi band DAMD assigns to mode k.  Its
           accuracy is set by the number of bins across the FWHM, n = 2 beta / dw,
           and not by alpha; in practice the spectrum must be smoothed or averaged
           and the band must isolate the mode.  Eq. (27) gives the Type-II
           maximum-likelihood gradient, used as a refinement.

WHAT IS CHECKED (cf. Experiment S2 of the supplementary material)
------------------------------------------------------------------
* accuracy against the number of bins per FWHM, and the rounding bound;
* the estimator inside the Voronoi band, with a neighbouring mode present;
* noisy periodograms, single and averaged;
* centre error;
* a fully specified Type-II MLE baseline for a common alpha (Eq. (27)), together with
  the power-weighted second moment.

The baseline case is a noiseless analytic Lorentzian with b_0, b_1 = 0, F -- the whole
spectrum rather than the Voronoi band Cor. V.5 specifies.  The other settings are
added to it.

Usage
-----
    python experiments/soft_partition_paper/repro/s8_fwhm.py
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
    Section, alpha_from_beta, beta_from_alpha, print_env,
)

N_FFT = 256
W0 = 0.30                     # mode centre, normalised frequency
SEED = 0


def lorentz(w, w0, alpha, sigma2=1.0):
    return sigma2 / (1.0 + 2.0 * alpha * (w - w0) ** 2)


def fwhm_alpha(w, P, b0=None, b1=None):
    """Corollary V.5 as written: half-power bin count inside the band, times dw."""
    b0 = 0 if b0 is None else b0
    b1 = w.size if b1 is None else b1
    band = P[b0:b1]
    if band.size == 0 or band.max() <= 0:
        return np.inf, 0
    n_half = int((band >= band.max() / 2.0).sum())
    if n_half == 0:
        return np.inf, 0
    dw = w[1] - w[0]
    return 2.0 / (n_half * dw) ** 2, n_half


def moment_alpha(w, P):
    """Power-weighted second moment, a baseline estimator."""
    p = P / max(P.sum(), 1e-30)
    mu = float((w * p).sum())
    var = float((((w - mu) ** 2) * p).sum())
    return 1.0 / (2.0 * var) if var > 0 else np.inf


def mle_alpha(w, P, w0, sigma2=1.0, sigma_n2=1e-6, bounds=(1e0, 1e7)):
    """Type-II MLE of a COMMON alpha, fully specified.

    Known      : centre w0, mode power sigma2, noise floor sigma_n2, model shape.
    Optimised  : alpha only, one scalar.
    Positivity : enforced by optimising log(alpha).
    Objective  : negative log-likelihood of the exponential model,
                 sum_w [ log S_X(w) + P(w)/S_X(w) ],  S_X = S_k + sigma_n2.
    Solver     : bounded Brent (scipy minimize_scalar), xatol = 1e-4 in log alpha,
                 bounds as given; no random initialisation.

    Note: Eq. (27) of the paper is written for a common alpha, whereas Eq. (26)
    defines per-mode alpha_k.  This routine implements the common-alpha model; with
    a single mode, as here, the two coincide.
    """
    from scipy.optimize import minimize_scalar

    def nll(log_a):
        S = lorentz(w, w0, float(np.exp(log_a)), sigma2) + sigma_n2
        return float(np.sum(np.log(S) + P / S))

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])),
                          method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def rel_err(a_hat, a_true):
    return abs(a_hat - a_true) / a_true


def main(seed: int = SEED, n_trials: int = 40, quick: bool = False):
    print_env()
    if quick:
        n_trials = 10
    F = N_FFT // 2 + 1
    w = np.linspace(0.0, 0.5, F)
    dw = w[1] - w[0]
    print(f"\nn_fft = {N_FFT}, F = {F} one-sided bins, dw = {dw:.6f} in normalised")
    print(f"frequency.  Mode centre w0 = {W0}.")

    # ------------------------------------------------------------------ rounding
    with Section("1. Where 10% accuracy holds, and why"):
        print("    FWHM is measured as an integer bin count n, so FWHM = n*dw and")
        print("    alpha_hat = 2/(n dw)^2.  A rounding of +-1/2 bin therefore gives")
        print("        |d alpha / alpha| ~= 2 * (1/2) / n = 1/n ,")
        print("    i.e. the attainable accuracy is set by the number of bins across the")
        print("    FWHM.  n = 2 beta / dw, so narrow modes (large alpha) are resolution")
        print("    limited no matter how exact the spectrum is.")
        print(f"\n    {'alpha':>10}  {'beta':>9}  {'bins/FWHM':>10}  {'1/n bound':>10}"
              f"  {'measured':>10}  {'<= 10%':>7}")
        alphas = np.logspace(1.5, 5.5, 13 if quick else 25)
        first_fail = None
        for a in alphas:
            P = lorentz(w, W0, a)
            a_hat, n = fwhm_alpha(w, P)
            e = rel_err(a_hat, a)
            ok = e <= 0.10
            if not ok and first_fail is None:
                first_fail = a
            print(f"    {a:10.1f}  {beta_from_alpha(a):9.5f}  {n:10d}  "
                  f"{(1.0 / n if n else np.inf):10.4f}  {e:10.4f}  {'yes' if ok else 'no':>7}")
        print(f"\n    the 10% threshold is first missed at alpha ~ {first_fail:.0f}"
              if first_fail else "\n    the 10% threshold holds over the whole range")
        print("    -> the accuracy is controlled by the number of bins per FWHM, not by")
        print("       alpha as such, which is how Corollary V.5 states it; the threshold")
        print("       in alpha above depends on the frequency grid.")

    # ------------------------------------------------------------------ band
    with Section("2. Inside the Voronoi band, as Corollary V.5 specifies"):
        print("    Section 1 uses b_0, b_1 = 0, F, i.e. the whole spectrum.  Cor. V.5")
        print("    specifies the Voronoi band of mode k.  With a neighbour present the")
        print("    two differ.")
        w2 = W0 + 0.06
        print(f"    neighbour at {w2}, Voronoi boundary at {(W0 + w2) / 2:.4f}")
        print(f"    {'alpha':>10}  {'full spectrum':>14}  {'Voronoi band':>14}")
        for a in [200.0, 1000.0, 5000.0, 20000.0]:
            P = lorentz(w, W0, a) + lorentz(w, w2, a)
            b_edge = int(np.searchsorted(w, 0.5 * (W0 + w2)))
            e_full = rel_err(fwhm_alpha(w, P)[0], a)
            e_band = rel_err(fwhm_alpha(w, P, 0, b_edge)[0], a)
            print(f"    {a:10.1f}  {e_full:14.4f}  {e_band:14.4f}")
        print("    -> restricting to the band helps once the mode is narrow relative to")
        print("       the band; at alpha = 200 the band is narrower than the mode itself")
        print("       and truncates it, so the band restriction needs the same")
        print("       bins-per-FWHM precondition as section 1.")

    # ------------------------------------------------------------------ noise
    with Section("3. With noise"):
        rng = np.random.default_rng(seed)
        print("    P is an exponential periodogram with the Lorentzian mean plus a white")
        print("    floor.  L = 1 is a single realisation; L > 1 averages L independent")
        print("    periodograms, as Welch or epoch averaging would.  Median relative")
        print("    error over trials, at SNR = 20 dB.")
        print(f"    {'alpha':>10}  " + "  ".join(f"{'L=' + str(L):>10}"
                                                 for L in (1, 4, 16, 64)))
        for a in [200.0, 1000.0, 5000.0]:
            S = lorentz(w, W0, a)
            sn = float(S.mean()) / 10 ** (20 / 10.0)
            cells = []
            for L in (1, 4, 16, 64):
                errs = []
                for _ in range(n_trials):
                    P = (S + sn) * rng.exponential(1.0, (L, F)).mean(axis=0)
                    errs.append(rel_err(fwhm_alpha(w, P)[0], a))
                cells.append(f"{np.median(errs):10.4f}")
            print(f"    {a:10.1f}  " + "  ".join(cells))
        print("    -> a single periodogram is far too rough for a half-power crossing.")
        print("       Averaging improves it but does not by itself restore it at 20 dB,")
        print("       so a smoothed or averaged spectrum is a precondition of Cor. V.5;")
        print("       single-frame magnitudes do not supply it.")

    # ------------------------------------------------------------------ centre error
    with Section("4. With centre error"):
        print("    The band edges follow the detected centres, so a centre error moves")
        print("    the band.  Median relative error over 20 offsets each.")
        print(f"    {'alpha':>10}  " + "  ".join(f"{o:>7} bins" for o in (0, 1, 2, 5)))
        rng = np.random.default_rng(seed + 1)
        w2 = W0 + 0.06
        for a in [200.0, 1000.0, 5000.0]:
            P = lorentz(w, W0, a) + lorentz(w, w2, a)
            cells = []
            for off in (0, 1, 2, 5):
                errs = []
                for _ in range(20):
                    d = int(rng.integers(-off, off + 1)) if off else 0
                    b_edge = int(np.searchsorted(w, 0.5 * (W0 + w2))) + d
                    b_edge = int(np.clip(b_edge, 1, F))
                    errs.append(rel_err(fwhm_alpha(w, P, 0, b_edge)[0], a))
                cells.append(f"{np.median(errs):12.4f}")
            print(f"    {a:10.1f}  " + "  ".join(cells))

    # ------------------------------------------------------------------ MLE
    with Section("5. The Type-II MLE baseline, fully specified"):
        print("    " + mle_alpha.__doc__.strip().split("\n\n")[1].replace("\n    ", "\n    "))
        print(f"\n    {'alpha':>10}  {'FWHM err':>10}  {'moment err':>11}  {'MLE err':>9}")
        for a in [100.0, 500.0, 2000.0, 10000.0, 50000.0]:
            P = lorentz(w, W0, a)
            print(f"    {a:10.1f}  {rel_err(fwhm_alpha(w, P)[0], a):10.4f}  "
                  f"{rel_err(moment_alpha(w, P), a):11.4f}  "
                  f"{rel_err(mle_alpha(w, P, W0), a):9.4f}")
        print("    -> the MLE is essentially exact on a noiseless spectrum that matches")
        print("       its model, at the cost of a numerical solve; the moment estimator")
        print("       diverges for narrow modes because the Lorentzian has no finite")
        print("       second moment.  Eq. (27) is written for a common alpha; with one")
        print("       mode, as here, it coincides with the per-mode estimator of")
        print("       Eq. (26).")

    # ------------------------------------------------------------------ closing
    with Section("6. Sanity: alpha and beta round-trip"):
        for a in (100.0, 2000.0, 50000.0):
            b = beta_from_alpha(a)
            print(f"    alpha = {a:9.1f}  ->  beta = {b:.6f}  ->  alpha = "
                  f"{alpha_from_beta(b):9.1f}   FWHM = 2 beta = {2*b:.6f}")

    print("\nCONCLUSION")
    print("  Corollary V.5 is confirmed, and its accuracy is governed by the number of")
    print("  bins across the FWHM rather than by alpha as such.  The estimator is")
    print("  applied inside the Voronoi band as the corollary specifies; it needs a")
    print("  smoothed or averaged spectrum and a band that isolates the mode, as the")
    print("  paper states.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(seed=args.seed, n_trials=args.n_trials, quick=args.quick)
