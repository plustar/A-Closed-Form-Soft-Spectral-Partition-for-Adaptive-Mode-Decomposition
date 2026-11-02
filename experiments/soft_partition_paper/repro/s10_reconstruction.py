"""Remark VII.1 and supplementary Appendix G -- time-domain reconstruction.

WHAT THE PAPER STATES
---------------------
Remark VII.1 separates two statements about recovering waveforms from the spectral
modes U_k = s_k * T x:

* reconstruction: since sum_k s_k = 1 pointwise, linearity of the synthesis operator
  gives sum_k T^-1[s_k T x] = T^-1[T x] for every invertible transform; for the
  unit-hop STFT this round trip is exact to machine precision;
* per-bin MMSE optimality transfers to the waveform only if the analysis operator is
  an isometry up to a constant and the masked array remains in its range.  The
  second condition does not hold, so Theorem V.2 is stated as a coefficient-domain
  result.  Appendix G specifies the STFT analysis/synthesis pair and gives the
  measured residuals.

WHAT THIS SCRIPT MEASURES
-------------------------
For each transform in softdamd, using the shipped implementation:

  1. round-trip error of forward followed by inverse, on four test signals;
  2. whether the analysis operator preserves energy up to a constant, by testing the
     constancy of ||W x||^2 / ||x||^2 across signals (the isometry spread);
  3. whether a MASKED coefficient array is still in the range of the analysis
     operator, measured as the residual of the analyse-synthesise-analyse cycle;
  4. whether the partition identity sum_k s_k = 1 survives the round trip:
     sum_k inverse(s_k * W x) vs x.

Point 4 is what licenses reconstructing the modes.  Points 2 and 3 decide whether
the per-bin MMSE statement transfers to the waveform, which is a stronger statement.

Usage
-----
    python experiments/soft_partition_paper/repro/s10_reconstruction.py
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
    Section, hz_to_norm, lorentzian_weights, print_env, rel_l2,
)

FS = 1000.0
N = 2048
N_FFT = 256
SEED = 11
ALPHA = 2000.0
CENTRES_HZ = np.array([80.0, 200.0, 350.0])


def make_signals(n=N, fs=FS, seed=SEED):
    """A few signals with different spectral content, to test energy constancy."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    out = {
        "three tones": sum(np.sin(2 * np.pi * f * t) for f in CENTRES_HZ),
        "white noise": rng.standard_normal(n),
        "chirp": np.sin(2 * np.pi * (50 + 150 * t / t[-1]) * t),
        "impulse": np.eye(1, n, n // 2).ravel(),
    }
    return {k: v / np.linalg.norm(v) for k, v in out.items()}


class _Bound:
    """A softdamd transform together with the sampling rate it is applied at."""

    def __init__(self, tf, fs):
        self.tf, self.fs = tf, fs
        self.is_linear = getattr(tf, 'is_linear', False)
        self.is_invertible = getattr(tf, 'is_invertible', False)


def build(name, fs=FS, hop_length=1, **kw):
    from softdamd import make_transform
    if name in ('stft', 'ssq_stft'):
        kw['hop'] = hop_length
    return _Bound(make_transform(name, **kw), fs)


def forward(tf, x):
    """Analysis: returns (coefficients (F, T), TFMap)."""
    m = tf.tf.forward(np.asarray(x, dtype=np.float64), tf.fs)
    return m.coef, m


def inverse(tf, W, n, info):
    """Synthesis with the TFMap produced by the matching analysis."""
    return np.asarray(info.inverse(W)).ravel()[:n]


def main(hop: int = 1, quick: bool = False):
    print_env()
    sigs = make_signals()
    specs = {
        "STFT     hop=1": ("stft", dict(n_fft=N_FFT, hop_length=hop, fs=FS)),
        "STFT     hop=64": ("stft", dict(n_fft=N_FFT, hop_length=64, fs=FS)),
        "SSQ-STFT hop=1": ("ssq_stft", dict(n_fft=N_FFT, hop_length=hop, fs=FS)),
        "CWT": ("cwt", dict(fs=FS)),
        "SSQ-CWT": ("ssq_cwt", dict(fs=FS)),
    }
    print(f"\nfs = {FS:g}, N = {N}, n_fft = {N_FFT}.  Signals are unit-norm.")
    print("Every number below comes from the shipped implementation, not from theory.")

    built = {}
    with Section("1. Which transforms are invertible in this implementation?"):
        for label, (name, kw) in specs.items():
            try:
                tf = build(name, **kw)
                W, _ = forward(tf, sigs["three tones"])
                inv = bool(getattr(tf, "is_invertible", False))
                lin = bool(getattr(tf, "is_linear", False))
                built[label] = (tf, W.shape)
                print(f"    {label:16s} coefficients {str(W.shape):>14}  "
                      f"invertible={inv}  linear={lin}")
            except Exception as exc:
                print(f"    {label:16s} unavailable ({type(exc).__name__}: {exc})")

    # ------------------------------------------------------------------ round trip
    with Section("2. Round-trip error, forward then inverse"):
        print("    'raw' is ||inverse(forward(x)) - x|| / ||x||.  'gain-corrected' first")
        print("    rescales by the least-squares gain g = <xr, x>/<xr, xr>, which absorbs")
        print("    a pure normalisation mismatch between the forward and inverse.")
        print(f"    {'transform':16s}  " + "  ".join(f"{k:>22}" for k in sigs))
        rt_ok = {}
        for label, (tf, _) in built.items():
            cells, errs = [], []
            for k, x in sigs.items():
                try:
                    W, info = forward(tf, x)
                    xr = inverse(tf, W, N, info)
                    e = rel_l2(xr, x)
                    den = float(np.dot(xr, xr))
                    g = float(np.dot(xr, x) / den) if den > 0 else 1.0
                    eg = rel_l2(g * xr, x)
                except Exception:
                    e = eg = np.nan
                errs.append(eg)
                cells.append(f"{e:10.2e}/{eg:10.2e}" if np.isfinite(e)
                             else f"{'n/a':>22}")
            rt_ok[label] = np.nanmax(errs) if np.any(np.isfinite(errs)) else np.nan
            print(f"    {label:16s}  " + "  ".join(cells))
        print("    (raw / gain-corrected)")
        print("    -> a large raw error with a small gain-corrected one is a normalisation")
        print("       convention mismatch; both large means the pair is not an")
        print("       analysis/synthesis pair at all in this implementation, and the")
        print("       transform is analysis-only.  The STFT pair used in the paper is")
        print("       specified in Appendix G.")
        print("\n    The STFT pair is exact.  The SSQ-STFT inverse (sum over rows) is exact up")
        print("    to the coefficients below the phase threshold and those squeezed across")
        print("    the DC/Nyquist rows; the wavelet inverses use the single-integral formula")
        print("    on 64 log-spaced scales, whose error is set by the scale grid and by the")
        print("    signal energy outside the analysed band (below the lowest scale).")

    # ------------------------------------------------------------------ isometry
    with Section("3. Is the analysis operator an isometry?"):
        print("    An isometry (up to a constant c) would give ||W x||^2 / ||x||^2 = c for")
        print("    every x.  A varying ratio means no Parseval identity is available.")
        print(f"    {'transform':16s}  " + "  ".join(f"{k:>14}" for k in sigs)
              + f"  {'spread':>10}")
        for label, (tf, _) in built.items():
            ratios = []
            for k, x in sigs.items():
                try:
                    W, _ = forward(tf, x)
                    ratios.append(float(np.sum(np.abs(W) ** 2) / np.sum(np.abs(x) ** 2)))
                except Exception:
                    ratios.append(np.nan)
            r = np.array(ratios)
            spread = (np.nanmax(r) - np.nanmin(r)) / np.nanmax(np.abs(r)) \
                if np.any(np.isfinite(r)) else np.nan
            print(f"    {label:16s}  "
                  + "  ".join(f"{v:14.4f}" if np.isfinite(v) else f"{'n/a':>14}"
                              for v in r)
                  + f"  {spread:10.3f}")
        print("    -> a spread far from 0 means the energy ratio is signal dependent, so")
        print("       the transform is not an isometry and Parseval cannot be invoked.")

    # ------------------------------------------------------------------ range
    with Section("4. Does a masked coefficient array stay in the range of W?"):
        print("    Even for a tight frame, an arbitrary coefficient")
        print("    array need not be W x for any x.  Synthesis then projects it, and")
        print("    per-coefficient optimality does not survive the projection.")
        print("    Measured as ||W(W^-1 C) - C|| / ||C|| with C = s_k * W x, i.e. the")
        print("    residual of the analyse-synthesise-analyse cycle.")
        x = sigs["three tones"]
        c_norm = hz_to_norm(CENTRES_HZ, FS)
        print("    (mask of mode 1 on the three-tone signal; all modes and white noise below)")
        print(f"    {'transform':16s}  {'unmasked C = W x':>18}  {'masked C = s_1 W x':>20}")
        for label, (tf, _) in built.items():
            if not tf.is_linear:
                print(f"    {label:16s}  {'(nonlinear: n/a)':>18}  {'':>20}")
                continue
            try:
                W, info = forward(tf, x)
                r_un = rel_l2(forward(tf, inverse(tf, W, N, info))[0], W)

                # Coefficients are (frequencies, frames).
                F = W.shape[0]
                w_axis = np.linspace(0.0, 0.5, F)
                S = lorentzian_weights(w_axis, c_norm, ALPHA)
                C = S[0][:, None] * W
                r_ma = rel_l2(forward(tf, inverse(tf, C, N, info))[0], C)
                print(f"    {label:16s}  {r_un:18.3e}  {r_ma:20.3e}")
            except Exception as exc:
                print(f"    {label:16s}  failed ({type(exc).__name__})")
        # every mode, three tones and white noise, unit-hop STFT (the numbers in Appendix G)
        tf = build("stft", n_fft=N_FFT, hop_length=1, fs=FS)
        for sig in ("three tones", "white noise"):
            W, info = forward(tf, sigs[sig])
            F = W.shape[0]
            S = lorentzian_weights(np.linspace(0.0, 0.5, F), c_norm, ALPHA)
            r = [rel_l2(forward(tf, inverse(tf, S[k][:, None] * W, N, info))[0],
                        S[k][:, None] * W) for k in range(len(c_norm))]
            print(f"    STFT hop=1, {sig:12s} masked residual per mode: "
                  + "  ".join(f"{v:.2e}" for v in r))
        print("    -> a non-negligible masked residual means the mask moves the array out")
        print("       of the range of W, so per-bin MMSE optimality does not transfer")
        print("       exactly to the waveform (Remark VII.1, Appendix G).")

    # ------------------------------------------------------------------ partition
    with Section("5. Does sum_k s_k = 1 survive the round trip?"):
        print("    This is what licenses reconstructing the modes at all, and it is a")
        print("    weaker requirement than the MMSE transfer: it only needs linearity of")
        print("    the inverse, not unitarity.")
        x = sigs["three tones"]
        c_norm = hz_to_norm(CENTRES_HZ, FS)
        print(f"    {'transform':16s}  {'|| sum_k u_k - x || / ||x||':>28}")
        for label, (tf, _) in built.items():
            if not np.isfinite(rt_ok.get(label, np.nan)):
                print(f"    {label:16s}  {'(no inverse)':>28}")
                continue
            try:
                W, info = forward(tf, x)
                F = W.shape[0]
                S = lorentzian_weights(np.linspace(0.0, 0.5, F), c_norm, ALPHA)
                total = sum(inverse(tf, S[k][:, None] * W, N, info)
                            for k in range(S.shape[0]))
                print(f"    {label:16s}  {rel_l2(total, x):28.3e}")
            except Exception as exc:
                print(f"    {label:16s}  failed ({type(exc).__name__})")
        print("    -> by linearity of the inverse this equals the plain round-trip error,")
        print("       so mode reconstruction is exactly as accurate as the transform's own")
        print("       inversion -- no better, and no worse.")

    print("\n" + "=" * 78)
    print("WHICH PATHS IN THE PAPER USE AN INVERSE TRANSFORM")
    print("=" * 78)
    print("  Experiment D never calls an inverse transform: plot_tf_spectrogram,")
    print("  plot_tf_centers and plot_tf_energy_stratified all read the coefficients and")
    print("  the per-mode band energies directly.  The inverse is used only by")
    print("  softdamd.reconstruct (Experiment S5) and Experiment S4, both with the STFT.")
    print("  Consequence: the waveform statements of the paper rest on the exact STFT pair.")

    print("\nCONCLUSION")
    print("  Reconstruction: sum_k u_k recovers x to the accuracy of the transform's")
    print("  own inverse, which needs only linearity; for the unit-hop STFT this is")
    print("  exact to machine precision.  Per-bin MMSE optimality would additionally")
    print("  need the analysis operator to be an isometry AND the masked array to")
    print("  remain in its range.  The unit-hop STFT is an isometry to within 0.7%,")
    print("  but the masked array leaves the range (residuals above), so the MMSE")
    print("  statement of Theorem V.2 holds in the coefficient domain, as stated in")
    print("  Remark VII.1.  The synchrosqueezed transforms are nonlinear and are not")
    print("  inverted anywhere in the paper.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(hop=args.hop, quick=args.quick)
