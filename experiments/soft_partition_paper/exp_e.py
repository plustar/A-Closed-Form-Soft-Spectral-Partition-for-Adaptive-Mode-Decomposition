# experiments/soft_partition_paper/exp_e.py
"""
fig_alpha_estimation  —  Corollary V.5, the FWHM estimator for alpha

WHAT THE COROLLARY CLAIMS
-------------------------
For a Lorentzian mode, FWHM = 2*beta, so alpha_hat_k = 2 / FWHM_k^2 with FWHM_k
measured as the half-power bin count inside the Voronoi band of mode k, times the bin
width.

WHAT CONTROLS ITS ACCURACY
--------------------------
FWHM is measured as an INTEGER bin count m, so alpha_hat = 2/(m*dw)^2.  With
n = 2*beta/dw the (non-integer) number of bins across the true FWHM, the count satisfies
|m - n| < 1 (up to an O(1/n^2) effect of sampling the peak off-grid), hence

    worst case  |alpha_hat/alpha - 1| <= (1 - 1/n)^-2 - 1  ~= 2/n ,
    typical     |alpha_hat/alpha - 1| ~= 1/n  (half-bin rounding).

The worst case depends on where the centre falls relative to the grid, so panel (a)
reports both the error at one centre position and the worst case over 20 sub-bin
offsets of the centre.  The 10% level is guaranteed for n >= 22.

The accuracy is therefore set by the number of bins across the FWHM, not by alpha as
such: a narrow mode is resolution limited however exact the spectrum is.  Plotting
against alpha hides this, because the same alpha corresponds to different bin counts
at different signal lengths and sampling rates.

Panel (a) therefore uses bins per FWHM as the abscissa, with both bounds overlaid.
Panel (b) adds the precondition the corollary needs in practice: a single periodogram
is far too rough for a half-power crossing, and the estimator is restored by spectral
averaging -- which the STFT pipeline already provides.

Baselines: the power-weighted second moment (which diverges for a Lorentzian, whose
second moment is infinite) and the Type-II MLE of Eq. (27).  The MLE is specified in
full: centre, mode power and noise floor known; alpha alone optimised; positivity
enforced by optimising log alpha; bounded Brent with xatol = 1e-4; no random
initialisation.  Eq. (27) is written for a common alpha, unlike the per-mode Eq. (26).
"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where softdamd lives
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.plot_utils import setup_font_config
from soft_partition_paper.figures import save_figure

FS       = 1000.0
N_FFT    = 256
W0       = 0.30           # mode centre, normalised frequency (f/fs)
SIGMA2   = 1.0
ALPHAS   = np.logspace(1.2, 5.5, 40)
SNR_DB   = 20.0
L_AVG    = (1, 4, 16, 64)
N_TRIALS = 60
SEED     = 0


def lorentz(w, w0, alpha, sigma2=SIGMA2):
    return sigma2 / (1.0 + 2.0 * alpha * (w - w0) ** 2)


def fwhm_estimate(w, P, b0=0, b1=None):
    """Corollary V.5.  Returns (alpha_hat, bins across the FWHM)."""
    b1 = w.size if b1 is None else b1
    band = P[b0:b1]
    if band.size == 0 or band.max() <= 0:
        return np.inf, 0
    n = int((band >= band.max() / 2.0).sum())
    if n == 0:
        return np.inf, 0
    return 2.0 / (n * (w[1] - w[0])) ** 2, n


def moment_estimate(w, P):
    p = P / max(P.sum(), 1e-30)
    mu = float((w * p).sum())
    var = float((((w - mu) ** 2) * p).sum())
    return 1.0 / (2.0 * var) if var > 0 else np.inf


def mle_estimate(w, P, w0, sigma_n2=1e-6):
    def nll(log_a):
        S = lorentz(w, w0, float(np.exp(log_a))) + sigma_n2
        return float(np.sum(np.log(S) + P / S))
    r = minimize_scalar(nll, bounds=(np.log(1.0), np.log(1e7)),
                        method='bounded', options={'xatol': 1e-4})
    return float(np.exp(r.x))


def rel_err(a_hat, a_true):
    return abs(a_hat - a_true) / a_true


def main(save_dir='outputs/soft_partition'):
    setup_font_config(fontsize=9)
    matplotlib.rcParams.update({'font.size': 9})

    F = N_FFT // 2 + 1
    w = np.linspace(0.0, 0.5, F)
    dw = w[1] - w[0]

    # ── panel (a): noiseless, against bins per FWHM ─────────────────────────
    # n is the TRUE number of bins across the FWHM, 2*beta/dw (not the measured count).
    offsets = np.arange(20) / 20.0                      # sub-bin centre positions
    k0 = int(np.floor(W0 / dw))
    bins, e_fwhm, e_worst, e_mom, e_mle, alph = [], [], [], [], [], []
    for a in ALPHAS:
        P = lorentz(w, W0, a)
        a_f, m = fwhm_estimate(w, P)
        if m == 0:
            continue
        bins.append(2.0 / np.sqrt(2.0 * a) / dw)
        alph.append(a)
        e_fwhm.append(rel_err(a_f, a))
        e_worst.append(max(rel_err(fwhm_estimate(w, lorentz(w, (k0 + o) * dw, a))[0], a)
                           for o in offsets))
        e_mom.append(rel_err(moment_estimate(w, P), a))
        e_mle.append(rel_err(mle_estimate(w, P, W0), a))
    bins = np.array(bins, dtype=float)
    e_fwhm, e_worst, e_mom, e_mle = map(np.array, (e_fwhm, e_worst, e_mom, e_mle))
    alph = np.array(alph)

    # ── panel (b): with noise, against the number of averaged periodograms ──
    rng = np.random.default_rng(SEED)
    alphas_b = [200.0, 1000.0, 5000.0]
    curves = {}
    for a in alphas_b:
        S = lorentz(w, W0, a)
        sn = float(S.mean()) / 10 ** (SNR_DB / 10.0)
        med = []
        for L in L_AVG:
            errs = [rel_err(fwhm_estimate(w, (S + sn)
                                          * rng.exponential(1.0, (L, F)).mean(axis=0))[0], a)
                    for _ in range(N_TRIALS)]
            med.append(float(np.median(errs)))
        curves[a] = np.array(med)

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    order = np.argsort(bins)
    ax.loglog(bins[order], 100 * e_fwhm[order], '-o', ms=3.2, lw=1.4,
              color='#4363d8', label=fr'FWHM, Cor. V.5 (centre at $\omega={W0}$)')
    ax.loglog(bins[order], 100 * e_worst[order], '-', lw=1.2, color='#911eb4',
              label='FWHM, worst of 20 sub-bin centre offsets')
    ax.loglog(bins[order], 100 * e_mom[order], '-s', ms=3.2, lw=1.4,
              color='#e6194b', alpha=0.85, label='power-weighted second moment')
    ax.loglog(bins[order], 100 * e_mle[order], '-^', ms=3.2, lw=1.4,
              color='#3cb44b', alpha=0.85, label='Type-II MLE, Eq. (27)')
    nn = np.linspace(max(bins.min(), 1.5), bins.max(), 200)
    ax.loglog(nn, 100 * ((1 - 1 / nn) ** -2 - 1), 'k-', lw=0.9, alpha=0.8,
              label=r'worst-case bound $(1-1/n)^{-2}-1$')
    ax.loglog(nn, 100 / nn, 'k--', lw=0.9, alpha=0.6, label=r'typical $1/n$')
    ax.axhline(10, color='0.55', lw=0.8, ls=':', label='10% threshold')
    ax.set_xlabel(r'bins across the FWHM,  $n = 2\beta/\Delta\omega$')
    ax.set_ylabel(r'relative error in $\alpha$ (%)')
    ax.legend(fontsize=5.4, loc='lower left', framealpha=0.92)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(r'(a) noiseless: accuracy is set by bins per FWHM, not by $\alpha$',
                 fontsize=8)

    ax2 = axes[1]
    cols = {200.0: '#e6194b', 1000.0: '#f58231', 5000.0: '#4363d8'}
    for a, med in curves.items():
        n_b = 2.0 / np.sqrt(2.0 * a) / dw               # true bins across the FWHM
        ax2.loglog(L_AVG, 100 * med, '-o', ms=3.6, lw=1.4, color=cols[a],
                   label=fr'$\alpha={a:.0f}$  ($n={n_b:.1f}$)')
    ax2.axhline(10, color='0.55', lw=0.8, ls=':')
    ax2.set_xlabel(r'averaged periodograms $L$')
    ax2.set_ylabel(r'median relative error in $\alpha$ (%)')
    ax2.set_xticks(list(L_AVG))
    ax2.set_xticklabels([str(v) for v in L_AVG])
    ax2.legend(fontsize=6.4, loc='upper right', framealpha=0.92)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_title(fr'(b) a smoothed spectrum is a precondition (SNR $=$ {SNR_DB:.0f} dB)',
                  fontsize=8)

    plt.tight_layout()
    out = save_figure(fig, 'alpha_estimation', save_dir)

    bad = bins[order][100 * e_worst[order] > 10]
    print(f'\n  bin width dw = {dw:.6f} normalised = {dw * FS:.3f} Hz at fs = {FS:g}')
    print(f'  single centre position: largest error with n >= 10 is '
          f'{100 * e_fwhm[bins >= 10].max():.1f}%')
    print(f'  worst over sub-bin offsets: largest n with error > 10% is '
          f'{bad.max() if bad.size else float("nan"):.1f}; '
          f'largest error with n >= 10 is {100 * e_worst[bins >= 10].max():.1f}%')
    print(f"\n  {'alpha':>10}  {'n':>4}  {'FWHM err':>9}  {'1/n':>7}  "
          f"{'moment':>9}  {'MLE':>9}")
    for a, n, ef, em, el in zip(alph[order], bins[order], e_fwhm[order],
                                e_mom[order], e_mle[order]):
        print(f'  {a:10.1f}  {n:6.2f}  {ef:9.4f}  {1/n:7.4f}  {em:9.4f}  {el:9.4f}')
    print(f"\n  noise, SNR = {SNR_DB:.0f} dB, median over {N_TRIALS} trials")
    print(f"  {'alpha':>10}  " + '  '.join(f'{"L=" + str(L):>9}' for L in L_AVG))
    for a, med in curves.items():
        print(f'  {a:10.1f}  ' + '  '.join(f'{v:9.4f}' for v in med))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    args = ap.parse_args()
    main(args.save_dir)
