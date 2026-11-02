# experiments/soft_partition_paper/exp_d.py
"""
fig_mmse_shrinkage  —  Theorem V.2 and Remark V.4

Direct frequency-domain verification of the Bayesian result and of the finite-noise
correction that accompanies it.

Signal model (Definition 2 of the paper):
  X(omega) = U1_true(omega) + U2_true(omega) + N(omega)
  U_k ~ CN(0, S_k(omega)),  S_k(omega) = sigma^2 / (1 + 2*alpha*(omega-omega_k)^2)
  N   ~ CN(0, sigma_N^2)

This is exactly the Lorentzian model under which Theorem V.2 proves that the
Lorentzian soft partition is the MMSE estimator.

Three extractors for mode 1:
  Oracle: Û_k = [S_k / (S1+S2+S_N)] * X           exact MMSE (Theorem V.2)
  Soft:   Û_k = [1+2α(ω-ω_k)^2]^{-1}/Σ  * X       Lorentzian soft partition
          (= Oracle when α = α_true, Theorem V.2)
  Hard:   Û_k = 1[ω ∈ B_k] * X                     Voronoi indicator

Metric: normalised MSE  nMSE_k = E[|Û_k - U_k^true|^2] / E[|U_k^true|^2]

Four extractors for mode 1:
  Oracle:  Û_k = [S_k / (S1+S2+S_N)] · X            exact MMSE (Thm. V.2)
  Soft:    Û_k = s_k*(ω) · X                        Eq. (12), a partition of unity
  Shrunk:  Û_k = s_k*(ω) · [ΣS_j/(ΣS_j+σ_N²)] · X   Eq. (12) with the Rem. V.4 factor (identical to the oracle)
  Hard:    Û_k = 1[ω ∈ B_k] · X                     Voronoi indicator

What the figure shows (Thm. V.2 and Rem. V.4):
  - Soft → Oracle in the LOW-NOISE limit, which is the regime Thm. V.2 covers.
  - At finite noise the two separate, because a partition of unity cannot attenuate:
    Σ_k s_k = 1 by construction, whereas the MMSE weights sum to less than one.
    The gap is exactly the shrinkage factor ΣS_j/(ΣS_j+σ_N²) of Rem. V.4, and the
    "Shrunk" estimator — Eq. (12) times that factor — equals the oracle identically (not plotted).
  - The common high-SNR value is the irreducible error of separating two overlapping
    Lorentzian modes; it is a property of the model, not a numerical floor.
  - Hard extraction sits above the soft curve at all SNR, by the strict inequality of
    Cor. III.2.

Layout: two panels
  (a) nMSE vs SNR for the four extractors
  (b) the shrinkage factor across frequency at three SNRs, showing that it tends to 1
      at the mode centres and to 0 in the gap between them (Rem. V.4)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where softdamd lives
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.plot_utils import setup_font_config
from soft_partition_paper.figures import save_figure
import os

# ── Signal parameters ──────────────────────────────────────────
F          = 512               # frequency bins (dense for clean curves)
OMEGA1     = 0.12              # mode 1 centre (normalised, ≈ 60 Hz at fs=500)
OMEGA2     = 0.30              # mode 2 centre (well-separated)
ALPHA_TRUE = 5000.0            # true bandwidth parameter β=1/√(2α)≈0.01
SIGMA2     = 1.0               # mode signal variance
SNR_DB     = np.arange(-10, 45, 5)   # -10 to 40 dB
N_TRIALS   = 2000              # Monte Carlo trials per SNR
SEED       = 42
SHRINK_SHOW = (0, 20, 40)     # SNRs whose shrinkage profile is drawn in panel (b)


def _lorentzian(freqs, omega_k, alpha, sigma2=1.0):
    """Lorentzian PSD: S_k(ω) = σ²/[1 + 2α(ω-ω_k)²]."""
    return sigma2 / (1.0 + 2.0 * alpha * (freqs - omega_k)**2)


def main(save_dir='outputs/soft_partition'):
    setup_font_config(fontsize=10)
    matplotlib.rcParams.update({'font.size': 10})

    rng    = np.random.default_rng(SEED)
    freqs  = np.linspace(0.0, 0.5, F)

    # ── True PSDs ─────────────────────────────────────────────────
    S1 = _lorentzian(freqs, OMEGA1, ALPHA_TRUE, SIGMA2)   # (F,)
    S2 = _lorentzian(freqs, OMEGA2, ALPHA_TRUE, SIGMA2)

    # Voronoi boundary for hard partition
    boundary = (np.abs(freqs - OMEGA1) < np.abs(freqs - OMEGA2))  # True for mode 1

    # Soft Lorentzian weights with true α  (= oracle only as σ_N² → 0, Thm. V.2)
    D1  = (freqs - OMEGA1)**2
    D2  = (freqs - OMEGA2)**2
    W1_soft = (1 + 2*ALPHA_TRUE*D1)**-1
    W2_soft = (1 + 2*ALPHA_TRUE*D2)**-1
    W1_soft = W1_soft / (W1_soft + W2_soft)   # normalised, K=2

    # Hard weights
    W1_hard = boundary.astype(float)

    # VMD fixed-point allocation (5) at the same centres: the inverse-square weights
    # d_1^-2 / (d_1^-2 + d_2^-2), which Experiment A shows is what the ADMM solver
    # converges to once reconstruction is enforced.  Included as a baseline; it
    # is evaluated on the same draws and does not alter the random sequence.
    W1_inv = D2 / np.maximum(D1 + D2, 1e-300)       # = D1^-1/(D1^-1+D2^-1), safe at D1 = 0

    # ── Monte Carlo over SNR ──────────────────────────────────────
    nmse_oracle = np.zeros(len(SNR_DB))
    nmse_soft   = np.zeros(len(SNR_DB))
    nmse_hard   = np.zeros(len(SNR_DB))
    nmse_shrunk = np.zeros(len(SNR_DB))
    nmse_inv    = np.zeros(len(SNR_DB))
    shrink_curves = {}          # SNR -> shrinkage factor across frequency
    rel_se = {'oracle': [], 'soft': [], 'hard': []}

    for si, snr_db in enumerate(SNR_DB):
        sigma_n2 = SIGMA2 / 10**(snr_db / 10.0)
        S_N      = sigma_n2 * np.ones(F)
        S_X      = S1 + S2 + S_N

        # Oracle weights: MMSE = S_k / S_X  (Thm. V.2)
        W1_oracle = S1 / S_X   # (F,)
        # Remark V.4: the finite-noise MMSE is the soft mask times this factor
        shrink   = (S1 + S2) / S_X
        W1_shrunk = W1_soft * shrink
        if int(snr_db) in SHRINK_SHOW:
            shrink_curves[int(snr_db)] = shrink

        # Generate complex Gaussian realisations
        std1   = np.sqrt(S1 / 2.0)
        std2   = np.sqrt(S2 / 2.0)
        std_n  = np.sqrt(S_N / 2.0)

        U1 = std1 * (rng.standard_normal((N_TRIALS, F))
                     + 1j * rng.standard_normal((N_TRIALS, F)))
        U2 = std2 * (rng.standard_normal((N_TRIALS, F))
                     + 1j * rng.standard_normal((N_TRIALS, F)))
        Nz = std_n * (rng.standard_normal((N_TRIALS, F))
                      + 1j * rng.standard_normal((N_TRIALS, F)))
        X  = U1 + U2 + Nz                              # (N_trials, F)

        # Extract mode 1
        U1_oracle = W1_oracle[None, :] * X
        U1_soft   = W1_soft[None,   :] * X
        U1_hard   = W1_hard[None,   :] * X
        U1_shrunk = W1_shrunk[None, :] * X

        # nMSE = E[|Û_1 - U1|²] / E[|U1|²]
        denom = float(np.mean(np.abs(U1)**2))
        # Monte-Carlo standard error of each nMSE, from the per-trial squared errors
        # (the denominator is estimated from 2000 x F draws and treated as exact).
        for key, Uh in (('oracle', U1_oracle), ('soft', U1_soft), ('hard', U1_hard)):
            per_trial = np.mean(np.abs(Uh - U1)**2, axis=1) / denom
            rel_se[key].append(float(per_trial.std(ddof=1) / np.sqrt(N_TRIALS)
                                     / max(per_trial.mean(), 1e-30)))
        nmse_oracle[si] = float(np.mean(np.abs(U1_oracle - U1)**2)) / denom
        nmse_soft[si]   = float(np.mean(np.abs(U1_soft   - U1)**2)) / denom
        nmse_hard[si]   = float(np.mean(np.abs(U1_hard   - U1)**2)) / denom
        nmse_shrunk[si] = float(np.mean(np.abs(U1_shrunk - U1)**2)) / denom
        nmse_inv[si]    = float(np.mean(np.abs(W1_inv[None, :] * X - U1)**2)) / denom

    print('  SNR(dB)   oracle     soft       VMD f.p.(5)  hard')
    for si, snr_db in enumerate(SNR_DB):
        print(f'  {int(snr_db):6d}  {nmse_oracle[si]:.4f}   {nmse_soft[si]:.4f}   '
              f'{nmse_inv[si]:.4f}      {nmse_hard[si]:.4f}')

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))

    # (a) nMSE vs SNR
    ax = axes[0]
    ax.semilogy(SNR_DB, nmse_hard, '-s', color='#e6194b', lw=1.5, ms=4,
                label=r'Hard (Voronoi)')
    ax.semilogy(SNR_DB, nmse_soft, '-o', color='#4363d8', lw=1.5, ms=4,
                label=r'Soft $s_k^*$, Eq. (12)')
    # The "soft x shrinkage" estimator is NOT plotted: (12) times (22) equals S_k/S_X
    # identically, so on the same draws it reproduces the oracle to floating-point
    # precision and carries no independent evidence.  Its agreement is asserted below.
    ax.semilogy(SNR_DB, nmse_oracle, '--^', color='#3cb44b', lw=1.2, ms=4,
                alpha=0.85, label=r'Oracle MMSE, Thm. V.2')

    # the two regimes, annotated on the curves themselves
    i_lo = 2
    ax.annotate('a partition of unity\ncannot attenuate',
                xy=(SNR_DB[i_lo], 0.5 * (nmse_soft[i_lo] + nmse_oracle[i_lo])),
                xytext=(SNR_DB[i_lo] + 6.0, nmse_soft[i_lo] * 1.4),
                fontsize=6.2, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.7, color='0.35'))
    i_hi = len(SNR_DB) - 1
    ax.annotate('irreducible\nmixture error',
                xy=(SNR_DB[i_hi], nmse_oracle[i_hi]),
                xytext=(SNR_DB[i_hi] - 17, nmse_oracle[i_hi] * 2.6),
                fontsize=6.2, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.7, color='0.35'))
    ax.set_ylim(top=ax.get_ylim()[1] * 3.0)

    ax.set_xlabel(r'peak SNR $\sigma^2/\sigma_N^2$ (dB)')
    ax.set_ylabel('Normalised MSE')
    ax.legend(fontsize=6.4, loc='upper right', framealpha=0.92)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(r'(a) soft $\to$ oracle as $\sigma_N^2\!\to\!0$ (Thm. V.2)', fontsize=8)

    # (b) the shrinkage factor across frequency
    ax2 = axes[1]
    colors = {0: '#e6194b', 20: '#f58231', 40: '#4363d8'}
    for snr in sorted(shrink_curves):
        ax2.plot(freqs, shrink_curves[snr], lw=1.4,
                 color=colors.get(snr, '0.4'), label=fr'SNR $=$ {snr} dB')
    for w0, lab in ((OMEGA1, r'$\omega_1$'), (OMEGA2, r'$\omega_2$')):
        ax2.axvline(w0, color='0.6', lw=0.6, ls=':')
        ax2.text(w0, 1.06, lab, ha='center', fontsize=7, color='0.35')
    ax2.axvline(0.5 * (OMEGA1 + OMEGA2), color='0.8', lw=0.6, ls=':')
    ax2.set_xlim(0.0, 0.45)
    ax2.set_ylim(-0.04, 1.14)
    ax2.set_xlabel(r'normalised frequency $\omega$')
    ax2.set_ylabel(r'$\sum_j S_j\,/\,(\sum_j S_j+\sigma_N^2)$')
    ax2.legend(fontsize=6.4, loc='center left', framealpha=0.92)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r'(b) Rem. V.4: $\to 1$ at the modes, $\to 0$ in the gap',
                  fontsize=8)

    plt.tight_layout(pad=0.35, w_pad=1.1)
    assert np.allclose(nmse_shrunk, nmse_oracle, rtol=1e-12, atol=0), \
        "(12) x (22) must equal the oracle identically"
    out = save_figure(fig, 'mmse_shrinkage', save_dir)
    print(f'\n{"SNR":>5}  {"Hard":>10}  {"Soft":>10}  {"Shrunk":>10}  {"Oracle":>10}')
    print('-' * 56)
    for snr_v, h, sf, sh, o in zip(SNR_DB, nmse_hard, nmse_soft,
                                   nmse_shrunk, nmse_oracle):
        eq = '=' if abs(sh - o) / max(o, 1e-30) < 1e-3 else ' '
        print(f'{snr_v:5d}  {h:10.4f}  {sf:10.4f}  {sh:10.4f} {eq}{o:10.4f}')
    print("\nlargest relative Monte-Carlo standard error: " +
          ", ".join(f"{k} {100*max(v):.2f}%" for k, v in rel_se.items()))
    print("\n'=' : the shrunk mask equals the oracle identically, (12) x (22) = S_k/S_X (not plotted).")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    args = ap.parse_args()
    main(args.save_dir)
