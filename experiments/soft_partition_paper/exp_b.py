# experiments/soft_partition_paper/exp_b.py
"""
figS_wiener_identity  —  implementation consistency check (supplementary)

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Corollary IV.5 states that Eq. (12) and Eq. (6) are the same expression.  Remark IV.2
concedes that the regulariser 1/(2a) of Definition IV.1 was chosen precisely so that
the inverse cost reproduces the Lorentzian denominator of Eq. (5), so the identity
holds BY CONSTRUCTION: it is a definitional coincidence, not a derived result.

This figure therefore verifies only that two independent implementations of that one
expression agree -- ``softdamd.lorentzian_weights`` evaluated on the pipeline's
detected centres, and ``vmd_wiener_weights`` below, written standalone.  It is a unit test of the code, and it
is evidence for nothing else.  In particular it says nothing about whether the mask is
the output of the VMD ADMM solver; that question is settled analytically and measured
against the actual solver output in exp_a.py (fig_vmd_agreement).

It belongs in the supplementary material for that reason.
"""

import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where softdamd lives
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.plot_utils import setup_font_config
from soft_partition_paper.figures import save_figure
from common.runner import run_damd, freq_axis

FS        = 1000.0
DURATION  = 2.0
N_FFT     = 256
F1, F2    = 80.0, 200.0
ALPHA_VMD = 2000.0
SEED      = 0
COLORS    = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#42d4f4']


def vmd_wiener_weights(freqs, centers, alpha):
    """Standalone implementation of the same expression, written independently of
    the package: S_k(f) = [1 + 2α(f - ω_k)²]⁻¹ / Σ_j [1 + 2α(f - ω_j)²]⁻¹."""
    K = len(centers)
    F = len(freqs)
    S = np.zeros((K, F))
    for k in range(K):
        S[k] = 1.0 / (1.0 + 2.0 * alpha * (freqs - centers[k]) ** 2)
    Z = S.sum(axis=0, keepdims=True) + 1e-15
    return S / Z


def main(save_dir='outputs/soft_partition'):
    setup_font_config(fontsize=10)
    matplotlib.rcParams.update({'font.size': 10})

    rng = np.random.default_rng(SEED)
    t   = np.linspace(0, DURATION, int(FS*DURATION), endpoint=False)
    x   = np.sin(2*np.pi*F1*t) + np.sin(2*np.pi*F2*t)
    x  += 0.01 * rng.standard_normal(len(t))

    res, _   = run_damd(x, FS, transform='stft', n_fft=N_FFT, adapt='none')
    p        = res.partition
    freqs_hz = freq_axis(res, FS)
    freqs_n  = res.freqs

    t_idx   = int(p.T // 2)
    K_t     = int(p.n_modes[t_idx])
    centers = p.centers_at(t_idx)

    # Closed-form weights (Theorem IV.3), package implementation of Eq. (12)
    W_closed = res.weights(t_idx, alpha=ALPHA_VMD)      # (K_t, F)
    # VMD Wiener formula (independent implementation)
    W_vmd    = vmd_wiener_weights(freqs_n, centers, ALPHA_VMD)  # (K_t, F)
    max_err  = np.max(np.abs(W_closed - W_vmd))

    # Frequencies are normalised as f/fs in [0, 0.5]; 80 Hz at fs = 1000 is 0.08.
    targets   = [F1 / FS, F2 / FS]
    mode_idxs = [int(np.argmin(np.abs(centers - tgt))) for tgt in targets]
    picked_hz = [float(centers[i] * FS) for i in mode_idxs]
    labels    = [f'Mode at {h:.1f} Hz' for h in picked_hz]
    print(f'targets (normalised)  = {np.round(targets, 4)}'
          f'  -> {np.round(np.array(targets) * FS, 1)} Hz')
    print(f'selected mode centres = {np.round(picked_hz, 2)} Hz '
          f'(indices {mode_idxs} of {K_t})')

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for ki, label in zip(mode_idxs, labels):
        color = COLORS[ki % len(COLORS)]
        ax.plot(freqs_hz, W_closed[ki], '-',  color=color, lw=2.0,
                label=label)
        ax.plot(freqs_hz, W_vmd[ki],   '--',  color='k',   lw=0.8,
                alpha=0.6)

    # Dummy entries for formula legend
    ax.plot([], [], '-',  color='gray', lw=2.0,
            label=r'pipeline implementation of Eq. (12)')
    ax.plot([], [], '--', color='k',    lw=0.8, alpha=0.6,
            label=r'standalone implementation')

    ax.text(0.97, 0.55,
            f'$\\|s_k^* - w_k\\|_\\infty = {max_err:.1e}$\n'
            'two implementations of\none expression',
            transform=ax.transAxes, ha='right', va='center', fontsize=6.4,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Weight $s_k^*(\\omega)$')
    ax.set_xlim(0, FS/2)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=1, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'Implementation consistency, Rem. IV.5 '
        f'($\\alpha={ALPHA_VMD:.0f}$, frame {t_idx})',
        fontsize=8.5
    )

    plt.tight_layout()
    out = save_figure(fig, 'wiener_identity', save_dir)
    print(f'K={K_t} modes  max |s* - w| = {max_err:.2e}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    args = ap.parse_args()
    main(args.save_dir)