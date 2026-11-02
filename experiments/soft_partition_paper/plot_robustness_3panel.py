"""
Generate the 3-panel fig_robustness.pdf for Experiment C'.

Layout:
  (a) nMSE vs SNR for four spectral shapes
  (b) nMSE vs SNR under centre perturbation
  (c) High-SNR error floor vs centre separation
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where softdamd lives
for _p in (_ROOT, _EXP, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soft_partition_paper.figures import save_figure
from exp_robustness import (
    F, OMEGA1, OMEGA2, ALPHA_TRUE, SIGMA2, SNR_DB, N_TRIALS, SEED,
    BETA, FWHM,
    psd_lorentzian, psd_gaussian, psd_raised_cosine, psd_rectangular,
    soft_mask, hard_mask, run_mc
)


def main():
    matplotlib.rcParams.update({'font.size': 9, 'font.family': 'serif'})

    true_centres = np.array([OMEGA1, OMEGA2])

    # ── Panel (a): spectral shapes ──
    shapes = [
        ('Lorentzian',    psd_lorentzian),
        ('Gaussian',      psd_gaussian),
        ('Raised cosine', psd_raised_cosine),
        ('Rectangular',   psd_rectangular),
    ]
    results_exact = {}
    for name, psd_func in shapes:
        print(f'Running {name}...')
        results_exact[name] = run_mc(psd_func, true_centres, ALPHA_TRUE)

    # ── Panel (b): centre-estimation error ──
    # The TRUE modes stay at (OMEGA1, OMEGA2); only the centres ASSUMED by the masks are
    # shifted (mask_centres).  Shifting the true centres instead would only change the
    # separation, which is panel (c).
    shifts = {
        'Exact centres':             np.array([0.0, 0.0]),
        r'$\pm 1$ FWHM, inward':     np.array([+FWHM, -FWHM]),
        r'$\pm 2$ FWHM, inward':     np.array([+2*FWHM, -2*FWHM]),
        r'$+2$ FWHM, common shift':  np.array([+2*FWHM, +2*FWHM]),
    }
    results_perturbed = {}
    for label, shift in shifts.items():
        print(f'Running centre error: {label}...')
        results_perturbed[label] = run_mc(
            psd_lorentzian, true_centres, ALPHA_TRUE,
            mask_centres=true_centres + shift)
    for label, r in results_perturbed.items():
        print(f'  {label:28s} 40 dB: soft {r["soft"][-1]:.4f}  hard {r["hard"][-1]:.4f}'
              f"   35 dB: soft {r['soft'][-2]:.4f}  hard {r['hard'][-2]:.4f}")

    # ── Panel (c): separation sweep (load precomputed) ──
    sep_data = np.load(os.path.join(_HERE, 'separation_sweep_results.npz'))
    SEP_FWHM = sep_data['SEP_FWHM']

    # ── Figure: 3 panels ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6))

    # --- (a) spectral shapes ---
    colors_shape = {
        'Lorentzian':    '#4363d8',
        'Gaussian':      '#e6194b',
        'Raised cosine': '#f58231',
        'Rectangular':   '#3cb44b',
    }
    for name, _ in shapes:
        c = colors_shape[name]
        r = results_exact[name]
        ax1.semilogy(SNR_DB, r['hard'], '--', color=c, lw=1.0, alpha=0.6)
        ax1.semilogy(SNR_DB, r['soft'], '-',  color=c, lw=1.5, label=name)
        ax1.semilogy(SNR_DB, r['oracle'], ':', color=c, lw=0.8, alpha=0.5)

    ax1.plot([], [], 'k-',  lw=1.5,          label='Soft (solid)')
    ax1.plot([], [], 'k--', lw=1.0, alpha=0.6, label='Hard (dashed)')
    ax1.plot([], [], 'k:',  lw=0.8, alpha=0.5, label='Oracle (dotted)')

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Normalised MSE')
    ax1.legend(fontsize=4.8, loc='upper right', framealpha=0.92, ncol=2)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_title('(a) Spectral shape', fontsize=8)

    # --- (b) centre perturbation ---
    colors_shift = {
        'Exact centres':             '#4363d8',
        r'$\pm 1$ FWHM, inward':     '#f58231',
        r'$\pm 2$ FWHM, inward':     '#e6194b',
        r'$+2$ FWHM, common shift':  '#3cb44b',
    }
    for label in shifts:
        c = colors_shift[label]
        r = results_perturbed[label]
        ax2.plot(SNR_DB, r['soft'] / r['hard'], '-o', color=c, lw=1.4, ms=2.8, label=label)
    ax2.axhline(1.0, color='k', lw=0.8, ls='--')
    ax2.text(SNR_DB[0], 0.99, 'below 1: soft better than hard', fontsize=5.5, color='0.3', va='top')
    ax2.set_ylim(0.78, 1.16)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel(r'nMSE$_{\rm soft}$ / nMSE$_{\rm hard}$')
    ax2.legend(fontsize=4.8, loc='upper left', framealpha=0.92)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) Centre-estimation error', fontsize=8)

    # --- (c) separation sweep: error floor vs separation ---
    floors_hard   = np.array([sep_data[f'sep{s:.1f}_hard'][-1]   for s in SEP_FWHM])
    floors_soft   = np.array([sep_data[f'sep{s:.1f}_soft'][-1]   for s in SEP_FWHM])
    floors_oracle = np.array([sep_data[f'sep{s:.1f}_oracle'][-1] for s in SEP_FWHM])

    ax3.semilogy(SEP_FWHM, floors_hard, 's--', color='#e6194b', lw=1.2,
                 ms=4, label='Hard (Voronoi)')
    ax3.semilogy(SEP_FWHM, floors_soft, 'o-', color='#4363d8', lw=1.5,
                 ms=4, label='Soft (Lorentzian)')
    ax3.semilogy(SEP_FWHM, floors_oracle, 'x:', color='#808080', lw=0.8,
                 ms=4, label='Oracle MMSE')

    # annotate advantage percentages at two extremes
    for i, s in enumerate(SEP_FWHM):
        adv = (1 - floors_soft[i] / floors_hard[i]) * 100
        if s in (1.5, 9.0):
            ax3.annotate(f'{adv:.0f}%',
                         xy=(s, floors_soft[i]),
                         xytext=(0, -12), textcoords='offset points',
                         fontsize=6, ha='center', color='#4363d8')

    ax3.set_xlabel(r'Separation ($\times$ FWHM)')
    ax3.set_ylabel('nMSE floor (40 dB)')
    ax3.legend(fontsize=5.5, loc='upper right', framealpha=0.92)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_title('(c) Centre separation', fontsize=8)

    plt.tight_layout(pad=0.3, w_pad=0.8)

    return fig


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    ap.add_argument('out_dir', nargs='?', default=None, help='same as --save_dir')
    args = ap.parse_args()
    save_figure(main(), 'robustness', args.out_dir or args.save_dir)
