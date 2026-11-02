"""
Plot the centre separation sweep results.

Two visualisations:
  (c1) nMSE vs SNR for each separation level (like panels a/b)
  (c2) High-SNR error floor vs separation (summary plot)

Either can serve as panel (c) of fig_robustness.  The summary plot (c2)
is more compact and conveys the message in one glance.
"""

import os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    matplotlib.rcParams.update({'font.size': 9, 'font.family': 'serif'})

    d = np.load(os.path.join(_HERE, 'separation_sweep_results.npz'))
    SNR_DB   = d['SNR_DB']
    SEP_FWHM = d['SEP_FWHM']
    FWHM     = float(d['FWHM'])

    # ── Option 1: Full nMSE vs SNR curves ──
    fig1, ax1 = plt.subplots(figsize=(3.58, 2.8))
    cmap = plt.cm.viridis
    norm = plt.Normalize(SEP_FWHM.min(), SEP_FWHM.max())

    for sep_f in SEP_FWHM:
        c = cmap(norm(sep_f))
        hard = d[f'sep{sep_f:.1f}_hard']
        soft = d[f'sep{sep_f:.1f}_soft']
        oracle = d[f'sep{sep_f:.1f}_oracle']
        label = f'{sep_f:.0f}' if sep_f == int(sep_f) else f'{sep_f:.1f}'
        ax1.semilogy(SNR_DB, soft, '-', color=c, lw=1.5,
                     label=f'{label} FWHM')
        ax1.semilogy(SNR_DB, hard, '--', color=c, lw=1.0, alpha=0.6)

    ax1.plot([], [], 'k-', lw=1.5, label='Soft (solid)')
    ax1.plot([], [], 'k--', lw=1.0, alpha=0.6, label='Hard (dashed)')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Normalised MSE')
    ax1.legend(fontsize=5.5, loc='upper right', framealpha=0.92, ncol=2)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_title('(c) Centre separation sweep', fontsize=8)
    fig1.tight_layout(pad=0.4)

    for ext in ('pdf', 'png'):
        fig1.savefig(f'fig_separation_curves.{ext}', dpi=200,
                     bbox_inches='tight')
    print('Saved fig_separation_curves.{pdf,png}')

    # ── Option 2: Summary — error floor and advantage vs separation ──
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(7.16, 2.4))

    floors_hard   = np.array([d[f'sep{s:.1f}_hard'][-1]   for s in SEP_FWHM])
    floors_soft   = np.array([d[f'sep{s:.1f}_soft'][-1]   for s in SEP_FWHM])
    floors_oracle = np.array([d[f'sep{s:.1f}_oracle'][-1] for s in SEP_FWHM])
    advantage = (1 - floors_soft / floors_hard) * 100

    ax2a.semilogy(SEP_FWHM, floors_hard, 's--', color='#e6194b', lw=1.2,
                  ms=5, label='Hard (Voronoi)')
    ax2a.semilogy(SEP_FWHM, floors_soft, 'o-', color='#4363d8', lw=1.5,
                  ms=5, label='Soft (Lorentzian)')
    ax2a.semilogy(SEP_FWHM, floors_oracle, 'x:', color='#808080', lw=0.8,
                  ms=5, label='Oracle MMSE')
    ax2a.set_xlabel(r'Centre separation ($\times$ FWHM)')
    ax2a.set_ylabel('nMSE floor (40 dB)')
    ax2a.legend(fontsize=6.5, loc='upper right', framealpha=0.92)
    ax2a.grid(True, which='both', alpha=0.3)
    ax2a.set_title('(c) Error floor vs separation', fontsize=8)

    ax2b.plot(SEP_FWHM, advantage, 'D-', color='#3cb44b', lw=1.5, ms=5)
    ax2b.set_xlabel(r'Centre separation ($\times$ FWHM)')
    ax2b.set_ylabel('Soft advantage over hard (%)')
    ax2b.grid(True, alpha=0.3)
    ax2b.set_ylim(bottom=0)
    ax2b.set_title('(d) Soft mask advantage grows with overlap',
                   fontsize=8)
    fig2.tight_layout(pad=0.4, w_pad=1.0)

    for ext in ('pdf', 'png'):
        fig2.savefig(f'fig_separation_summary.{ext}', dpi=200,
                     bbox_inches='tight')
    print('Saved fig_separation_summary.{pdf,png}')


if __name__ == '__main__':
    main()
