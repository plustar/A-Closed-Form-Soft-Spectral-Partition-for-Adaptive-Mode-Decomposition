# experiments/soft_partition_paper/exp_d_synthetic.py
"""
Hard and soft extraction on the four synthetic signals of ``synthetic_suite.py``,
and the boundary measurement of Experiment D computed on them.

Layout: one column per signal, three rows
  row 0: spectrogram                                        — plot_tf_spectrogram
  row 1: detected centres coloured by the quintile of the energy the soft mask
         allocates to them, E_k = Σ_ω s_k(ω)|𝒯(ω,t)|²      — plot_tf_energy_stratified
  row 2: the allocation difference ½ Σ_k |s_k^soft − s_k^hard| |𝒯|² of every frame, as a
         percentage of the frame energy, with the detected mode count K_t

For each signal the script also prints the quantities of Table IV — boundary exposure
and the allocation difference between the two masks — with the metadata that makes the
four rows comparable (transform, fs, duration, frame count, bin width in hertz, and the
mode count). The numbers are those of this synthetic suite, not of the third-party
recordings used in the paper, which are not included; everything else about the
measurement is identical.
"""

import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where softdamd lives
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.plot_utils import (plot_tf_spectrogram, plot_tf_energy_stratified,
                               setup_font_config)
from soft_partition_paper.figures import save_figure
from soft_partition_paper.synthetic_suite import SUITE
from common.runner import run_damd
from softdamd import lorentzian_weights, voronoi_weights

ALPHA_SOFT = 2000.0
RADII      = (1, 2, 3, 5)      # boundary neighbourhood, in bins


# ── boundary statistics (as in Table IV) ─────────────────────────────────────
# The routines below compute the boundary exposure per signal, together with the two
# quantities the metric needs to be interpretable:
#   * the neighbourhood width in hertz, since one bin is a different physical width
#     in every row of the table;
#   * the allocation difference, an input-power-weighted difference between the hard
#     and the soft allocation -- not energy transferred between modes.  Hard and soft
#     act on the same coefficients in the boundary bins, so the exposure figure on its
#     own cannot support a statement about either rule.

def _boundary_mask(centres_norm, freqs_norm, radius):
    F = freqs_norm.size
    m = np.zeros(F, dtype=bool)
    c = np.sort(np.asarray(centres_norm))
    for j in range(c.size - 1):
        i = int(np.searchsorted(freqs_norm, 0.5 * (c[j] + c[j + 1])))
        m[max(0, i - radius):min(F, i + radius + 1)] = True
    return m


def _soft_hard_weights(freqs_norm, centres_norm, alpha):
    """Eq. (12) and Eq. (7) (ties shared) on the same centres, (K, F) each."""
    return (lorentzian_weights(freqs_norm, centres_norm, alpha),
            voronoi_weights(freqs_norm, centres_norm))


def boundary_report(label, res, fs, duration, alpha=ALPHA_SOFT):
    """Print the boundary row for one signal, with its metadata."""
    p = res.partition
    freqs = np.asarray(res.freqs, dtype=np.float64)
    P_all = np.abs(res.tfmap.coef) ** 2
    n_modes = np.asarray(p.n_modes, dtype=int)
    df = float(freqs[1] - freqs[0])

    acc = {r: [0.0, 0.0, 0.0] for r in RADII}
    for t in range(p.T):
        c = np.asarray(p.centers_at(t), dtype=np.float64)
        P = P_all[:, t]
        tot = float(P.sum())
        if tot <= 0:
            continue
        for rad in RADII:                  # every frame enters the denominator;
            acc[rad][2] += tot             # a single-mode frame has no boundary
        if c.size < 2:
            continue
        soft, hard = _soft_hard_weights(freqs, c, alpha)
        moved_w = 0.5 * np.abs(soft - hard).sum(axis=0)
        for rad in RADII:
            m = _boundary_mask(c, freqs, rad)
            acc[rad][0] += float(P[m].sum())
            acc[rad][1] += float((moved_w * P)[m].sum())

    print(f'\n  [{label}] boundary measurement')
    print(f'    fs = {fs:g} Hz, duration = {duration:.2f} s, frames = {p.T}, '
          f'bins = {freqs.size}')
    print(f'    bin width = {df:.6f} normalised = {df * fs:.3f} Hz')
    print(f'    K_t: mean {n_modes.mean():.1f}, median {int(np.median(n_modes))}, '
          f'range [{n_modes.min()}, {n_modes.max()}], '
          f'total detections {int(n_modes.sum())}')
    print(f"    {'radius':>7}  {'width (Hz)':>11}  {'boundary energy':>16}"
          f"  {'alloc. difference':>18}")
    for rad, (eb, mv, tot) in acc.items():
        if tot <= 0:
            continue
        print(f'    {rad:7d}  {(2 * rad + 1) * df * fs:11.2f}  '
              f'{100 * eb / tot:15.2f}%  {100 * mv / tot:17.2f}%')
    return acc


# ── DAMD runner ──────────────────────────────────────────────────
def run_detection(signal, fs, transform_type):
    """Detection settings are stated explicitly rather than left to defaults.

    n_fft = fs // 2, percentile bandwidth estimation and curvature adaptation are the
    settings used for Experiment D in the paper.
    """
    res, _ = run_damd(signal, fs, transform=transform_type,
                      n_fft=max(int(fs) // 2, 64),
                      bandwidth='percentile', adapt='curvature', alpha=ALPHA_SOFT)
    return res


# ── the four signals ─────────────────────────────────────────────
# (generator, transform title for the panel, label); defined in synthetic_suite.py,
# which states the construction of each signal and its seed.
DATASETS = [(fn, f'{title} Spectrogram', label) for fn, title, label in SUITE]


# ── one row per dataset ─────────────────────────────────────────
def _per_frame_difference(res, alpha=ALPHA_SOFT):
    """Allocation difference of every frame over all bins, as a fraction of frame energy."""
    p = res.partition
    freqs = np.asarray(res.freqs, dtype=np.float64)
    P_all = np.abs(res.tfmap.coef) ** 2
    out = np.zeros(p.T)
    for t in range(p.T):
        c = np.asarray(p.centers_at(t), dtype=np.float64)
        tot = float(P_all[:, t].sum())
        if tot <= 0 or c.size < 2:
            continue
        soft, hard = _soft_hard_weights(freqs, c, alpha)
        out[t] = float((0.5 * np.abs(soft - hard).sum(axis=0) * P_all[:, t]).sum()) / tot
    return out


def _fill_column(axes, res, fs, tf_title, alpha=ALPHA_SOFT):
    """Three panels: spectrogram; soft centres by energy quintile; per-frame difference."""
    nm = np.asarray(res.n_modes)
    T = res.partition.T
    times = np.arange(T) * res.tfmap.hop / fs

    # top: the spectrogram alone
    plot_tf_spectrogram(axes[0], res, fs, title=tf_title, add_colorbar=False)

    # middle: detected centres, coloured by the quintile of the energy the soft mask
    # allocates to them (the mode-cloud spread)
    plot_tf_energy_stratified(axes[1], res, fs, mode='soft',
                              title=f'detected centres by soft-energy quintile '
                                    f'({nm.sum()} total, $K_t$ = {nm.min()}–{nm.max()})')

    # bottom: how much the two allocations differ, frame by frame, with K_t
    d = 100 * _per_frame_difference(res, alpha)
    axes[2].fill_between(times, 0, d, color='#2b5fd9', alpha=0.3)
    axes[2].plot(times, d, color='#2b5fd9', lw=0.9)
    axes[2].set_xlim(times[0], times[-1])
    axes[2].set_ylim(0, max(5.0, 1.1 * d.max()))
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.25)
    axes[2].set_title(f'allocation difference over all bins, % of frame energy '
                      f'(mean {d.mean():.1f}%)', fontsize=9)
    ax2 = axes[2].twinx()
    ax2.step(times, nm, where='mid', color='0.45', lw=0.9)
    ax2.set_ylim(0, nm.max() + 2)
    ax2.set_ylabel('$K_t$ (grey)', color='0.45', fontsize=9)
    ax2.tick_params(labelsize=8, colors='0.45')


# ── main ──────────────────────────────────────────────────────────
def main(save_dir='outputs/soft_partition'):
    setup_font_config(fontsize=10)

    # Signals run across the page; three panels down it.
    n_cols = len(DATASETS)
    fig = plt.figure(figsize=(4.0 * n_cols, 8.4))
    gs  = GridSpec(3, n_cols, figure=fig, height_ratios=[1.0, 1.0, 0.85],
                   hspace=0.46, wspace=0.30,
                   left=0.045, right=0.975,
                   top=0.93, bottom=0.06)
    fig.suptitle(f'Hard versus soft allocation on the four signals of synthetic_suite.py '
                 f'($\\alpha={ALPHA_SOFT:.0f}$; the first and last frames of each record '
                 f'show the transform\'s edge effects)', fontsize=10, y=0.985)

    for col, (loader, tf_title, label) in enumerate(DATASETS):
        axes = [fig.add_subplot(gs[r, col]) for r in range(3)]

        signal, fs, transform = loader()

        print(f'\n[{label}]  N={len(signal)}, fs={fs}, {transform}')

        res = run_detection(signal, fs, transform)

        nm = np.asarray(res.n_modes)
        print(f'  K_total={nm.sum()}, K_mean={nm.mean():.1f}')
        try:
            boundary_report(label, res, fs, len(signal) / fs)
        except Exception as exc:
            print(f'  boundary report failed: {type(exc).__name__}: {exc}')

        _fill_column(axes, res, fs, f'{label} ({tf_title})')

        for ax in axes[:2]:
            ax.tick_params(labelsize=9)
            ax.set_ylabel('Frequency (Hz)' if col == 0 else '', fontsize=10)
        axes[2].tick_params(labelsize=9)
        axes[2].set_ylabel('difference (%)' if col == 0 else '', fontsize=10)

    return save_figure(fig, 'synthetic_suite', save_dir)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    args = ap.parse_args()
    main(args.save_dir)
