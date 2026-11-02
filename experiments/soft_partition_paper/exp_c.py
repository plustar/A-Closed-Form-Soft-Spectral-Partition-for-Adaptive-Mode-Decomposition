# experiments/soft_partition_paper/exp_c.py
"""
Experiment B  —  fig_lorentzian_limits
Corollary IV.9 + Remark IV.10
====================================================

Corollary IV.9 gives both endpoints of the Lorentzian family:

    alpha -> 0    s_k* -> 1/K                            (uniform)
    alpha -> inf  s_k* -> d_k^{-2} / sum_j d_j^{-2}      (inverse-square)

and Remark IV.10 states that the second is NOT the hard Voronoi indicator, so the
hard mask lies outside the family.  The inverse-square endpoint is also where the
family meets the VMD ADMM fixed point (see exp_a.py), which is what makes it the
structurally interesting end.

This script draws both limits and measures the distance to each.

Panel (a)  Weight profile s_1(omega; alpha) across the full frequency
           axis for alpha in {50, 200, 2000, 1e5}, overlaid with:
             - inverse-square asymptote  d_k^{-2}/(d_k^{-2}+d_j^{-2})
                                                              (dashed)
             - hard Voronoi indicator    1_{B_1}              (dotted)
           As alpha grows, the Lorentzian collapses onto the
           inverse-square S-curve, *not* onto the Voronoi step.
           At the midpoint omega = (f_1+f_2)/2 the Lorentzian weight
           equals 0.5 for every alpha tested (the counter-example of Remark IV.10).

Panel (b)  Normalised L^2 extraction discrepancy
             D(alpha) = || U_soft(alpha) - U_hard ||_2 / || U_hard ||_2
           where U_soft = s_1(.; alpha) * X and U_hard = 1_{B_1} * X,
           on a narrow-band two-tone signal.  D(alpha) decreases with
           alpha and saturates at a non-zero plateau set by the
           inverse-square mismatch, confirming the non-vanishing
           soft-vs-hard discrepancy as alpha -> infinity.

Frequency convention
--------------------
The Lorentzian formula 1 / (1 + 2*alpha*(omega - omega_k)^2) uses the
*normalised* frequency omega in [0, 1/2], matching the VMD / DAMD
convention under which alpha = 2000 is the paper's default.  Internal
math therefore uses normalised frequency; the x-axis is displayed in
Hz (= normalised * FS) for the reader's convenience.
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

from soft_partition_paper.figures import save_figure

try:
    from common.plot_utils import setup_font_config
except ImportError:
    def setup_font_config(fontsize=10):
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.size':   fontsize,
            'text.usetex': False,
        })


# ── Parameters ──────────────────────────────────────────────────────
FS       = 1000.0          # sampling rate (Hz)
DURATION = 4.0             # signal duration (s) for L^2 panel
F1_HZ    = 80.0            # mode 1 centre in Hz
F2_HZ    = 200.0           # mode 2 centre in Hz
SEED     = 42

# Normalised centre frequencies (omega in [0, 1/2]) for internal math
W1 = F1_HZ / FS
W2 = F2_HZ / FS
W_MID = 0.5 * (W1 + W2)

# alpha values for the weight-profile panel (matches paper convention)
ALPHA_SHOW    = [1e-2, 50.0, 200.0, 2000.0, 1e5]
ALPHA_LABELS  = [r'$\alpha=10^{-2}$', r'$\alpha=50$', r'$\alpha=200$',
                 r'$\alpha=2000$',    r'$\alpha=10^{5}$']
ALPHA_COLORS  = ['#911eb4', '#e6194b', '#f58231', '#4363d8', '#3cb44b']

# alpha sweep for the L^2 discrepancy panel
ALPHAS_SWEEP  = np.logspace(-1, 6, 50)

# Dense normalised frequency grid for visual weight profiles
W_GRID = np.linspace(0.0, 0.5, 1024)


# ── Core Lorentzian / Voronoi formulas (normalised frequency) ───────
def lorentzian_weight(omega_n, omega_k, omega_other, alpha):
    """Two-mode Lorentzian soft weight  s_k(omega_n; alpha)."""
    a_k = 1.0 / (1.0 + 2.0 * alpha * (omega_n - omega_k)**2)
    a_o = 1.0 / (1.0 + 2.0 * alpha * (omega_n - omega_other)**2)
    return a_k / (a_k + a_o)


def inv_square_limit(omega_n, omega_k, omega_other, eps=1e-15):
    """alpha -> infinity limit: d_k^{-2} / (d_k^{-2} + d_o^{-2})."""
    d_k = np.abs(omega_n - omega_k)
    d_o = np.abs(omega_n - omega_other)
    d_k_safe = np.where(d_k < eps, eps, d_k)
    d_o_safe = np.where(d_o < eps, eps, d_o)
    s = d_k_safe**-2 / (d_k_safe**-2 + d_o_safe**-2)
    s = np.where(d_k < eps, 1.0, s)
    s = np.where(d_o < eps, 0.0, s)
    return s


def voronoi_indicator(omega_n, omega_k, omega_other):
    """Hard Voronoi mask: 1 where omega is at least as close to omega_k."""
    return (np.abs(omega_n - omega_k) <= np.abs(omega_n - omega_other)).astype(float)


def make_narrowband_two_tone(fs, duration, f1_hz, f2_hz, seed):
    """Two AM-modulated tones; returns FFT spectrum X, normalised
    frequency axis (in [0, 1/2]), and Hz axis.

    Mild AM gives the spectrum a realistic narrow but finite spread
    around each centre, so that the L^2 discrepancy plateau in panel (b)
    is small but strictly positive (consistent with rem:not_voronoi).
    """
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.linspace(0.0, duration, N, endpoint=False)
    env1 = 1.0 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
    env2 = 1.0 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    x  = env1 * np.sin(2 * np.pi * f1_hz * t)
    x += env2 * np.sin(2 * np.pi * f2_hz * t)
    x += 0.005 * rng.standard_normal(N)
    X = np.fft.rfft(x)
    f_hz = np.fft.rfftfreq(N, d=1.0 / fs)
    w_n  = f_hz / fs                      # normalised: 0 .. 1/2
    return X, w_n, f_hz


# ── Main ────────────────────────────────────────────────────────────
def main(save_dir='outputs/soft_partition'):
    setup_font_config()
    matplotlib.rcParams.update({'font.size': 10})

    # ---- Panel (a) data: weight profiles on dense visual grid -------
    profiles    = [lorentzian_weight(W_GRID, W1, W2, a) for a in ALPHA_SHOW]
    invsq_curve = inv_square_limit(W_GRID, W1, W2)
    voronoi_a   = voronoi_indicator(W_GRID, W1, W2)
    F_GRID_HZ   = W_GRID * FS

    # ---- Panel (b) data: L^2 discrepancy on FFT grid ----------------
    X, w_fft, f_fft_hz = make_narrowband_two_tone(FS, DURATION,
                                                  F1_HZ, F2_HZ, SEED)
    s_hard       = voronoi_indicator(w_fft, W1, W2)
    U_hard       = s_hard * X
    norm_hard    = np.linalg.norm(U_hard)
    if norm_hard == 0.0:
        raise RuntimeError("Hard extraction has zero energy; check signal.")

    discrepancy = np.empty(len(ALPHAS_SWEEP))
    for i, alpha in enumerate(ALPHAS_SWEEP):
        s_soft = lorentzian_weight(w_fft, W1, W2, alpha)
        U_soft = s_soft * X
        discrepancy[i] = np.linalg.norm(U_soft - U_hard) / norm_hard

    s_inv     = inv_square_limit(w_fft, W1, W2)
    U_inv     = s_inv * X
    plateau   = float(np.linalg.norm(U_inv - U_hard) / norm_hard)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))

    # ---- Panel (a) ----
    ax = axes[0]
    ax.plot(F_GRID_HZ, voronoi_a, color='k', ls=':',  lw=1.4, alpha=0.75,
            label=r'Hard Voronoi $\mathbb{1}_{B_1}$')
    ax.plot(F_GRID_HZ, invsq_curve, color='0.15', ls='-', lw=3.2, alpha=0.30,
            solid_capstyle='round',
            label=r'Inverse-square limit ($\alpha\to\infty$)')
    ax.axhline(0.5, color='0.15', ls='-', lw=3.2, alpha=0.30,
               label=r'Uniform $1/K$ ($\alpha\to0$)')
    for w, lbl, c in zip(profiles, ALPHA_LABELS, ALPHA_COLORS):
        ax.plot(F_GRID_HZ, w, color=c, lw=1.4, alpha=0.95, label=lbl)

    ax.axvline(F1_HZ,         color='gray', lw=0.6, ls=':', alpha=0.45)
    ax.axvline(F2_HZ,         color='gray', lw=0.6, ls=':', alpha=0.45)
    ax.axvline(W_MID * FS,    color='gray', lw=0.5, ls=':', alpha=0.35)

    # Annotate the fixed point at the midpoint: s_1(140) = 0.5 for any
    # alpha.  Place the annotation low-right
    # where no Lorentzian curve passes.
    ax.annotate(r'$s_1(\omega\!=\!140) = 0.5$' + '\n' + r'for all $\alpha$',
                xy=(W_MID * FS, 0.5),
                xytext=(280, 0.62),
                fontsize=7.0,
                ha='left',
                arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))

    ax.set_xlim(0, FS / 2.0)
    ax.set_ylim(-0.05, 1.18)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Weight $s_1(\omega;\alpha)$  (mode at $f_1\!=\!80$ Hz)')
    # Legend placed at top: single row of 3 colour curves + 2 references
    # below.  Top area is where curves are densely packed near 1 (near
    # f=80 Hz only); spreading across the top keeps it clear of the
    # crossing region around 140-200 Hz.
    ax.legend(fontsize=5.6, ncol=2, loc='upper right', framealpha=0.92,
              columnspacing=0.9, handlelength=1.5, borderpad=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'(a) Cor. IV.9: uniform $\leftarrow\alpha\rightarrow$ inverse-square;'
                 '\n'
                 r'Rem. IV.10: never the Voronoi step',
                 fontsize=8, pad=4)

    # ---- Panel (b) ----
    ax2 = axes[1]
    ax2.loglog(ALPHAS_SWEEP, discrepancy, '-o', ms=3.2, lw=1.4,
                 color='#4363d8',
                 label=(r'$\|\hat U^{\rm soft}-\hat U^{\rm hard}\|_2'
                        r'/\|\hat U^{\rm hard}\|_2$'))
    ax2.axhline(plateau, color='k', lw=1.0, ls='--',
                label=fr'Plateau (inv-sq limit): {plateau:.4f}')
    ax2.set_xlabel(r'Bandwidth parameter $\alpha$')
    ax2.set_ylabel(r'Normalised $L^2$ discrepancy')
    ax2.legend(fontsize=7.0, loc='upper right', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_ylim(plateau * 0.35, max(float(discrepancy.max()), plateau) * 1.8)
    ax2.annotate(f'plateau {plateau:.4f}, strictly positive:\n'
                 r'the hard mask is not in the family',
                 xy=(ALPHAS_SWEEP[-1], plateau),
                 xytext=(ALPHAS_SWEEP[6], plateau * 1.8),
                 fontsize=6.2, ha='left',
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='0.35'))
    ax2.set_title(r'(b) Rem. IV.10: the soft-vs-hard gap saturates'
                  '\n'
                  r'at a strictly positive plateau as $\alpha\!\to\!\infty$',
                  fontsize=8)

    plt.tight_layout(pad=0.35, w_pad=1.1)
    out_pdf = save_figure(fig, 'lorentzian_limits', save_dir)

    # Console summary
    print()
    print(f'Lorentzian at midpoint omega = {W_MID*FS:.0f} Hz '
          f'(normalised omega = {W_MID:.3f}):')
    for a in ALPHA_SHOW:
        v = lorentzian_weight(np.array([W_MID]), W1, W2, a)[0]
        print(f"   alpha = {a:>8.1e}   s_1(midpoint) = {v:.6f}")
    v_inf = inv_square_limit(np.array([W_MID]), W1, W2)[0]
    print(f"   alpha = inf        s_1(midpoint) = {v_inf:.6f}  (inv-sq limit)")
    print()
    print(f'Lorentzian peak self-weight at omega = 80 Hz '
          f'(normalised W1 = {W1:.3f}):')
    for a in ALPHA_SHOW:
        v = lorentzian_weight(np.array([W1]), W1, W2, a)[0]
        print(f"   alpha = {a:>8.1e}   s_1(80 Hz) = {v:.6f}")
    print()
    print('Corollary IV.9, both endpoints (max over the frequency grid):')
    print(f"   {'alpha':>10}  {'max |s_1 - 1/K|':>17}  {'max |s_1 - inv-sq|':>20}")
    for a in [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8]:
        prof = lorentzian_weight(W_GRID, W1, W2, a)
        print(f"   {a:10.0e}  {np.max(np.abs(prof - 0.5)):17.3e}"
              f"  {np.max(np.abs(prof - invsq_curve)):20.3e}")
    print()
    print('Remark IV.10, the residual is asymptotic rather than exact:')
    s_inf = inv_square_limit(np.array([0.9 / FS * FS]), 0.0, 2.0)[0]
    print(f"   {'alpha':>10}  {'|s_1(0.9) - inv-sq|':>21}   (centres 0 and 2)")
    for a in [1e0, 1e1, 1e2, 1e3, 1e4, 1e6]:
        d = abs(lorentzian_weight(np.array([0.9]), 0.0, 2.0, a)[0] - s_inf)
        print(f"   {a:10.0e}  {d:21.3e}")
    print()
    print(f'L^2 discrepancy summary:')
    print(f"   at alpha = {ALPHAS_SWEEP[0]:.1e}: {discrepancy[0]:.4f}  (broad / uniform)")
    print(f"   at alpha = {ALPHAS_SWEEP[-1]:.1e}: {discrepancy[-1]:.4f}  (plateau)")
    print(f"   inv-square vs hard plateau:    {plateau:.4f}")
    return out_pdf


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args.save_dir)
