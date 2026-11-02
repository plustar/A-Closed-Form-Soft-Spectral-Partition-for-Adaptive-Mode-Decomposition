# experiments/soft_partition_paper/exp_h.py
"""
fig_closed_form  —  Theorem IV.3, and the three functionals on the same simplex

WHY THIS FIGURE EXISTS
----------------------
Theorem IV.3 carries the paper's central claim -- that Eq. (12) is the global minimiser
of the regularised objective (11), unique wherever P > 0.  This figure checks it numerically
(Experiment S3) and, in the same frame, locates the three allocation rules the paper discusses.

WHAT IS SHOWN
-------------
(a) The per-frequency problem is a diagonal quadratic programme on the simplex, so it
    can be drawn in full for K = 3: the objective is plotted over the 2-simplex in
    barycentric coordinates and the closed form (11), the numerical minimiser, and the
    two competing rules are marked on it.  The closed form sits on the numerical
    minimiser; the hard Voronoi mask sits at a vertex; the inverse-square allocation
    sits elsewhere in the interior.

(b) The three rules optimise three different functionals over the same feasible set:

        hard Voronoi     minimises the LINEAR K-means functional   sum_k d_k^2 s_k P
        inverse-square   minimises the unregularised QUADRATIC     sum_k d_k^2 s_k^2 P
        Lorentzian (11)  minimises the regularised QUADRATIC       sum_k (d_k^2+1/2a) s_k^2 P

    Each rule is evaluated under each functional, normalised by the optimum of that
    functional.  Every rule attains 1 under its own functional and strictly more than 1
    under the others, which is Theorem III.1 and Corollary III.2 seen as a table.

Verification is against a generic SLSQP minimisation, never against a second
evaluation of the closed form.
"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where softdamd lives
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.plot_utils import setup_font_config
from soft_partition_paper.figures import save_figure

FS      = 1000.0
CENTRES = np.array([0.08, 0.20, 0.35])      # normalised frequency, f/fs
W_EVAL  = 0.155                             # the frequency panel (a) is drawn at
ALPHA   = 2000.0
N_GRID  = 320
SEED    = 0
N_RAND  = 400


# ── the three functionals, per frequency (P factors out of every argmin) ─────
def cost_regularised(w, centres, alpha):
    return (w - centres) ** 2 + 1.0 / (2.0 * alpha)


def cost_unregularised(w, centres):
    return (w - centres) ** 2


def J_quadratic(s, c):
    return float(np.asarray(c) @ (np.asarray(s) ** 2))


def J_linear(s, c):
    return float(np.asarray(c) @ np.asarray(s))


# ── the three allocation rules ───────────────────────────────────────────────
def rule_lorentzian(w, centres, alpha):
    a = 1.0 / (1.0 + 2.0 * alpha * (w - centres) ** 2)
    return a / a.sum()


def rule_inverse_square(w, centres, eps=1e-15):
    d = np.maximum(np.abs(w - centres), eps)
    a = d ** -2.0
    return a / a.sum()


def rule_voronoi(w, centres):
    s = np.zeros(centres.size)
    s[int(np.argmin(np.abs(w - centres)))] = 1.0
    return s


# ── independent numerical minimisers ─────────────────────────────────────────
def argmin_simplex(c, quadratic=True, n_restarts=4, seed=SEED):
    K = c.size
    cons = ({'type': 'eq', 'fun': lambda s: s.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    rng = np.random.default_rng(seed)
    f = (lambda s: J_quadratic(s, c)) if quadratic else (lambda s: J_linear(s, c))
    g = ((lambda s: 2.0 * c * s) if quadratic else (lambda s: c))
    best, bv = None, np.inf
    for s0 in [np.full(K, 1.0 / K)] + [rng.dirichlet(np.ones(K))
                                       for _ in range(n_restarts - 1)]:
        r = minimize(f, s0, jac=g, method='SLSQP', bounds=bnds, constraints=cons,
                     options={'ftol': 1e-14, 'maxiter': 600})
        if r.fun < bv:
            best, bv = r.x, r.fun
    return np.asarray(best)


# ── barycentric helpers for the K = 3 simplex ────────────────────────────────
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def bary_to_xy(s):
    return np.asarray(s) @ _V


def main(save_dir='outputs/soft_partition'):
    setup_font_config(fontsize=9)
    matplotlib.rcParams.update({'font.size': 9})
    rng = np.random.default_rng(SEED)

    c_reg = cost_regularised(W_EVAL, CENTRES, ALPHA)
    c_unr = cost_unregularised(W_EVAL, CENTRES)

    s_closed = rule_lorentzian(W_EVAL, CENTRES, ALPHA)
    s_numer = argmin_simplex(c_reg, quadratic=True)
    s_inv = rule_inverse_square(W_EVAL, CENTRES)
    s_hard = rule_voronoi(W_EVAL, CENTRES)

    print(f'  frequency w = {W_EVAL} ({W_EVAL * FS:.1f} Hz), centres '
          f'{np.round(CENTRES * FS, 1)} Hz, alpha = {ALPHA:g}')
    print(f'  closed form  Eq. (12) = {np.round(s_closed, 6)}')
    print(f'  SLSQP minimiser       = {np.round(s_numer, 6)}')
    print(f'  max difference        = {np.abs(s_closed - s_numer).max():.3e}')
    print(f'  objective at closed form = {J_quadratic(s_closed, c_reg):.12f}')
    print(f'  objective at SLSQP       = {J_quadratic(s_numer, c_reg):.12f}')

    # ── panel (a): the objective over the 2-simplex ─────────────────────────
    lin = np.linspace(0.0, 1.0, N_GRID)
    A, B = np.meshgrid(lin, lin)
    C = 1.0 - A - B
    inside = C >= 0
    S = np.stack([A, B, C], axis=-1)
    Z = np.where(inside, (S ** 2 * c_reg).sum(axis=-1), np.nan)
    XY = S.reshape(-1, 3) @ _V
    Xg = XY[:, 0].reshape(A.shape)
    Yg = XY[:, 1].reshape(A.shape)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

    ax = axes[0]
    cf = ax.contourf(Xg, Yg, Z, levels=28, cmap='viridis', alpha=0.92)
    ax.contour(Xg, Yg, Z, levels=14, colors='w', linewidths=0.35, alpha=0.5)
    tri = np.vstack([_V, _V[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color='0.25', lw=1.0)

    xy = bary_to_xy(s_numer)
    ax.scatter([xy[0]], [xy[1]], marker='o', s=110, facecolors='none',
               edgecolors='w', linewidths=1.8, zorder=5, label='SLSQP minimiser')
    xy = bary_to_xy(s_closed)
    ax.scatter([xy[0]], [xy[1]], marker='x', s=52, c='#e6194b', linewidths=1.9,
               zorder=6, label=r'closed form, Eq. (12)')
    xy = bary_to_xy(s_inv)
    ax.scatter([xy[0]], [xy[1]], marker='s', s=26, c='#f58231',
               edgecolors='0.15', linewidths=0.9, zorder=6,
               label=r'inverse-square ($\alpha\to\infty$)')
    xy = bary_to_xy(s_hard)
    ax.scatter([xy[0]], [xy[1]], marker='^', s=34, c='#4363d8',
               edgecolors='0.15', linewidths=0.9, zorder=6,
               label='hard Voronoi (a vertex)')
    for i, lab in enumerate((r'$s_1{=}1$', r'$s_2{=}1$', r'$s_3{=}1$')):
        off = (_V[i] - _V.mean(axis=0)) * 0.16
        ax.text(_V[i, 0] + off[0], _V[i, 1] + off[1], lab, fontsize=6.6,
                ha='center', va='center', color='0.25')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.14, 1.02)
    ax.legend(fontsize=5.8, loc='upper right', framealpha=0.9,
              bbox_to_anchor=(1.04, 1.0), handletextpad=0.4, borderpad=0.35)
    ax.set_title(r'(a) Thm. IV.3: the regularised objective on $\Delta^3$'
                 '\n'
                 fr'at $\omega={W_EVAL * FS:.0f}$ Hz', fontsize=8)
    cb = fig.colorbar(cf, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label(r'$\sum_k c_k s_k^2$', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ── panel (b): each rule under each functional ──────────────────────────
    ws = np.linspace(0.02, 0.45, 220)
    rules = {
        'hard Voronoi': lambda w: rule_voronoi(w, CENTRES),
        'inverse-square': lambda w: rule_inverse_square(w, CENTRES),
        r'Lorentzian, Eq. (12)': lambda w: rule_lorentzian(w, CENTRES, ALPHA),
    }
    funcs = {
        'linear (K-means)': ('lin', lambda w: cost_unregularised(w, CENTRES)),
        'quadratic': ('quad', lambda w: cost_unregularised(w, CENTRES)),
        'regularised quadratic': ('quad', lambda w: cost_regularised(w, CENTRES, ALPHA)),
    }

    table = np.zeros((len(rules), len(funcs)))
    for j, (fname, (kind, cf_fn)) in enumerate(funcs.items()):
        num = []
        for w in ws:
            c = cf_fn(w)
            quad = kind == 'quad'
            s_opt = argmin_simplex(c, quadratic=quad, n_restarts=2)
            Jopt = J_quadratic(s_opt, c) if quad else J_linear(s_opt, c)
            row = []
            for rname, rfn in rules.items():
                s = rfn(w)
                Jv = J_quadratic(s, c) if quad else J_linear(s, c)
                row.append(Jv / max(Jopt, 1e-300))
            num.append(row)
        table[:, j] = np.mean(np.array(num), axis=0)

    ax2 = axes[1]
    from matplotlib.colors import LogNorm
    im = ax2.imshow(table, cmap='RdYlGn_r', aspect='auto',
                    norm=LogNorm(vmin=1.0, vmax=table.max()))
    ax2.set_xticks(range(len(funcs)))
    ax2.set_xticklabels(['linear\n(K-means)', 'quadratic\n(unregularised)',
                         'quadratic\n(regularised)'], fontsize=6.8)
    ax2.set_yticks(range(len(rules)))
    ax2.set_yticklabels(list(rules), fontsize=6.8)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table[i, j]
            ax2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.4,
                     color='w' if v > 8.0 else '0.1',
                     fontweight='bold' if abs(v - 1.0) < 1e-6 else 'normal')
    ax2.set_xlabel('functional being minimised', fontsize=7.4)
    ax2.set_ylabel('allocation rule', fontsize=7.4)
    ax2.set_title('(b) each rule optimises its own functional\n'
                  r'(objective $/$ optimum, averaged over $\omega$)', fontsize=8)
    cb2 = fig.colorbar(im, ax=ax2, fraction=0.042, pad=0.02)
    cb2.ax.tick_params(labelsize=6)
    cb2.set_label('objective / optimum (log scale)', fontsize=6.5)

    plt.tight_layout()
    out = save_figure(fig, 'closed_form', save_dir)

    # ── random verification of Theorem IV.3 ─────────────────────────────────
    errs = []
    for _ in range(N_RAND):
        K = int(rng.integers(2, 7))
        cen = np.sort(rng.uniform(0.0, 0.5, K))
        wq = float(rng.uniform(0.0, 0.5))
        a = float(10.0 ** rng.uniform(1.0, 5.0))
        c = cost_regularised(wq, cen, a)
        errs.append(float(np.abs(rule_lorentzian(wq, cen, a)
                                 - argmin_simplex(c, seed=1)).max()))
    errs = np.array(errs)
    print(f'\n  Theorem IV.3 over {N_RAND} random configurations, K in [2, 6],'
          f' alpha in [1e1, 1e5]:')
    print(f'    max |Eq. (12) - SLSQP| : median {np.median(errs):.3e}'
          f'   max {errs.max():.3e}')

    print(f"\n  panel (b), objective / optimum averaged over omega:")
    print(f"    {'rule':>22}  " + '  '.join(f'{k:>22}' for k in funcs))
    for i, rname in enumerate(rules):
        print(f'    {rname:>22}  '
              + '  '.join(f'{table[i, j]:22.4f}' for j in range(len(funcs))))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    args = ap.parse_args()
    main(args.save_dir)
