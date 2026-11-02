# experiments/soft_partition_paper/exp_g.py
"""
fig_runtime  —  Experiment S5 (supplement, Appendix F; summarised in Section VII-B):
Table II and Algorithm 1, where the cost goes

WHAT THE PAPER CLAIMS
---------------------
STVMD solves a variational problem at every one of the T frames, so its cost carries
the ADMM iteration count N_A and a mode count K that must be fixed in advance:

    C_STVMD = O(T . F log F)  +  O(T . K . N_A . F)             iterative, K preset

Algorithm 1 replaces the per-frame ADMM by one density-clustering pass, which also
returns K_t automatically, and this paper adds a closed-form soft extraction costing one
elementwise pass per detected mode:

    C_DAMD       = O(T . F log F)  +  O(T . S . N_M . F)
    C_allocation = O(T . F . K_t)                                no iteration at all

The allocation is timed by evaluating Eq. (12) for every frame and applying it to
that frame's coefficients (``allocate_all_frames``).

WHAT IS REPORTED
----------------
* the mode counts each method actually returns, since STVMD runs at a fixed K
  while DAMD chooses K_t per frame -- the two do not solve the same problem and the
  end-to-end ratio is a pipeline comparison, not a speedup for an identical
  decomposition;
* the stages separately rather than one wall-clock number;
* hardware, versions, threading, warm-up policy, repetitions and dispersion
  (N_REPS repetitions per point).

Panel (a) is the stage-separated cost against signal length.  Panel (b) is the mode
count each method returns at the largest length, which is the information needed to
read panel (a) honestly.

This script runs STVMD and is slow: at hop = 1 the number of frames equals the number
of samples.  Use --quick for a first pass, and --lengths to control the sweep.

Usage
-----
    python experiments/soft_partition_paper/exp_g.py --save_dir figs
    python experiments/soft_partition_paper/exp_g.py --quick
    python experiments/soft_partition_paper/exp_g.py --lengths 256 512 1024 2048 4096
"""

import json
import os
import platform
import sys
import time

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

FS          = 1000.0
N_FFT       = 128
HOP         = 1
K_STVMD     = 10          # STVMD needs K in advance; this is the value it is given
ALPHA_STVMD = 50.0        # original VMD convention: denominator 1 + alpha_orig * d^2
ALPHA       = 2000.0      # paper convention:      denominator 1 + 2 * alpha * d^2
LENGTHS     = (64, 128, 256, 512, 1024, 2048)
N_REPS      = 9
N_WARMUP    = 1
SEED        = 42
COL         = {'stvmd': '#e6194b', 'transform': '#911eb4',
               'detect': '#4363d8', 'alloc': '#3cb44b'}


def make_signal(n, fs=FS, seed=SEED):
    """Three time-varying components: the regime STVMD exists for.

    Broadband noise would make every method degenerate and would not exercise the
    detection stage, so a signal with genuine modes is used instead.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    span = max(t[-1], 1e-9)
    x = np.sin(2 * np.pi * (60 + 20 * np.sin(2 * np.pi * 0.5 * t)) * t)
    x += 0.7 * np.sin(2 * np.pi * (180 + 40 * t / span) * t)
    x += 0.5 * np.sin(2 * np.pi * 320 * t) * (t > span / 2)
    return x + 0.05 * rng.standard_normal(n)


def timed(fn, reps=N_REPS, warmup=N_WARMUP):
    """Median and interquartile range, after untimed warm-up calls."""
    for _ in range(warmup):
        fn()
    ts = np.array([(lambda t0: (fn(), time.perf_counter() - t0)[1])(time.perf_counter())
                   for _ in range(reps)])
    return float(np.median(ts)), float(np.percentile(ts, 75) - np.percentile(ts, 25))


def env_line():
    try:
        import scipy
        sp = scipy.__version__
    except Exception:
        sp = 'n/a'
    thr = os.environ.get('OMP_NUM_THREADS', 'unset')
    return (f"{platform.python_version()} / numpy {np.__version__} / scipy {sp} / "
            f"{platform.machine()} / {os.cpu_count()} cores / OMP_NUM_THREADS={thr}")


CACHE_NAME = 'runtime_timings.json'


def cache_path(save_dir):
    return os.path.join(save_dir, CACHE_NAME)


def load_cache(save_dir):
    p = cache_path(save_dir)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.get('by_length', {}).items()}


def save_cache(save_dir, by_length, meta):
    os.makedirs(save_dir, exist_ok=True)
    with open(cache_path(save_dir), 'w', encoding='utf-8') as fh:
        json.dump({'meta': meta,
                   'by_length': {str(k): v for k, v in sorted(by_length.items())}},
                  fh, indent=1)
    print(f'Saved {cache_path(save_dir)}')


def build_damd():
    """STFT (n_fft = N_FFT, hop = HOP) and mean-shift detection with its defaults
    (unit bandwidth, no adaptation, seed spacing 5, 1000 iterations)."""
    from softdamd import SoftDAMD, STFT, MeanShift
    return SoftDAMD(STFT(n_fft=N_FFT, hop=HOP), MeanShift(), alpha=ALPHA)


def allocate_all_frames(result, alpha=ALPHA):
    """Evaluate Eq. (12) and apply it, for every frame.

    This is the operation Table II counts as O(T . F . K_t): for each frame the
    (K_t, F) weight matrix is formed and multiplied into that frame's time-frequency
    slice, giving the mode coefficients {U_k,t(w)}.  Those coefficients are what
    STVMD's ADMM loop also produces, which is what makes the two columns comparable.
    Stopping earlier -- at the weight matrix -- would undercount; going
    further, to the inverse transform, would add an O(T . F log F) term that Table II
    does not include and that the Experiment D measurement does not perform.

    """
    tf = result.tfmap.coef
    acc = 0.0
    for t in range(result.partition.T):
        W = result.weights(t, alpha=alpha)         # (K_t, F), Eq. (12)
        U = W * tf[:, t][None, :]                  # mode coefficients for frame t
        acc += float(U.real.sum())                 # consume U so it cannot be elided
    return acc


def time_stages(x, reps):
    """(transform, detection, soft extraction) medians and IQRs, plus K_t."""
    p = build_damd()
    t_tf, d_tf = timed(lambda: p.transform.forward(x, FS), reps)
    t_pipe, d_pipe = timed(lambda: p.run(x, FS), reps)

    r = p.run(x, FS)
    t_alloc, d_alloc = timed(lambda: allocate_all_frames(r, ALPHA), reps)

    n_modes = np.asarray(r.n_modes, dtype=int)
    return dict(transform=(t_tf, d_tf),
                detect=(max(t_pipe - t_tf, 0.0), d_pipe),
                alloc=(t_alloc, d_alloc),
                pipeline=(t_pipe + t_alloc, d_pipe + d_alloc),
                n_modes=n_modes)


def time_stvmd(x, reps, k=K_STVMD):
    from softdamd.baselines import STVMD
    obj = STVMD(K=k, n_fft=N_FFT, alpha=ALPHA_STVMD,
                tau=1e-5, tol=1e-9, maxiters=500)
    return timed(lambda: obj.decompose(x, fs=FS, dynamic=True), reps)


def main(save_dir='outputs/soft_partition', lengths=LENGTHS, reps=N_REPS,
         quick=False, reuse=False, plot_only=False, skip_stvmd=False):
    setup_font_config(fontsize=9)
    matplotlib.rcParams.update({'font.size': 9})
    if quick:
        lengths, reps = lengths[:3], 3

    print(f'environment: {env_line()}')
    print(f'n_fft = {N_FFT}, hop = {HOP} (one frame per sample, the STVMD mode)')
    print(f'STVMD: K = {K_STVMD} fixed, alpha_orig = {ALPHA_STVMD:g}, maxiters = 500')
    print(f'DAMD : K_t automatic; soft-extraction alpha = {ALPHA:g} '
          f'(= alpha_orig {2 * ALPHA:g})')
    print(f'timings: median of {reps} repetitions after {N_WARMUP} warm-up, '
          f'bars are the interquartile range\n')

    cache = load_cache(save_dir) if (reuse or plot_only) else {}
    if cache:
        print(f'reusing {cache_path(save_dir)}: lengths '
              f'{sorted(cache)}\n')

    rows = []
    print(f"  {'N':>6}  {'STVMD':>16}  {'transform':>14}  {'detection':>14}"
          f"  {'extraction':>14}  {'K_t mean':>9}  {'ratio':>7}  {'source':>12}")
    for n in lengths:
        entry = dict(cache.get(n, {}))
        need_stvmd = 'stvmd' not in entry and not skip_stvmd
        need_stages = any(k not in entry for k in
                          ('transform', 'detect', 'alloc', 'n_modes'))
        if plot_only and (need_stvmd or need_stages):
            print(f'  {n:6d}  incomplete in cache, skipped (--plot-only)')
            continue

        tag = []
        if need_stvmd:
            x = make_signal(n)
            entry['stvmd'] = list(time_stvmd(x, reps))
            tag.append('stvmd')
        if need_stages:
            x = make_signal(n)
            st = time_stages(x, reps)
            entry.update(transform=list(st['transform']),
                         detect=list(st['detect']),
                         alloc=list(st['alloc']),
                         n_modes=[int(v) for v in st['n_modes']])
            tag.append('stages')
        cache[n] = entry

        if 'stvmd' not in entry:
            print(f'  {n:6d}  no STVMD timing available, skipped')
            continue

        t_st, d_st = entry['stvmd']
        nm = np.asarray(entry['n_modes'], dtype=int)
        pipe_n = entry['transform'][0] + entry['detect'][0] + entry['alloc'][0]
        ratio = t_st / max(pipe_n, 1e-12)
        rows.append(dict(n=n, stvmd=(t_st, d_st), ratio=ratio, n_modes=nm,
                         transform=entry['transform'], detect=entry['detect'],
                         alloc=entry['alloc'], pipeline=(pipe_n, 0.0)))
        src = '+'.join(tag) if tag else 'cache'
        print(f"  {n:6d}  {t_st:9.3f}s±{d_st:5.3f}  {entry['transform'][0]*1e3:9.1f}ms"
              f"  {entry['detect'][0]*1e3:11.1f}ms  {entry['alloc'][0]*1e3:12.2f}ms"
              f"  {nm.mean():9.1f}  {ratio:6.1f}x  {src:>12}")

    if not rows:
        print('\nnothing to plot')
        return None
    if not plot_only:
        save_cache(save_dir, cache,
                   dict(env=env_line(), n_fft=N_FFT, hop=HOP, K_stvmd=K_STVMD,
                        alpha_stvmd=ALPHA_STVMD, alpha=ALPHA, reps=reps))

    N = np.array([r['n'] for r in rows], float)
    stv = np.array([r['stvmd'][0] for r in rows])
    stv_e = np.array([r['stvmd'][1] for r in rows])
    tra = np.array([r['transform'][0] for r in rows])
    det = np.array([r['detect'][0] for r in rows])
    alc = np.array([r['alloc'][0] for r in rows])
    pipe = tra + det + alc

    # ── plot ────────────────────────────────────────────────────────────────
    # Sized for \textwidth of an IEEE two-column page (7.16 in) as a figure*.
    # Font sizes are set so that the rendered text matches 8 pt body copy at 1:1
    # scale; the figure must therefore not be rescaled in LaTeX.
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.75),
        gridspec_kw={'width_ratios': [1.35, 1.0], 'wspace': 0.30,
                     'left': 0.075, 'right': 0.985, 'top': 0.86, 'bottom': 0.185})

    ax = axes[0]
    ax.errorbar(N, stv, yerr=stv_e, fmt='-s', ms=4.0, lw=1.5, capsize=2.0,
                color=COL['stvmd'], zorder=6,
                label=fr'STVMD, $K={K_STVMD}$ preset')
    ax.plot(N, pipe, '-o', ms=4.0, lw=1.5, color='0.15', zorder=6,
            label='DAMD-Soft, total')
    ax.plot(N, det, '--^', ms=3.4, lw=1.1, color=COL['detect'], alpha=0.95,
            zorder=4, label='detection (Meanshift)')
    ax.plot(N, tra, ':v', ms=3.4, lw=1.1, color=COL['transform'], alpha=0.95,
            zorder=4, label='STFT')
    ax.plot(N, alc, '-.D', ms=3.4, lw=1.1, color=COL['alloc'], alpha=0.95,
            zorder=4, label='soft extraction, Eq. (12)')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'signal length $N$ (samples)', fontsize=8)
    ax.set_ylabel('wall-clock time (s)', fontsize=8)
    ax.set_xticks(list(N))
    ax.set_xticklabels([str(int(v)) for v in N], fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='x', which='minor', bottom=False)
    lo = min(float(alc.min()), float(tra.min()))
    ax.set_ylim(lo * 0.35, stv.max() * 25.0)
    ax.grid(True, which='major', alpha=0.30, lw=0.5)
    ax.grid(True, which='minor', axis='y', alpha=0.15, lw=0.4)
    # Every curve rises to the right, so headroom is added above the topmost one and
    # the legend is placed there in two columns: it then sits over empty axes rather
    # than over data at any N.
    ax.legend(fontsize=6.6, loc='upper left', framealpha=0.95, ncol=2,
              handlelength=1.9, borderpad=0.35, labelspacing=0.28,
              columnspacing=1.0, borderaxespad=0.4)
    ax.set_title('(a) cost by stage', fontsize=8.5, pad=4)

    ax2 = axes[1]
    nm = rows[-1]['n_modes']
    edges = np.arange(nm.min() - 0.5, nm.max() + 1.5, 1.0)
    ax2.hist(nm, bins=edges, color=COL['detect'], alpha=0.85,
             edgecolor='w', lw=0.6)
    ax2.axvline(K_STVMD, color=COL['stvmd'], lw=1.8, zorder=5,
                label=fr'STVMD: $K={K_STVMD}$, every frame')
    ax2.axvline(nm.mean(), color='0.15', ls='--', lw=1.3, zorder=5,
                label=fr'DAMD: $\overline{{K_t}}={nm.mean():.1f}$, '
                      fr'range $[{nm.min()},{nm.max()}]$')
    ax2.set_xlabel(r'modes returned in a frame, $K_t$', fontsize=8)
    ax2.set_ylabel('frames', fontsize=8)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.set_xticks(np.arange(nm.min(), max(nm.max(), K_STVMD) + 1))
    ax2.margins(x=0.06)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.30)
    ax2.grid(True, axis='y', alpha=0.30, lw=0.5)
    ax2.legend(fontsize=6.6, loc='upper left', framealpha=0.95,
               handlelength=1.6, borderpad=0.35, labelspacing=0.30,
               borderaxespad=0.4)
    ax2.set_title(fr'(b) modes returned, $N={int(N[-1])}$', fontsize=8.5, pad=4)

    out = save_figure(fig, 'runtime', save_dir)

    # ── console summary ─────────────────────────────────────────────────────
    print(f"\n  soft extraction as a share of the DAMD-Soft pipeline:")
    for r, a, p_ in zip(rows, alc, pipe):
        print(f"    N = {r['n']:5d}:  {100 * a / p_:9.4g}%   "
              f"({a * 1e3:8.3f} ms of {p_ * 1e3:9.1f} ms)")
    slope_s = np.polyfit(np.log(N), np.log(stv), 1)[0]
    slope_d = np.polyfit(np.log(N), np.log(pipe), 1)[0]
    print(f"\n  empirical scaling  d log t / d log N:"
          f"  STVMD {slope_s:.2f},  DAMD-Soft {slope_d:.2f}")
    print('  Both are linear in the number of frames T = N at hop = 1; the gap is the')
    print('  constant K.N_A of the ADMM loop, not a different exponent.')
    nm = rows[-1]['n_modes']
    print(f"\n  at N = {int(N[-1])}: STVMD returns K = {K_STVMD} in every frame (preset);")
    print(f"  DAMD returns K_t in [{nm.min()}, {nm.max()}], mean {nm.mean():.1f}, "
          f"median {int(np.median(nm))}.")
    print('  The ratio in the table is therefore a pipeline comparison between methods')
    print('  with different mode-selection rules, not a speedup for an identical')
    print('  decomposition, and should be reported as such.')
    return out


def reconstruction_check(n=512):
    """The two reconstruction errors quoted in Section VII-B and Experiment S5.

    Soft allocation: every frame's Eq. (12) weights are applied to that frame's STFT
    slice, the modes are summed and the sum is inverted with the same transform, so the
    error is the round-trip error of the STFT (sum_k s_k = 1 pointwise).
    STVMD: the time-domain modes it returns are summed and compared with the signal.
    Both use the signal of this experiment at length n and the settings timed above.
    """
    from softdamd.baselines import STVMD
    x = make_signal(n)
    r = build_damd().run(x, FS)
    x_soft = r.reconstruct('soft', ALPHA)                   # inverse STFT of sum_k U_k
    err_soft = np.linalg.norm(x_soft - x) / np.linalg.norm(x)

    res = STVMD(K=K_STVMD, n_fft=N_FFT, alpha=ALPHA_STVMD,
                tau=1e-5, tol=1e-9, maxiters=500).decompose(
                    x, fs=FS, dynamic=True)
    x_stvmd = np.asarray(res.modes).sum(axis=0)[:n]
    err_stvmd = np.linalg.norm(x_stvmd - x) / np.linalg.norm(x)

    print(f'\nReconstruction at N = {n} (relative L2 error of the summed modes):')
    print(f'  soft allocation, Eq. (12): {err_soft:.2e}   (STFT round trip)')
    print(f'  STVMD, K = {K_STVMD}, {res.n_iterations} iterations: {err_stvmd:.2e}')
    return err_soft, err_stvmd


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default='.')
    ap.add_argument('--lengths', type=int, nargs='*', default=None)
    ap.add_argument('--reps', type=int, default=N_REPS)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--reuse', action='store_true',
                    help='read runtime_timings.json and recompute only what is missing')
    ap.add_argument('--plot-only', dest='plot_only', action='store_true',
                    help='redraw from runtime_timings.json without timing anything')
    ap.add_argument('--skip-stvmd', dest='skip_stvmd', action='store_true',
                    help='never run STVMD; use the cached value or drop the length')
    ap.add_argument('--recon', action='store_true',
                    help='print only the two reconstruction errors of Experiment S5')
    args = ap.parse_args()
    if args.recon:
        reconstruction_check()
        raise SystemExit
    main(save_dir=args.save_dir,
         lengths=tuple(args.lengths) if args.lengths else LENGTHS,
         reps=args.reps, quick=args.quick, reuse=args.reuse,
         plot_only=args.plot_only, skip_stvmd=args.skip_stvmd)
