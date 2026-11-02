# docs/make_readme_figures.py
"""
Regenerates every image used by README.md, from the package itself.

    python docs/make_readme_figures.py            # writes into docs/
    python docs/make_readme_figures.py --out DIR

Nothing here is copied from the paper's figure files: each panel is computed with the
public API (``softdamd``) or with the experiment scripts of this repository, so what the
README shows is what the shipped code does.  Where a figure repeats an experiment of the
paper it says so, and with how many trials.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EXP = os.path.join(_ROOT, 'experiments')
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402

import softdamd as sd                  # noqa: E402

DPI = 170
STYLE = {
    'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.grid': True, 'grid.alpha': 0.25, 'figure.facecolor': 'white',
}
HARD = dict(color='0.15', ls=':', lw=2.0)
INV = dict(color='0.65', lw=4.0, alpha=0.9)
ALPHAS = [(50, '#d62839'), (200, '#f28c28'), (2000, '#2b5fd9'), (1e5, '#2a9d4b')]


def _save(fig, out_dir, name, max_width=None):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    if max_width is not None:
        _shrink(path, max_width)
    print(f'  wrote {path}')


def _shrink(path, max_width):
    from PIL import Image
    im = Image.open(path)
    if im.size[0] > max_width:
        im.thumbnail((max_width, 10 ** 6), Image.LANCZOS)
    im.save(path, optimize=True)


# ---------------------------------------------------------------------------------
# 1. the object: the Lorentzian mask family against the hard Voronoi mask
# ---------------------------------------------------------------------------------
def fig_mask_family(out_dir):
    fs = 1000.0
    freqs = np.linspace(0.0, 0.5, 4001)
    centres = np.array([80.0, 200.0]) / fs
    hz = freqs * fs

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(hz, sd.voronoi_weights(freqs, centres)[0], label='hard Voronoi mask, Eq. (7)',
            **HARD)
    d1, d2 = (freqs - centres[0]) ** 2, (freqs - centres[1]) ** 2
    ax.plot(hz, d2 / np.maximum(d1 + d2, 1e-300), label='inverse-square, Eq. (5): α → ∞',
            **INV)
    for a, c in ALPHAS:
        ax.plot(hz, sd.lorentzian_weights(freqs, centres, a)[0], color=c, lw=1.8,
                label=f'Lorentzian mask, Eq. (12): α = {a:g}')
    for c in centres:
        ax.axvline(c * fs, color='0.8', lw=0.8, ls='--', zorder=0)
    ax.set(xlim=(0, 500), ylim=(-0.03, 1.08), xlabel='frequency (Hz)',
           ylabel='weight of the mode at 80 Hz',
           title='Two modes at 80 and 200 Hz: what each rule gives the first one')
    ax.legend(loc='upper right', framealpha=0.95)
    _save(fig, out_dir, 'mask_family.png')


# ---------------------------------------------------------------------------------
# 2. the pipeline on one signal: spectrogram, one frame, its modes, the difference
# ---------------------------------------------------------------------------------
def fig_pipeline_example(out_dir):
    from common.runner import run_damd
    from soft_partition_paper.synthetic_suite import crossing_chirps

    x, fs, tf = crossing_chirps()
    alpha = 2000.0
    res, _ = run_damd(x, fs, transform=tf, n_fft=max(int(fs) // 2, 64),
                      bandwidth='percentile', adapt='curvature', alpha=alpha)
    P = np.abs(res.tfmap.coef) ** 2                       # (F, T)
    F, T = P.shape
    hz = res.tfmap.freqs_hz
    times = np.arange(T) * res.tfmap.hop / fs
    freqs = np.asarray(res.freqs)

    # a frame with three modes shortly before the crossing
    k_t = np.asarray(res.n_modes)
    cand = np.where((times > 0.45) & (times < 0.7) & (k_t == 3))[0]
    t0 = int(cand[len(cand) // 2]) if cand.size else T // 5

    # allocation difference per frame (the Table IV quantity, all bins)
    diff = np.zeros(T)
    for t in range(T):
        tot = P[:, t].sum()
        if tot <= 0 or k_t[t] < 2:
            continue
        s_soft = res.weights(t, mode='soft')
        s_hard = res.weights(t, mode='hard')
        diff[t] = 0.5 * (np.abs(s_soft - s_hard).sum(axis=0) * P[:, t]).sum() / tot

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2))
    (a, b), (c, d) = axs

    # (a) spectrogram with detected centres
    a.imshow(np.log10(P + 1e-12), aspect='auto', origin='lower', cmap='magma',
             extent=[times[0], times[-1], hz[0], hz[-1]], vmin=np.log10(P.max()) - 5)
    for t in range(0, T, 3):                      # marker area ∝ soft energy share
        cs = np.asarray(res.partition.centers_at(t)) * fs
        e = res.energy(t, mode='soft')
        share = e / max(e.sum(), 1e-300)
        a.scatter(np.full(cs.size, times[t]), cs, s=2 + 40 * share, color='#5ee6ff',
                  alpha=0.9, linewidths=0)
    a.axvline(times[t0], color='white', lw=1.0, ls='--')
    a.grid(False)
    a.set(xlabel='time (s)', ylabel='frequency (Hz)',
          title=f'(a) |T(ω,t)|² and the detected centres, marker ∝ energy  (K_t = {k_t.min()}…{k_t.max()})')

    # (b) one frame: input power, the two allocations
    s_soft = res.weights(t0, mode='soft')
    s_hard = res.weights(t0, mode='hard')
    Pt = P[:, t0] / P[:, t0].max()
    b.fill_between(hz, 0, Pt, color='0.85', label='|T(ω,t)|² (scaled)')
    cols = plt.cm.tab10(np.arange(s_soft.shape[0]))
    for k in range(s_soft.shape[0]):
        b.plot(hz, s_hard[k], color=cols[k], ls=':', lw=1.6)
        b.plot(hz, s_soft[k], color=cols[k], lw=2.0, label=f'mode {k + 1}')
    b.set(xlim=(0, hz[-1]), ylim=(-0.03, 1.08), xlabel='frequency (Hz)', ylabel='weight',
          title=f'(b) frame at t = {times[t0]:.2f} s: s_k(ω) for both rules, α = {alpha:g}')
    b.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.17), frameon=False,
             title='soft mask solid, hard mask dotted', title_fontsize=8.5)

    # (c) the extracted mode spectra at that frame
    U_soft = np.abs(res.modes(t0, mode='soft'))
    U_hard = np.abs(res.modes(t0, mode='hard'))
    for k in range(U_soft.shape[0]):
        c.plot(hz, U_hard[k], color=cols[k], ls=':', lw=1.4)
        c.plot(hz, U_soft[k], color=cols[k], lw=1.8, label=f'|U_{k + 1}(ω)|')
    c.plot(hz, np.abs(res.tfmap.coef[:, t0]), color='0.3', lw=1.0, alpha=0.6, label='|T(ω,t)| = |Σ_k U_k|')
    c.set(xlim=(0, hz[-1]), xlabel='frequency (Hz)', ylabel='magnitude',
          title='(c) the modes it returns: soft (solid), hard (dotted); both sum to T')
    c.legend(loc='upper right')

    # (d) how much the two allocations differ, frame by frame
    d.fill_between(times, 0, 100 * diff, color='#2b5fd9', alpha=0.35)
    d.plot(times, 100 * diff, color='#2b5fd9', lw=1.2)
    d.axvline(times[t0], color='0.3', lw=1.0, ls='--')
    d.set(xlabel='time (s)', ylabel='allocation difference (% of frame energy)',
          title='(d) ½ Σ_k |s_k^soft − s_k^hard| |T|² per frame: largest where modes meet')
    d2 = d.twinx()
    d2.step(times, k_t, where='mid', color='0.4', lw=1.0)
    d2.set_ylabel('K_t (grey)', color='0.4')
    d2.set_ylim(0, k_t.max() + 2)
    d2.grid(False)

    fig.suptitle('Crossing chirps (synthetic_suite.py), SSQ-STFT, fs = 220 Hz', y=0.995)
    fig.tight_layout()
    _save(fig, out_dir, 'pipeline_example.png', max_width=1400)


# ---------------------------------------------------------------------------------
# 3. accuracy against SNR under the Lorentzian model (Experiment C, fewer trials)
# ---------------------------------------------------------------------------------
def fig_accuracy_vs_snr(out_dir, n_trials=500, seed=42):
    F = 512
    w1, w2, alpha_true, sigma2 = 0.12, 0.30, 5000.0, 1.0
    snr_db = np.arange(-10, 45, 5)
    rng = np.random.default_rng(seed)
    freqs = np.linspace(0.0, 0.5, F)
    S1 = sigma2 / (1 + 2 * alpha_true * (freqs - w1) ** 2)
    S2 = sigma2 / (1 + 2 * alpha_true * (freqs - w2) ** 2)
    centres = np.array([w1, w2])
    W_soft = sd.lorentzian_weights(freqs, centres, alpha_true)[0]
    W_hard = sd.voronoi_weights(freqs, centres)[0]
    d1, d2 = (freqs - w1) ** 2, (freqs - w2) ** 2
    W_inv = d2 / np.maximum(d1 + d2, 1e-300)

    rows = {k: [] for k in ('hard', 'soft', 'shrunk', 'inv', 'oracle')}
    for s in snr_db:
        sn2 = sigma2 / 10 ** (s / 10)
        SX = S1 + S2 + sn2
        W_or = S1 / SX
        W_sh = W_soft * (S1 + S2) / SX
        g = lambda S: np.sqrt(S / 2) * (rng.standard_normal((n_trials, F))
                                        + 1j * rng.standard_normal((n_trials, F)))
        U1, U2, N = g(S1), g(S2), g(sn2 * np.ones(F))
        X = U1 + U2 + N
        den = (np.abs(U1) ** 2).sum(axis=1).mean()
        for key, W in (('hard', W_hard), ('soft', W_soft), ('shrunk', W_sh),
                       ('inv', W_inv), ('oracle', W_or)):
            rows[key].append((np.abs(W * X - U1) ** 2).sum(axis=1).mean() / den)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    spec = [('hard', 'hard Voronoi mask', dict(color='0.15', ls=':', lw=2.2, marker='s')),
            ('inv', 'inverse-square (VMD fixed point)', dict(color='0.6', lw=3.0, marker='D', ms=4)),
            ('soft', 'Lorentzian mask, true α, Eq. (12)', dict(color='#2b5fd9', lw=2.0, marker='o')),
            ('oracle', 'oracle MMSE  S₁/S_X  (= soft × shrinkage factor (22))',
             dict(color='#2a9d4b', lw=1.2, ls='--', marker='x'))]
    for key, lab, kw in spec:
        ax.semilogy(snr_db, rows[key], label=lab, ms=kw.pop('ms', 5), **kw)
    ax.set(xlabel='SNR (dB)', ylabel='nMSE of mode 1',
           title='Two Lorentzian modes at 0.12 and 0.30 (normalised), α = 5000, '
                 f'{n_trials} trials per SNR')
    ax.legend(loc='upper right')
    _save(fig, out_dir, 'accuracy_vs_snr.png')
    return {k: np.array(v) for k, v in rows.items()}, snr_db


# ---------------------------------------------------------------------------------
# 4./5. figures produced by the experiment scripts, shrunk for the README
# ---------------------------------------------------------------------------------
def fig_from_experiment(out_dir, module, key, name, max_width):
    from soft_partition_paper import figures as reg
    mod = __import__(f'soft_partition_paper.{module}', fromlist=[module])
    with tempfile.TemporaryDirectory() as tmp:
        mod.main(save_dir=tmp)
        src = os.path.join(tmp, reg.FIGURES[key].stem + '.png')
        from PIL import Image
        im = Image.open(src)
        im.thumbnail((max_width, 10 ** 6), Image.LANCZOS)
        im.save(os.path.join(out_dir, name), optimize=True)
    print(f'  wrote {os.path.join(out_dir, name)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=_HERE)
    ap.add_argument('--only', nargs='*', default=None,
                    choices=['mask', 'pipeline', 'snr', 'alpha', 'suite'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update(STYLE)
    want = set(args.only or ['mask', 'pipeline', 'snr', 'alpha', 'suite'])

    if 'mask' in want:
        fig_mask_family(args.out)
    if 'pipeline' in want:
        fig_pipeline_example(args.out)
    if 'snr' in want:
        rows, snr = fig_accuracy_vs_snr(args.out)
        i40 = int(np.where(snr == 40)[0][0])
        print('  nMSE at 40 dB  hard %.4f  inverse-square %.4f  soft %.4f  oracle %.4f'
              % (rows['hard'][i40], rows['inv'][i40], rows['soft'][i40], rows['oracle'][i40]))
        i0 = int(np.where(snr == 0)[0][0])
        print('  nMSE at  0 dB  soft %.3f  soft×shrinkage %.3f  oracle %.3f'
              % (rows['soft'][i0], rows['shrunk'][i0], rows['oracle'][i0]))
    if 'alpha' in want:
        fig_from_experiment(args.out, 'exp_e', 'alpha_estimation', 'alpha_estimation.png', 1300)
    if 'suite' in want:
        fig_from_experiment(args.out, 'exp_d_synthetic', 'synthetic_suite', 'synthetic_suite.png', 1500)


if __name__ == '__main__':
    main()
