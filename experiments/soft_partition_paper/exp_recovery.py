# experiments/soft_partition_paper/exp_recovery.py
"""
Experiment S4 -- time-domain mode recovery with known ground truth.

Purpose
-------
Experiments C and C' compare the extraction rules in a frequency-domain Gaussian model.
This experiment runs the complete pipeline (STFT, density-based detection, extraction,
inverse STFT) on time-domain signals whose components are known, and scores the
recovered waveforms against them.

Signals (fs = 1000 Hz, N = 1024), x = u1 + u2 + u3 + white Gaussian noise,
SNR = total signal power / noise power.  Two classes, same spacing of u1 and u2:
  'tones'      : deterministic, random phases per trial
                 u1 = cos(2 pi 60 t + phi1), u2 = 0.8 cos(2 pi 110 t + phi2),
                 u3 = 0.6 cos(2 pi (200 t + 50 t^2) + phi3)  (chirp, 200 -> 302 Hz).
                 Line components: far narrower than the alpha = 2000 prior.
  'lorentzian' : the model of Definition V.1 in the time domain -- stationary Gaussian
                 modes with Lorentzian spectra centred at 60, 110 and 250 Hz and
                 half-width beta = fs/sqrt(2 alpha) = 15.8 Hz, i.e. matching alpha = 2000;
                 each realisation scaled to the RMS of the corresponding tone
                 (1, 0.8, 0.6)/sqrt(2).  This is the control that separates the effect
                 of the mode model from that of the time-domain pipeline.

Methods (no parameter is tuned for this experiment; all are the paper's settings)
  DAMD pipeline of Experiment S5 (STFT, n_fft = 128, hop = 1), detection run once per
  trial; the SAME detected centres are then used by
    soft  : Lorentzian weights, Eq. (12), alpha = 2000
    hard  : Voronoi mask, Eq. (7), ties shared equally
  and the masked STFT is inverted exactly (unit-hop STFT, round-trip error ~1e-16).
  VMD   : softdamd.baselines.VMD on the whole signal with the true K = 3, the same
          alpha (the library's frequency axis is 2f/fs, so alpha_lib = alpha/2 = 1000),
          tau = 1e-5 (library default), default initialisation and tolerance.

Association to ground truth (as stated in the paper)
  DAMD returns K_t unlabelled modes per frame.  Each detected mode at frame t is
  assigned to the component whose true instantaneous frequency at t is nearest, and the
  masks of the modes assigned to a component are summed.  This uses the true
  instantaneous frequencies, and it is the same for soft and hard.  VMD modes are
  matched to components by the assignment minimising the distance between the VMD
  centre frequency and the component's mean frequency.  Both methods therefore receive
  oracle information: DAMD the association, VMD the number of modes.

Metric
  relative waveform error ||u_hat_c - u_c|| / ||u_c|| per component, over 20 trials per
  SNR; soft vs hard is compared per trial (paired, same detection).
"""
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp_g as g                                                       # noqa: E402
from softdamd import voronoi_weights                                    # noqa: E402
from softdamd.baselines import VMD                                      # noqa: E402

FS, N = 1000.0, 1024
ALPHA = 2000.0
SNRS_DB = (0, 10, 20)
N_TRIALS = 20
SEED = 42
AMPS = np.array([1.0, 0.8, 0.6])
NAMES = ('60 Hz tone', '110 Hz tone', 'chirp')


LOR_CENTRES = (60.0, 110.0, 250.0)
BETA_HZ = FS / np.sqrt(2.0 * ALPHA)                           # 15.81 Hz


def components(kind, rng):
    t = np.arange(N) / FS
    if kind == 'tones':
        ph = rng.uniform(0, 2 * np.pi, 3)
        return np.stack([AMPS[0] * np.cos(2 * np.pi * 60 * t + ph[0]),
                         AMPS[1] * np.cos(2 * np.pi * 110 * t + ph[1]),
                         AMPS[2] * np.cos(2 * np.pi * (200 * t + 50 * t ** 2) + ph[2])])
    f = np.fft.rfftfreq(N, 1.0 / FS)
    out = []
    for fc, a in zip(LOR_CENTRES, AMPS):
        L = 1.0 / (1.0 + ((f - fc) / BETA_HZ) ** 2)
        C = np.sqrt(L / 2.0) * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
        u = np.fft.irfft(C, n=N)
        out.append(u * (a / np.sqrt(2.0)) / np.sqrt(np.mean(u ** 2)))
    return np.stack(out)


def true_if_norm(kind, n_frames):
    """True instantaneous (centre) frequencies, normalised f/fs, at each STFT frame.
    With hop 1 and centred frames, frame j is centred on sample j (checked by sanity run)."""
    if kind == 'lorentzian':
        return np.stack([np.full(n_frames, fc) for fc in LOR_CENTRES]) / FS
    tj = np.minimum(np.arange(n_frames), N - 1) / FS
    return np.stack([np.full(n_frames, 60.0), np.full(n_frames, 110.0),
                     200.0 + 100.0 * tj]) / FS


def damd_recover(x, stats, kind):
    res = g.build_damd().run(x, FS)
    p = res.partition
    Wf = res.tfmap.coef                                        # (F, T)
    freqs = np.asarray(res.freqs, dtype=float)
    nm = np.asarray(p.n_modes)
    off = p.offsets
    cen_all = np.asarray(p.centers, dtype=float)
    band_all = np.asarray(p.band)
    F, T = Wf.shape
    ift = true_if_norm(kind, T)
    M_soft = np.zeros((3, F, T))
    M_hard = np.zeros((3, F, T))
    for t in range(T):
        sl = slice(off[t], off[t + 1])
        c = cen_all[sl]
        S = res.weights(t, alpha=ALPHA)                          # Eq. (12)
        H = voronoi_weights(freqs, c)                            # Eq. (7), ties shared
        # check: library hard bands vs the Voronoi mask of the same centres
        lib = np.zeros_like(H)
        for k, (b0, b1) in enumerate(band_all[sl]):
            lib[k, int(b0):int(b1)] = 1.0
        stats['bins'] += F
        stats['band_mismatch'] += int(np.sum(np.any(np.abs(lib - H) > 1e-12, axis=0)))
        assign = np.argmin(np.abs(c[:, None] - ift[:, t][None, :]), axis=1)
        # detections other than the nearest one to each true component, and how many of
        # them lie above the highest true component
        keep = {int(np.argmin(np.abs(c - ift[j, t]))) for j in range(3)}
        extra = [k for k in range(len(c)) if k not in keep]
        stats['extra'][stats['snr']] += len(extra)
        stats['extra_above'][stats['snr']] += sum(int(c[k] > ift[:, t].max()) for k in extra)
        for k, cid in enumerate(assign):
            M_soft[cid, :, t] += S[k]
            M_hard[cid, :, t] += H[k]
        stats['kt'].append(len(c))
    rec = {}
    for name, M in (('soft', M_soft), ('hard', M_hard)):
        rec[name] = np.stack([res.tfmap.inverse(M[c] * Wf)[:N] for c in range(3)])
    return rec


def vmd_recover(x, kind):
    r = VMD(K=3, alpha=ALPHA / 2.0)(x, fs=FS)                  # library axis is 2f/fs
    modes = np.asarray(r.modes)[:, :N]
    f_mode = np.ravel(r.center_freqs) * FS / 2.0               # Hz
    f_true = (np.array(LOR_CENTRES) if kind == 'lorentzian'
              else np.array([60.0, 110.0, 251.15]))            # component mean frequency
    rows, cols = linear_sum_assignment(np.abs(f_true[:, None] - f_mode[None, :]))
    out = np.zeros((3, N))
    out[rows] = modes[cols]
    return out, f_mode, x - modes.sum(axis=0)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def run(kind):
    rng = np.random.default_rng(SEED)
    stats = dict(bins=0, band_mismatch=0, kt=[], snr=0,
                 extra=np.zeros(len(SNRS_DB), int), extra_above=np.zeros(len(SNRS_DB), int))
    err = {m: np.zeros((len(SNRS_DB), N_TRIALS, 3)) for m in ('soft', 'hard', 'vmd')}
    vmd_freqs = []
    vmd_res = np.zeros((len(SNRS_DB), N_TRIALS, 2))   # ||x - sum modes||/||x||, corr with noise
    for si, snr in enumerate(SNRS_DB):
        stats['snr'] = si
        for i in range(N_TRIALS):
            u = components(kind, rng)
            s = u.sum(axis=0)
            noise = rng.standard_normal(N)
            noise *= np.sqrt(np.mean(s ** 2) / 10 ** (snr / 10) / np.mean(noise ** 2))
            x = s + noise
            rec = damd_recover(x, stats, kind)
            for m in ('soft', 'hard'):
                if np.linalg.norm(rec[m].sum(0) - x) > 1e-10 * np.linalg.norm(x):
                    raise RuntimeError(f'{m} estimates do not sum to the signal')
            v, fm, r = vmd_recover(x, kind)
            vmd_freqs.append(np.sort(fm))
            vmd_res[si, i] = (np.linalg.norm(r) / np.linalg.norm(x), np.corrcoef(r, noise)[0, 1])
            for c in range(3):
                err['soft'][si, i, c] = rel(rec['soft'][c], u[c])
                err['hard'][si, i, c] = rel(rec['hard'][c], u[c])
                err['vmd'][si, i, c] = rel(v[c], u[c])
    return err, stats, np.array(vmd_freqs), vmd_res


def report(kind, err, stats, vf, vres):
    names = NAMES if kind == 'tones' else tuple(f'{f:g} Hz Lor.' for f in LOR_CENTRES)
    kt = np.array(stats['kt'])
    print(f"\n=== signal class '{kind}' ===")
    print(f'detected modes per frame: mean {kt.mean():.2f}, range [{kt.min()}, {kt.max()}]')
    print('detections beyond the nearest one per component, and the share above the highest '
          'component: ' + ', '.join(f'{snr} dB: {stats["extra"][si]} ({stats["extra_above"][si] / max(stats["extra"][si], 1):.2f})'
                                   for si, snr in enumerate(SNRS_DB)))
    print(f'library hard bands differ from the Voronoi mask in '
          f'{stats["band_mismatch"]} of {stats["bins"]} (bin, frame) cells')
    print(f'VMD centre frequencies (Hz), median over trials: {np.round(np.median(vf, 0), 1)}')
    print('VMD modes do not sum to x (tau = 1e-5); the soft and hard estimates do, exactly:')
    for si, snr in enumerate(SNRS_DB):
        print(f'  {snr:3d} dB: ||x - sum of VMD modes|| / ||x|| = {vres[si, :, 0].mean():.3f}, '
              f'correlation of that residual with the noise = {vres[si, :, 1].mean():.3f}')
    print('relative waveform error, mean +- standard error over trials')
    print(f"{'SNR':>4} {'component':>14} {'soft':>16} {'hard':>16} {'VMD (K=3)':>16}"
          f" {'soft<hard':>10} {'hard-soft (paired)':>22}")
    for si, snr in enumerate(SNRS_DB):
        for c in range(3):
            cells = []
            for m in ('soft', 'hard', 'vmd'):
                e = err[m][si, :, c]
                cells.append(f'{e.mean():.4f}+-{e.std(ddof=1) / np.sqrt(N_TRIALS):.4f}')
            d = err['hard'][si, :, c] - err['soft'][si, :, c]
            wins = int(np.sum(d > 0))
            print(f'{snr:4d} {names[c]:>14} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}'
                  f' {wins:>5}/{N_TRIALS:<4} {d.mean():+.4f}+-{d.std(ddof=1) / np.sqrt(N_TRIALS):.4f}')


def main():
    print(f'fs = {FS:g} Hz, N = {N}, n_fft = {g.N_FFT}, hop = {g.HOP}, alpha = {ALPHA:g}, '
          f'{N_TRIALS} trials per SNR, seed {SEED}')
    for kind in ('tones', 'lorentzian'):
        report(kind, *run(kind))


if __name__ == '__main__':
    main()
