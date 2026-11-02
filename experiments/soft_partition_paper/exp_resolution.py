"""
Experiment C' (frequency resolution; supplementary material, Appendix I).

Soft (Lorentzian) versus hard (Voronoi) extraction at identical, exact centres, as
a function of the STFT frequency resolution, i.e. of the number of bins across the
mode FWHM, n = FWHM * n_fft.

Why a time-domain simulation.  Experiments C and C' draw the TF coefficients
directly and independently per bin, so the frequency resolution enters them only
through how finely the same spectra are sampled; window leakage, which is what a
coarse resolution actually does to a mode, is absent from that model.  Here the two
modes are generated as time series, analysed with a windowed STFT, and the masks are
applied to the STFT coefficients.  Because the STFT is linear, the STFT of each true
mode is the exact coefficient-domain ground truth at every resolution.

Modes.  Each mode is a complex first-order autoregression
    u_k[n] = rho * exp(i 2 pi w_k) * u_k[n-1] + e_k[n],   e_k ~ CN(0, (1-rho)^2),
whose power spectral density (1-rho)^2 / |1 - rho exp(i 2 pi (w_k - w))|^2 has unit
peak and, near the peak, the Lorentzian shape with half-width at half-maximum
beta = (1-rho) / (2 pi sqrt(rho)).  rho is chosen so that beta = 0.01 (cycles per
sample), the half-width of the Lorentzian with alpha = 5000 used in Experiment C, so
the mask's alpha is matched to the modes at every resolution.  Away from the peak the
AR(1) spectrum is periodic and not exactly Lorentzian.

Centres 0.12 and 0.30 cycles/sample (Experiment C).  White complex noise; SNR is the
ratio of peak mode PSD to noise PSD.  Periodic Hann window, hop n_fft/4, full complex
FFT (the signals are complex), distances measured circularly on [0, 1).
nMSE of mode 1 = sum |s_1 X - U_1|^2 / sum |U_1|^2 over all bins and frames.
"""
import numpy as np

SEED     = 42
N        = 8192
BURN     = 2000
CENTRES  = np.array([0.12, 0.30])
ALPHA    = 5000.0
BETA     = 1.0 / np.sqrt(2.0 * ALPHA)          # 0.01
FWHM     = 2.0 * BETA                          # 0.02
N_FFTS   = [16, 32, 64, 128, 256, 512, 1024]
SNRS_DB  = [0, 20, 40]
N_TRIALS = 40


def rho_for_hwhm(beta):
    # solve (1 - rho) / (2 pi sqrt(rho)) = beta for rho in (0, 1)
    lo, hi = 0.5, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1 - mid) / (2 * np.pi * np.sqrt(mid)) > beta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


RHO = rho_for_hwhm(BETA)


def ar1_mode(rng, w_k, n):
    a = RHO * np.exp(2j * np.pi * w_k)
    e = (1 - RHO) * (rng.standard_normal(n + BURN) + 1j * rng.standard_normal(n + BURN)) / np.sqrt(2)
    u = np.empty(n + BURN, dtype=complex)
    u[0] = e[0]
    for i in range(1, n + BURN):
        u[i] = a * u[i - 1] + e[i]
    return u[BURN:]


def stft(x, n_fft):
    hop = n_fft // 4
    win = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))   # periodic Hann
    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.fft(x[idx] * win[None, :], axis=1)                     # (T, n_fft)


def masks(n_fft):
    w = np.arange(n_fft) / n_fft
    d = np.abs(w[:, None] - CENTRES[None, :])
    d = np.minimum(d, 1.0 - d)                                           # circular
    a = 1.0 / (1.0 + 2.0 * ALPHA * d ** 2)
    soft = a / a.sum(axis=1, keepdims=True)
    hard = np.zeros_like(soft)
    dmin = d.min(axis=1, keepdims=True)
    tie = np.isclose(d, dmin, rtol=0, atol=1e-15)
    hard[tie] = 1.0
    hard /= hard.sum(axis=1, keepdims=True)                              # ties shared 1/m
    return soft[:, 0], hard[:, 0]


def main():
    rng = np.random.default_rng(SEED)
    # draw all trials once, reuse across resolutions and SNRs (paired comparison)
    modes = [(ar1_mode(rng, CENTRES[0], N), ar1_mode(rng, CENTRES[1], N)) for _ in range(N_TRIALS)]
    noise = [(rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2) for _ in range(N_TRIALS)]

    print(f'rho = {RHO:.6f}, beta = {BETA}, FWHM = {FWHM}, alpha = {ALPHA:g}, '
          f'{N_TRIALS} trials of {N} samples, seed {SEED}')
    print(f"{'n_fft':>6} {'bins/FWHM':>9} " + ' '.join(
        f"{'SNR':>4}{s:>3}: soft   hard  ratio" for s in SNRS_DB))
    results = {}
    for n_fft in N_FFTS:
        s1, h1 = masks(n_fft)
        row = []
        for snr in SNRS_DB:
            sig_n = 10 ** (-snr / 20.0)          # noise std so that peak PSD / noise PSD = SNR
            es, eh, e0 = [], [], []
            for (u1, u2), nz in zip(modes, noise):
                U1 = stft(u1, n_fft)
                X = U1 + stft(u2, n_fft) + stft(sig_n * nz, n_fft)
                es.append(np.sum(np.abs(s1[None, :] * X - U1) ** 2))
                eh.append(np.sum(np.abs(h1[None, :] * X - U1) ** 2))
                e0.append(np.sum(np.abs(U1) ** 2))
            es, eh, e0 = map(np.asarray, (es, eh, e0))
            ns, nh = es.sum() / e0.sum(), eh.sum() / e0.sum()
            # standard error of the ratio of means via per-trial ratios (paired)
            r_tr = es / eh
            row.append((ns, nh, ns / nh, r_tr.std(ddof=1) / np.sqrt(N_TRIALS)))
        results[n_fft] = row
        print(f'{n_fft:6d} {FWHM * n_fft:9.2f} ' + ' '.join(
            f'{ns:10.4f} {nh:6.4f} {r:5.3f}' for ns, nh, r, _ in row))
    print('\nlargest standard error of the per-trial soft/hard ratio: '
          f"{max(se for row in results.values() for *_, se in row):.4f}")
    return results


if __name__ == '__main__':
    main()
