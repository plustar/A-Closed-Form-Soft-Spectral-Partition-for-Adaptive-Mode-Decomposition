"""
Experiment C': centre separation sweep (panel (c) of fig_robustness).

Measures nMSE of soft vs hard extraction as the distance between two
Lorentzian modes decreases from 9 FWHM (well-separated) to 1.5 FWHM
(heavily overlapping).  Centres are exact (known), so the only variable
is the spectral overlap controlled by the separation.

Uses the same Monte Carlo engine and signal model as exp_robustness.py.

Output:
  - Console table: nMSE(hard), nMSE(soft), nMSE(oracle) at each
    (separation, SNR) pair
  - .npz file with all numerical results for plotting
"""

import os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Parameters (must match exp_robustness.py) ───────────────────
F          = 512
OMEGA1     = 0.12
ALPHA_TRUE = 5000.0
SIGMA2     = 1.0
SNR_DB     = np.arange(-10, 45, 5)
N_TRIALS   = 2000
SEED       = 42

BETA = 1.0 / np.sqrt(2.0 * ALPHA_TRUE)   # ≈ 0.01
FWHM = 2.0 * BETA                         # ≈ 0.02

# Separation sweep: in units of FWHM
SEP_FWHM = np.array([1.5, 2.0, 3.0, 4.0, 6.0, 9.0])


# ── Spectral shape ──────────────────────────────────────────────
def psd_lorentzian(freqs, omega_k, sigma2=SIGMA2):
    return sigma2 / (1.0 + 2.0 * ALPHA_TRUE * (freqs - omega_k)**2)


# ── Masks ───────────────────────────────────────────────────────
def soft_mask(freqs, centres, alpha):
    w1 = 1.0 / (1.0 + 2.0 * alpha * (freqs - centres[0])**2)
    w2 = 1.0 / (1.0 + 2.0 * alpha * (freqs - centres[1])**2)
    return w1 / (w1 + w2)


def hard_mask(freqs, centres):
    d1 = np.abs(freqs - centres[0])
    d2 = np.abs(freqs - centres[1])
    mask = np.where(d1 < d2, 1.0, 0.0)
    mask = np.where(np.abs(d1 - d2) < 1e-12, 0.5, mask)
    return mask


# ── Monte Carlo ─────────────────────────────────────────────────
def run_mc(centres, alpha_soft):
    rng = np.random.default_rng(SEED)
    freqs = np.linspace(0.0, 0.5, F)

    S1 = psd_lorentzian(freqs, centres[0])
    S2 = psd_lorentzian(freqs, centres[1])
    S1 = np.maximum(S1, 1e-30)
    S2 = np.maximum(S2, 1e-30)

    W1_soft = soft_mask(freqs, centres, alpha_soft)
    W1_hard = hard_mask(freqs, centres)
    # VMD fixed-point allocation, Eq. (5): inverse-square weights at the same centres,
    # which is what the ADMM solver converges to once the multiplier enforces
    # reconstruction (Experiment A).  Included as a baseline; evaluated on
    # the same draws, so the random sequence and the other columns are unchanged.
    D1 = (freqs - centres[0])**2
    D2 = (freqs - centres[1])**2
    W1_inv = D2 / np.maximum(D1 + D2, 1e-300)

    nmse_oracle = np.zeros(len(SNR_DB))
    nmse_soft   = np.zeros(len(SNR_DB))
    nmse_hard   = np.zeros(len(SNR_DB))
    nmse_inv    = np.zeros(len(SNR_DB))

    for si, snr_db in enumerate(SNR_DB):
        sigma_n2 = SIGMA2 / 10.0**(snr_db / 10.0)
        S_N = sigma_n2 * np.ones(F)
        S_X = S1 + S2 + S_N

        W1_oracle = S1 / S_X

        std1  = np.sqrt(S1 / 2.0)
        std2  = np.sqrt(S2 / 2.0)
        std_n = np.sqrt(S_N / 2.0)

        U1 = std1 * (rng.standard_normal((N_TRIALS, F))
                     + 1j * rng.standard_normal((N_TRIALS, F)))
        U2 = std2 * (rng.standard_normal((N_TRIALS, F))
                     + 1j * rng.standard_normal((N_TRIALS, F)))
        Nz = std_n * (rng.standard_normal((N_TRIALS, F))
                      + 1j * rng.standard_normal((N_TRIALS, F)))
        X = U1 + U2 + Nz

        denom = float(np.mean(np.abs(U1)**2))
        if denom < 1e-30:
            nmse_oracle[si] = nmse_soft[si] = nmse_hard[si] = np.nan
            continue

        nmse_oracle[si] = float(np.mean(np.abs(W1_oracle * X - U1)**2)) / denom
        nmse_soft[si]   = float(np.mean(np.abs(W1_soft * X - U1)**2)) / denom
        nmse_hard[si]   = float(np.mean(np.abs(W1_hard * X - U1)**2)) / denom
        nmse_inv[si]    = float(np.mean(np.abs(W1_inv * X - U1)**2)) / denom

    return dict(oracle=nmse_oracle, soft=nmse_soft, hard=nmse_hard, inv=nmse_inv)


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("Centre separation sweep")
    print(f"FWHM = {FWHM:.4f},  alpha = {ALPHA_TRUE:.0f},  "
          f"N_trials = {N_TRIALS},  F = {F}")
    print("=" * 75)

    results = {}

    for sep_f in SEP_FWHM:
        sep = sep_f * FWHM
        omega2 = OMEGA1 + sep
        centres = np.array([OMEGA1, omega2])

        print(f"\n--- Separation = {sep_f:.1f} FWHM  "
              f"(omega1={OMEGA1:.4f}, omega2={omega2:.4f}) ---")

        res = run_mc(centres, ALPHA_TRUE)
        results[sep_f] = res

        print(f"  {'SNR':>5}  {'Hard':>10}  {'Soft':>10}  {'Oracle':>10}  "
              f"{'Soft/Hard':>10}  {'Soft<Hard?':>10}")
        for snr, h, s, o in zip(SNR_DB, res['hard'], res['soft'], res['oracle']):
            ratio = s / h if h > 1e-30 else float('nan')
            better = "YES" if s < h else "no"
            print(f"  {snr:5d}  {h:10.4f}  {s:10.4f}  {o:10.4f}  "
                  f"{ratio:10.4f}  {better:>10}")

    # ── Summary table: high-SNR error floor ──
    print("\n" + "=" * 75)
    print("Summary: high-SNR error floor (40 dB)")
    print("=" * 75)
    print(f"  {'Sep/FWHM':>10}  {'Hard':>10}  {'Soft':>10}  {'Oracle':>10}  "
          f"{'Soft/Hard':>10}  {'Soft adv':>10}")
    for sep_f in SEP_FWHM:
        h = results[sep_f]['hard'][-1]   # last SNR = 40 dB
        s = results[sep_f]['soft'][-1]
        o = results[sep_f]['oracle'][-1]
        ratio = s / h if h > 1e-30 else float('nan')
        adv = (1 - ratio) * 100
        print(f"  {sep_f:10.1f}  {h:10.4f}  {s:10.4f}  {o:10.4f}  "
              f"{ratio:10.4f}  {adv:9.1f}%")

    # ── VMD fixed-point baseline (not plotted; saved separately from the figure data) ──
    print("\n" + "=" * 75)
    print("VMD fixed-point allocation (5) vs soft mask, same draws")
    print("=" * 75)
    i10 = int(np.where(SNR_DB == 10)[0][0])
    print(f"  {'Sep/FWHM':>10}  {'SNR':>4}  {'Soft':>8}  {'VMD fp':>8}  {'Hard':>8}  "
          f"{'VMD/Soft-1':>11}  {'Hard/Soft-1':>11}")
    for sep_f in SEP_FWHM:
        r = results[sep_f]
        for i in (i10, -1):
            s, v, h = r['soft'][i], r['inv'][i], r['hard'][i]
            print(f"  {sep_f:10.1f}  {int(SNR_DB[i]):4d}  {s:8.4f}  {v:8.4f}  {h:8.4f}  "
                  f"{100*(v/s-1):10.1f}%  {100*(h/s-1):10.1f}%")

    # ── Save ──
    np.savez(os.path.join(_HERE, 'separation_sweep_results.npz'),
             SNR_DB=SNR_DB,
             SEP_FWHM=SEP_FWHM,
             OMEGA1=OMEGA1,
             FWHM=FWHM,
             ALPHA_TRUE=ALPHA_TRUE,
             **{f'sep{s:.1f}_hard': results[s]['hard'] for s in SEP_FWHM},
             **{f'sep{s:.1f}_soft': results[s]['soft'] for s in SEP_FWHM},
             **{f'sep{s:.1f}_oracle': results[s]['oracle'] for s in SEP_FWHM},
    )
    print(f"\nResults saved to {os.path.join(_HERE, 'separation_sweep_results.npz')}")


if __name__ == '__main__':
    main()
