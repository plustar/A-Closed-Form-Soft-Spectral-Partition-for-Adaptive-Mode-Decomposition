"""
Experiment C': Robustness of the Lorentzian soft mask to spectral model misspecification.

Covers two conditions:
  - the Monte Carlo nMSE comparison repeated under non-Lorentzian spectra
  - the mask's contribution isolated at fixed (and perturbed) centres

Signal model:
  X(w) = U1_true(w) + U2_true(w) + N(w)
  U_k ~ CN(0, S_k(w))    where S_k depends on the spectral shape
  N   ~ CN(0, sigma_N^2)

Spectral shapes tested:
  1. Lorentzian:     S_k(w) = sigma^2 / [1 + 2*alpha*(w - w_k)^2]
  2. Gaussian:       S_k(w) = sigma^2 * exp(-(w - w_k)^2 / (2*beta^2))
  3. Raised cosine:  S_k(w) = sigma^2 * 0.5*(1 + cos(pi*(w-w_k)/Delta))  for |w-w_k| < Delta, else 0
  4. Rectangular:    S_k(w) = sigma^2   for |w-w_k| < Delta, else 0

For each shape, the bandwidth parameter is matched so that the FWHM equals the
Lorentzian FWHM = 2/sqrt(2*alpha) = 2*beta.

Extractors:
  Oracle MMSE:  S_k / S_X  (requires knowing the true PSD — benchmark only)
  Soft:         Lorentzian mask s_k* with alpha matched to the FWHM
  Hard:         Voronoi indicator

Additional condition: perturbed centres (centres shifted by +/- perturbation)

Metric: nMSE of mode 1 = E[|U1_hat - U1_true|^2] / E[|U1_true|^2]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────
F          = 512
OMEGA1     = 0.12
OMEGA2     = 0.30
ALPHA_TRUE = 5000.0
SIGMA2     = 1.0
SNR_DB     = np.arange(-10, 45, 5)
N_TRIALS   = 2000
SEED       = 42

# Derived: Lorentzian FWHM (half-width at half-maximum)
BETA = 1.0 / np.sqrt(2.0 * ALPHA_TRUE)   # ≈ 0.01
FWHM = 2.0 * BETA                         # ≈ 0.02


# ── Spectral shape functions ────────────────────────────────────
def psd_lorentzian(freqs, omega_k, sigma2=SIGMA2):
    """Lorentzian PSD with global ALPHA_TRUE."""
    return sigma2 / (1.0 + 2.0 * ALPHA_TRUE * (freqs - omega_k)**2)


def psd_gaussian(freqs, omega_k, sigma2=SIGMA2):
    """Gaussian PSD, FWHM-matched to Lorentzian.
    Gaussian FWHM = 2*sqrt(2*ln2)*sigma_g, so sigma_g = FWHM/(2*sqrt(2*ln2))."""
    sigma_g = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma2 * np.exp(-0.5 * ((freqs - omega_k) / sigma_g)**2)


def psd_raised_cosine(freqs, omega_k, sigma2=SIGMA2):
    """Raised cosine PSD, FWHM-matched.
    Raised cosine: 0.5*(1+cos(pi*x/Delta)) for |x|<Delta.
    FWHM of raised cosine = 2*Delta/3 (half-max at x=Delta/3).
    Wait — let me derive this properly.
    f(x) = 0.5*(1+cos(pi*x/Delta)). f(0)=1. f(x)=0.5 when cos(pi*x/Delta)=0,
    i.e. pi*x/Delta = pi/2, so x = Delta/2.
    So FWHM = 2*(Delta/2) = Delta.
    Therefore Delta = FWHM.
    """
    Delta = FWHM
    d = np.abs(freqs - omega_k)
    S = np.where(d < Delta, sigma2 * 0.5 * (1.0 + np.cos(np.pi * d / Delta)), 0.0)
    # Ensure non-negative (numerical safety)
    return np.maximum(S, 0.0)


def psd_rectangular(freqs, omega_k, sigma2=SIGMA2):
    """Rectangular (boxcar) PSD, FWHM-matched.
    FWHM of a rectangle of half-width Delta = 2*Delta, so Delta = FWHM/2.
    Actually, for a rectangle at height sigma2 for |w-w_k| < Delta,
    the 'half maximum' is sigma2/2, but the rectangle is either sigma2 or 0,
    so the 'FWHM' concept doesn't directly apply the same way.
    
    For fair comparison: match the half-width to the Lorentzian FWHM half-width = BETA.
    i.e. the rectangle spans [omega_k - BETA, omega_k + BETA].
    """
    Delta = BETA  # half-width = Lorentzian half-width at half-max
    return np.where(np.abs(freqs - omega_k) < Delta, sigma2, 0.0)


# ── Lorentzian soft mask (the paper's proposed method) ──────────
def soft_mask(freqs, centres, alpha):
    """Lorentzian soft partition weights for mode 1."""
    w1 = 1.0 / (1.0 + 2.0 * alpha * (freqs - centres[0])**2)
    w2 = 1.0 / (1.0 + 2.0 * alpha * (freqs - centres[1])**2)
    return w1 / (w1 + w2)


def hard_mask(freqs, centres):
    """Voronoi hard partition for mode 1."""
    d1 = np.abs(freqs - centres[0])
    d2 = np.abs(freqs - centres[1])
    mask = np.where(d1 < d2, 1.0, 0.0)
    # Tie handling: equal sharing (as in the code)
    mask = np.where(np.abs(d1 - d2) < 1e-12, 0.5, mask)
    return mask


# ── Monte Carlo engine ──────────────────────────────────────────
def run_mc(psd_func, centres, alpha_for_soft, label="", mask_centres=None):
    """Run Monte Carlo for one spectral shape.

    centres       : TRUE mode centres, used to generate the data and the oracle.
    mask_centres  : centres ASSUMED by the soft and hard masks (a detector's output).
                    Defaults to `centres` (exact centres).  Passing shifted values here
                    is what simulates a centre-estimation error; shifting `centres`
                    instead moves the true modes as well and only changes their
                    separation, which is a different experiment.

    Returns dict with nMSE arrays for each extractor.
    """
    mask_centres = centres if mask_centres is None else mask_centres
    rng = np.random.default_rng(SEED)
    freqs = np.linspace(0.0, 0.5, F)
    
    # True PSDs
    S1 = psd_func(freqs, centres[0])
    S2 = psd_func(freqs, centres[1])
    
    # Check for zero-energy issues
    S1 = np.maximum(S1, 1e-30)
    S2 = np.maximum(S2, 1e-30)
    
    # Masks (fixed across SNR, depend only on centres and alpha)
    W1_soft = soft_mask(freqs, mask_centres, alpha_for_soft)
    W1_hard = hard_mask(freqs, mask_centres)
    
    nmse_oracle = np.zeros(len(SNR_DB))
    nmse_soft   = np.zeros(len(SNR_DB))
    nmse_hard   = np.zeros(len(SNR_DB))
    
    for si, snr_db in enumerate(SNR_DB):
        sigma_n2 = SIGMA2 / 10.0**(snr_db / 10.0)
        S_N = sigma_n2 * np.ones(F)
        S_X = S1 + S2 + S_N
        
        # Oracle MMSE (knows true PSD)
        W1_oracle = S1 / S_X
        
        # Generate data
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
        
        # Extract and compute nMSE
        denom = float(np.mean(np.abs(U1)**2))
        if denom < 1e-30:
            nmse_oracle[si] = nmse_soft[si] = nmse_hard[si] = np.nan
            continue
        
        nmse_oracle[si] = float(np.mean(np.abs(W1_oracle * X - U1)**2)) / denom
        nmse_soft[si]   = float(np.mean(np.abs(W1_soft * X - U1)**2)) / denom
        nmse_hard[si]   = float(np.mean(np.abs(W1_hard * X - U1)**2)) / denom
    
    return dict(oracle=nmse_oracle, soft=nmse_soft, hard=nmse_hard,
                freqs=freqs, S1=S1, S2=S2)


# ── Alpha matching ──────────────────────────────────────────────
def alpha_from_fwhm(fwhm):
    """Given a target FWHM, return the Lorentzian alpha that produces it.
    FWHM = 2/sqrt(2*alpha) => alpha = 2/FWHM^2."""
    return 2.0 / (fwhm**2)


# ── Main ────────────────────────────────────────────────────────
def main():
    freqs = np.linspace(0.0, 0.5, F)
    true_centres = np.array([OMEGA1, OMEGA2])
    
    # The Lorentzian alpha used for soft extraction
    # We use the same alpha as the Lorentzian FWHM match
    alpha_soft = ALPHA_TRUE
    
    shapes = {
        'Lorentzian':    psd_lorentzian,
        'Gaussian':      psd_gaussian,
        'Raised cosine': psd_raised_cosine,
        'Rectangular':   psd_rectangular,
    }
    
    # ── Part 1: different spectral shapes, exact centres ──
    print("="*70)
    print("PART 1: Non-Lorentzian spectra")
    print("="*70)
    
    results_exact = {}
    for name, psd_func in shapes.items():
        print(f"\n--- {name} spectrum ---")
        res = run_mc(psd_func, true_centres, alpha_soft, label=name)
        results_exact[name] = res
        
        print(f"  {'SNR':>5}  {'Hard':>10}  {'Soft':>10}  {'Oracle':>10}  {'Soft<Hard?':>10}")
        for snr, h, s, o in zip(SNR_DB, res['hard'], res['soft'], res['oracle']):
            better = "YES" if s < h else "no"
            print(f"  {snr:5d}  {h:10.4f}  {s:10.4f}  {o:10.4f}  {better:>10}")
    
    # ── Part 2: fixed centres vs perturbed centres ──
    print("\n" + "="*70)
    print("PART 2: Perturbed centres")
    print("="*70)
    
    # Use Lorentzian shape (simplest case) with different centre conditions
    perturbations = {
        'exact':       np.array([0.0, 0.0]),
        '+1 FWHM':     np.array([+FWHM, -FWHM]),      # assumed centres moved towards each other
        '+2 FWHM':     np.array([+2*FWHM, -2*FWHM]),
        'random ±1 FWHM': None,  # will use random per-trial (not possible in this framework)
    }
    
    results_perturbed = {}
    for label, shift in perturbations.items():
        if shift is None:
            continue  # skip random for now
        perturbed_centres = true_centres + shift
        print(f"\n--- Centres: {label} (omega = {perturbed_centres.round(4)}) ---")
        res = run_mc(psd_lorentzian, true_centres, alpha_soft, label=label,
                     mask_centres=perturbed_centres)
        results_perturbed[label] = res
        
        print(f"  {'SNR':>5}  {'Hard':>10}  {'Soft':>10}  {'Oracle':>10}")
        for snr, h, s, o in zip(SNR_DB, res['hard'], res['soft'], res['oracle']):
            print(f"  {snr:5d}  {h:10.4f}  {s:10.4f}  {o:10.4f}")
    
    # ── Part 3: cross-condition — all shapes × perturbed centres ──
    print("\n" + "="*70)
    print("PART 3: Non-Lorentzian × perturbed centres")
    print("="*70)
    
    # Just do +1 FWHM perturbation for each shape
    shift_1fwhm = np.array([+FWHM, -FWHM])
    perturbed_centres = true_centres + shift_1fwhm
    
    results_cross = {}
    for name, psd_func in shapes.items():
        print(f"\n--- {name}, centres perturbed +1 FWHM ---")
        res = run_mc(psd_func, true_centres, alpha_soft, label=name,
                     mask_centres=perturbed_centres)
        results_cross[name] = res
        
        print(f"  {'SNR':>5}  {'Hard':>10}  {'Soft':>10}  {'Oracle':>10}  {'Soft<Hard?':>10}")
        for snr, h, s, o in zip(SNR_DB, res['hard'], res['soft'], res['oracle']):
            better = "YES" if s < h else "no"
            print(f"  {snr:5d}  {h:10.4f}  {s:10.4f}  {o:10.4f}  {better:>10}")

    # ── Part 4: alpha of the mask mismatched to the true bandwidth ──
    # Lorentzian modes, exact centres; the soft mask uses alpha_soft * scale.
    # Reported in the text of Experiment C' at 40 dB, at the default separation
    # (0.18 = 9 FWHM) and at a separation of 2 FWHM.
    print("\n" + "="*70)
    print("PART 4: alpha mismatch, 40 dB")
    print("="*70)
    for sep_label, cen in (('9 FWHM (default)', true_centres),
                           ('2 FWHM', np.array([OMEGA1, OMEGA1 + 2.0 * FWHM]))):
        print(f"\n--- separation {sep_label} ---")
        print(f"  {'alpha scale':>12}  {'Soft':>8}  {'Hard':>8}")
        for scale in (1.0, 0.25, 4.0, 16.0, 1.0 / 16.0):
            r = run_mc(psd_lorentzian, cen, alpha_soft * scale, label=f'x{scale:g}')
            print(f"  {scale:12.4g}  {r['soft'][-1]:8.4f}  {r['hard'][-1]:8.4f}")

    # ── Save numerical results ──
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robustness_results.npz'),
             SNR_DB=SNR_DB,
             shapes=list(shapes.keys()),
             **{f'exact_{name}_hard': results_exact[name]['hard'] for name in shapes},
             **{f'exact_{name}_soft': results_exact[name]['soft'] for name in shapes},
             **{f'exact_{name}_oracle': results_exact[name]['oracle'] for name in shapes},
    )
    
    return results_exact, results_perturbed, results_cross


if __name__ == '__main__':
    results_exact, results_perturbed, results_cross = main()
