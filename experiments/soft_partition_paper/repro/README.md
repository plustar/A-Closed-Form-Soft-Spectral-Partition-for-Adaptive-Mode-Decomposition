# `repro/` — verification suite

Numerical checks of the statements in the paper. Nothing here is checked against a
second copy of the formula under test: closed forms are compared with a generic
simplex solver (SLSQP), the relation to VMD is measured against the actual ADMM
output, and the transform properties are measured on the shipped implementation.

```bash
python experiments/soft_partition_paper/repro/run_all.py
python experiments/soft_partition_paper/repro/s6_closed_form.py
```

Each script puts the repository root on `sys.path` itself, so it runs from any working
directory, installed or not. `s2`, `s6`, `s7`, `s8`, `s9` and `s10` accept `--quick`,
which shrinks the sweeps; the others are already short. Timing (Table II,
Experiment S5, `fig_runtime`) is produced by `exp_g.py`, not here. Statement numbers
below refer to the manuscript; "Appendix" refers to the supplementary material.

---

## Conventions (`_common.py`)

The same conventions as the package README, restated here because `_common.py` fixes
them independently of `softdamd` — a check that used the package's own constants could
not catch a disagreement between the two.

| item | convention |
|---|---|
| frequency | normalised `w = f/fs ∈ [0, 0.5]`, Nyquist at 0.5 — as in `softdamd.TFMap.freqs` and `runner.freq_axis` |
| paper α | denominator `1 + 2·α·(w − w_k)²` |
| implementation α | `softdamd/baselines/vmd.py` uses `1 + α_impl·(w − w_k)²`, so **α_impl = 2·α_paper** |
| β | `β = 1/√(2α)`, in normalised frequency |
| Voronoi ties | bins equidistant from `m` centres are shared, `s_k = 1/m` (Eq. (7)) |

α is dimensional. `α = 2000` is a half-bandwidth `β ≈ 0.0158` normalised — 15.8 Hz at
`fs = 1000`, 3.95 Hz at `fs = 250`. One value of α across four sampling rates is four
different physical bandwidths.

---

## Scripts and results

### `s5_algebraic.py` — Thm. III.1, Cor. III.2, Rem. IV.2, Rem. VI.4

- Thm. III.1, hard mask: 500 random signals and centre configurations,
  `|J_VMD − J_Kmeans| = 0`.
- Cor. III.2: strict inequality for every non-binary allocation tested (500/500
  random interior draws).
- The converse of Thm. III.1 is quantified over all signals: a non-binary allocation
  whose non-binary part lies where `P = 0` gives `J_VMD − J_Kmeans = 0` exactly.
- The identity holds at the simplex vertices only: at `s = (½, ½, 0)`,
  `J_VMD = 2.639` against `J_Kmeans = 5.277`; over 200 draws each, vertices 200/200,
  face `s₃ = 0` 0/200, interior 0/200.
- Rem. IV.2 and Rem. VI.4: without the `1/(2α)` offset the per-frequency minimiser is
  the inverse-square allocation (31) (agreement with the solver ≤ 1.4e-08), so the
  Lorentzian shape comes from the regulariser.

### `s6_closed_form.py` — Thm. IV.3, Rem. IV.4, Cor. IV.9, Rem. IV.10

This is the script's own run (800 configurations); the paper's Experiment S3 is the
same comparison over 400 configurations, produced by `exp_h.py` (Fig. S3).

| check | result |
|---|---|
| vs SLSQP, 800 configs, K ∈ [2,6], α ∈ [10, 10⁵] | median 6.7e-09, max 3.3e-07 |
| objective gap | 2.1e-17 |
| 20 000 random feasible perturbations with a lower objective | 0 |
| KKT: spread of `2c_k s_k*` | 8.7e-19 |

Rem. IV.4: with 2 coincident centres the closed form gives `[0.497268, 0.497268,
0.005464]`, matching the solver.

Cor. IV.9: α→∞ to inverse-square (5.9e-09 at α = 10¹⁰), α→0 to `1/K` (5.3e-10 at
α = 10⁻⁸). `s_k(w_k) → 1` for distinct centres; with m coincident centres the limit
is `1/m` (measured 0.49999994 and 0.33333331).

Rem. IV.10: worked example, inverse-square limit 0.599010 (≈ 0.60). The
α-dependence cancels asymptotically with an `O(1/α)` residual (4.7e-03 at α = 10,
4.9e-06 at α = 10⁴). The distance from the Lorentzian family to the Voronoi mask
saturates at 0.528.

### `s2_equivalence.py` — relation to the VMD ADMM output (Rem. IV.6, Rem. VI.5)

Compares the Lorentzian mask with `result.modes` of the ADMM solver *as it stops under
its default rule* — not with the converged fixed point, which Experiment A (`exp_a.py`)
reaches by iterating to a primal residual of 1e-10 and finds within 1e-10 of the
inverse-square allocation. The distances below are therefore dominated by the solver's
remaining reconstruction residual. At α = 2000,
three well-separated modes: per-mode relative distance `0.0918 / 0.1153 / 0.1712`;
after removing the ADMM primal residual `0.0948 / 0.1157 / 0.1663`. Reconstruction
error: ADMM 0.1308, allocation 3.3e-16.

| axis | best agreement | where |
|---|---|---|
| α | 0.109 | α = 1000 |
| separation | 0.114 | sep/β ≈ 7.6, flat beyond ≈ 5 |
| SNR | 0.119 | ≥ 30 dB |

### `unit_weighted_qp.py` — Cor. IV.7, Rem. IV.8, Thm. V.2, Rem. V.3

- With the cost of Eq. (14), `(d_k² + β_k²)/(σ_k² β_k²)`, Eq. (15) is the exact
  minimiser: max deviation from SLSQP 1.8e-08 over 500 configurations.
- Rem. IV.8: normalising by `σ_k²` alone adds a factor `α_k`. Instance `w = 0.5`,
  centres `(0, 1)`, `σ² = (1, 1)`, `α = (1, 10)`: Eq. (15) gives `s₁ = 0.800`, the
  minimiser of that cost `s₁ = 0.286` (SLSQP agrees). Over 2000 random configurations
  with unequal α the median deviation is 0.379; with equal α it is 1.0e-10, so
  Eq. (12) is unaffected.
- Thm. V.2's MMSE weights equal Eq. (15) exactly (difference 0.0); Eq. (12) follows
  when both `σ_k² = σ²` and `α_k = α` (0.527 with unequal α, 0.0 with equal α).

### `s7_mmse.py` — Thm. V.2, Rem. V.4 (Experiment C)

Thm. V.2 exact in the limit: `max |S₁/S_X − s₁*| = 0.0` at `σ_N² = 0`.

Rem. V.4 pointwise: the shrinkage factor is 0.999988 at the mode centre and 0.9995
between modes at 40 dB; 0.892 and 0.167 at 0 dB. `Σ_k s_k* = 1` at every SNR; the MMSE
weights sum to 0.297 at 0 dB and 0.999 at 40 dB.

Monte Carlo (1500 trials per SNR, nMSE):

| SNR | hard | soft | soft × shrinkage | oracle |
|---|---|---|---|---|
| −10 | 8.51 | 7.26 | **0.730** | **0.730** |
| 0 | 0.901 | 0.768 | **0.311** | **0.311** |
| 20 | 0.062 | 0.052 | 0.0513 | 0.0513 |
| 60 | 0.053 | 0.044 | 0.0443 | 0.0443 |

Soft × shrinkage matches the oracle at every SNR. The high-SNR value 0.0443 is the
irreducible error of separating two overlapping Lorentzian modes.

### `s8_fwhm.py`, `fwhm_bound_scan.py`, `fwhm_overlap.py` — Cor. V.5 (Experiment S2)

Accuracy is governed by the number of bins across the FWHM, `n = 2β/dw`, not by α:

| α | bins/FWHM | 1/n | measured | ≤10% |
|---|---|---|---|---|
| 316 | 20 | 0.0500 | 0.0362 | yes |
| 681 | 14 | 0.0714 | 0.0184 | yes |
| 1468 | 9 | 0.1111 | 0.1024 | no |
| 3162 | 7 | 0.1429 | 0.1541 | no |

`fwhm_bound_scan.py`: the worst case over the centre offset never exceeds
`(1 − 1/n)⁻² − 1`; it exceeds 10% up to n = 20.0 on the continuous scan; the mean
follows `4f(1 − f)/n` to within 0.013 for n ≥ 10.

Band: with a neighbour at 0.36 and α = 5000, the error is 0.738 over the full
spectrum and 0.049 inside the Voronoi band; at α = 200 the band truncates the mode
(0.55 → 0.82).

Noise (20 dB, median relative error, L averaged periodograms, α = 1000):
`L = 1/4/16/64` gives `7.19 / 5.72 / 0.83 / 0.090` over 40 trials. The paper's Fig. S2(b)
is produced by `exp_e.py` with 60 trials and reads 0.31 at the same point; the two are
medians of an error quantised by the bin count, and differ by sampling scatter, not by
setting. Both say the same thing: a smoothed spectrum is a precondition, and averaging
is necessary but not sufficient at this SNR.

`fwhm_overlap.py` (n = 11.45): isolated error 8.3%; neighbour at 3/2/1.5 FWHM above
gives 9.0/22.4/41.7%; a common centre shift of up to one FWHM leaves the error at
8.3–9.0%, two FWHM gives 75–424%.

The Type-II MLE baseline (Eq. (27)) is specified in full: centre, mode power and
noise floor known; `log α` optimised by bounded Brent, `xatol = 1e-4`.

### `s9_convergence.py` — Prop. VI.1, Prop. VI.2, Rem. VI.3

Prop. VI.1: gradient at the centroid ≤ 1.8e-14, no perturbation better, and
`J̃ − J_VMD` matches `(1/2α)∫Σ_k s_k² P` to 6e-17.

Prop. VI.2: the objective is non-increasing from every start (largest increase
1.1e-16). Objective values converge; the limiting centres depend on the start
(Rem. VI.3):

| start | limit `J̃` | limit centres |
|---|---|---|
| near true centres | 0.25545411 | 0.087 / 0.215 / 0.348 |
| poor start | 0.25545411 | 0.087 / 0.215 / 0.348 |
| two coincident | 0.40712270 | 0.123 / 0.123 / 0.344 |
| all in one hump | 0.25545411 | 0.087 / 0.215 / 0.348 |

### `s10_reconstruction.py` — Rem. VII.1 (Appendix G)

| transform | round trip raw / gain-corrected (three tones) | energy-ratio spread |
|---|---|---|
| STFT hop=1 | 6.1e-16 / 5.8e-16 | 0.007 |
| STFT hop=64 | 2.1e-16 / 2.6e-16 | 0.007 |
| SSQ-STFT hop=1 | 7.7e-06 / 7.7e-06 | 0.609 |
| CWT | 3.2e-02 / 2.7e-02 | 0.058 |
| SSQ-CWT | 3.2e-02 / 2.7e-02 | 0.844 |

The unit-hop STFT pair is exact and an isometry to within 0.7%. The masked array
`s_k W x` does not stay in the range of `W`: the analyse–synthesise–analyse residual
is 1.2–2.1e-03 per mode for three tones and 1.7–3.2e-02 for white noise. By linearity,
`Σ_k u_k` recovers `x` to the transform's own round-trip error (6.5e-16 for the
unit-hop STFT). Experiment D applies no inverse transform; the inverse is used only by
`softdamd.reconstruct` (Experiment S5) and Experiment S4, both with the STFT.

### `s11_boundary_energy.py` — the measurement of Table IV (Experiment D, Appendix H)

Computes the two quantities of Table IV: Boundary is the fraction of frame energy
within ±r bins of a Voronoi boundary; the allocation difference is
`½ Σ_k |s_k^soft − s_k^hard| P` over the same bins, divided by the frame energy; both
averaged over frames, weighted by frame energy. The second is an input-power-weighted
difference between the two allocations, not energy transferred between reconstructed
modes. Also prints transform, fs, duration, frame count, bin width, neighbourhood width
in Hz, and `K_t` (mean/median/min/max, total detections), then the Appendix H detail:
the difference as a share of the boundary energy, the fraction of bins covered, and, on
the logarithmic SSQ-CWT axis, the split at 5 Hz and the fraction of adjacent centre
pairs closer than the mask FWHM.

The four recordings of Experiment D are third-party data and are not included, so the
paper's Table IV is not reproduced here. The script measures the same quantities on the four synthetic
signals of `synthetic_suite.py`, with the Experiment D detection settings
(`n_fft = fs/2`, percentile bandwidth, curvature adaptation):

| signal | transform | boundary, r = 1 | allocation difference, r = 1 |
|---|---|---|---|
| Harmonic stack | SSQ-CWT | 5.55% | 2.76% |
| Crossing chirps | SSQ-STFT | 2.61% | 0.71% |
| Intermittent tones | SSQ-STFT | 0.10% | 0.05% |
| Moving formants | SSQ-STFT | 5.07% | 2.45% |

Section 1 of the output repeats the measurement on the crossing chirps with the
detector's own defaults instead, where the finer bin and the larger mode count raise
both figures (18.17% and 10.02% at r = 1): the percentages are read with the
neighbourhood width and the detection settings, never on their own.

### `ssq_stft_tone_check.py` — SSQ-STFT frequency axis (Appendix H)

The SSQ-STFT rows lie on the DFT grid `k·fs/n_fft` (2.000 Hz for all three SSQ-STFT
rows of Table IV, including the odd `n_fft = 125`); stationary-tone centroids fall
within ±0.30 DFT bins of the true frequency.

### `meanshift_iterations.py` — Table II note, Section VII-B (Experiment S5)

13 seeds per frame; 16.5 (N = 256) and 13.6 (N = 2048) mean-shift iterations per seed
on average; slowest seed per frame median 35 / 30, maximum 193 / 305. Soft extraction
costs 3.2% / 4.5% of the detection operation count (mean `K_t` = 6.84 / 7.95).

---

## Statement coverage

| statement | script |
|---|---|
| Thm. III.1, Cor. III.2 | `s5_algebraic` |
| Rem. IV.2 | `s5_algebraic` |
| Thm. IV.3, Rem. IV.4 | `s6_closed_form` |
| Rem. IV.6, Rem. VI.5 | `s2_equivalence` (see also `exp_a.py`) |
| Cor. IV.7, Rem. IV.8 | `unit_weighted_qp` |
| Cor. IV.9, Rem. IV.10 | `s6_closed_form`, `exp_c.py` |
| Thm. V.2, Rem. V.3 | `unit_weighted_qp`, `s7_mmse` |
| Rem. V.4 | `s7_mmse` |
| Cor. V.5 | `s8_fwhm`, `fwhm_bound_scan`, `fwhm_overlap` |
| Prop. VI.1, Prop. VI.2, Rem. VI.3 | `s9_convergence` |
| Rem. VII.1 | `s10_reconstruction` |
| Alg. 1, Table II | `exp_g.py`, `meanshift_iterations` |
| Table IV (measurement) | `s11_boundary_energy`, `ssq_stft_tone_check` |
