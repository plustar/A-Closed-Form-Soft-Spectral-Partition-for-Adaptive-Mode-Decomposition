# Experiments — A Closed-Form Soft Spectral Partition for Adaptive Mode Decomposition

This tree contains the experiments of the paper.  Every script uses the
`softdamd` package at the repository root.

```
experiments/
  common/                      pipeline runner (runner.py) and plotting (plot_utils.py)
  soft_partition_paper/        the figures and tables of the paper
    exp_a.py .. exp_h.py       main-text and supplementary figures (mapping below)
    exp_robustness.py          Experiment C' numbers: spectral shape, centre error,
                               shape x centre error (Part 3), alpha mismatch (Part 4)
    exp_separation_sweep.py    Experiment C' separation sweep (writes the .npz read by
                               plot_robustness_3panel.py, which draws Fig. 4)
    exp_resolution.py          Experiment C' frequency-resolution ablation
                               (supplement, Appendix I)
    exp_g.py                   Experiment S5 timings (supplement, Appendix F); run on the
                               timing machine.  `exp_g.py --recon` prints only the two
                               reconstruction errors quoted in Section VII-B
    synthetic_suite.py         four synthetic multi-component signals, with their seeds
    exp_d_synthetic.py         the Experiment D pipeline and boundary measurement on
                               those four signals
    run_all.py                 regenerate all figures
    repro/                     verification suite for the paper's statements
```

## Which script produces what

The paper does not name files; this table is the mapping (seeds as stated in the
paper, or in `synthetic_suite.py` for the synthetic signals).

| Experiment | Figure / table | Script(s) in `soft_partition_paper/` | Seed |
|---|---|---|---|
| A — VMD fixed point | Fig. 1 | `exp_a.py` | 42 |
| B — limits of the Lorentzian family | Fig. 2 | `exp_c.py` | 42 |
| C — MMSE and shrinkage | Fig. 3 | `exp_d.py` | 42 |
| C′ — robustness | Fig. 4 | `exp_robustness.py` (numbers), `exp_separation_sweep.py`, `plot_robustness_3panel.py` (figure) | 42 |
| C′ — frequency resolution | Appendix I | `exp_resolution.py` | 42 |
| D — the pipeline on four signals | `fig_synthetic_suite` | `exp_d_synthetic.py`, `synthetic_suite.py` | 0–3 |
| S1 — implementation consistency | Fig. S1 | `exp_b.py` | 0 |
| S2 — FWHM estimator | Fig. S2 | `exp_e.py`, `repro/fwhm_bound_scan.py`, `repro/fwhm_overlap.py` | 0 |
| S3 — Theorem IV.3 vs numerical minimiser | Fig. S3 | `exp_h.py` | 0 |
| S4 — time-domain recovery | Table S1 | `exp_recovery.py` | 42 |
| S5 — computational cost | Fig. S4 | `exp_g.py` (timing machine); `exp_g.py --recon` for the two reconstruction errors | 42 |
| Remark VII.1 / Appendix G | — | `repro/s10_reconstruction.py` | — |
| Table II note (mean-shift counts) | — | `repro/meanshift_iterations.py` | — |

## Running

```bash
pip install -e ".[viz]"
python experiments/soft_partition_paper/run_all.py --save_dir outputs
python experiments/soft_partition_paper/repro/run_all.py
```

Every script puts the repository root and `experiments/` on `sys.path` itself, so no
`PYTHONPATH` is needed and the working directory does not matter.

Every experiment here is self-contained. Fig. 5 and Table IV of the paper come from
four third-party recordings that are not included and are therefore not reproduced in
this repository; `exp_d_synthetic.py` runs the same pipeline and the same measurement on the
synthetic signals of `synthetic_suite.py`.

## `repro/`

`repro/` verifies the numbered statements of the paper, never against a second copy of
the formula under test. `repro/README.md` describes each script and records what it
measured.
