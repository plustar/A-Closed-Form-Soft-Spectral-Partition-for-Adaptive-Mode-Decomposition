# experiments/soft_partition_paper/__init__.py
"""
Experiments for:
  "A Closed-Form Soft Spectral Partition for Adaptive Mode Decomposition"
  IEEE Transactions on Signal Processing

Experiment index (labels as in the paper; S = supplementary material)
--------------------------------------------------------------------
A   exp_a.py             fig_vmd_agreement      Remark VI.5 -- VMD ADMM fixed point
B   exp_c.py             fig_lorentzian_limits  Corollary IV.9, Remark IV.10
C   exp_d.py             fig_mmse_shrinkage     Theorem V.2, Remark V.4 -- MMSE
C'  exp_robustness.py,   fig_robustness         robustness to misspecification
    exp_separation_sweep.py, exp_resolution.py
-   exp_d_synthetic.py   fig_synthetic_suite    hard vs soft on the synthetic suite
S1  exp_b.py             figS_wiener_identity   implementation consistency
S2  exp_e.py             fig_alpha_estimation   Corollary V.5 -- FWHM estimator
S3  exp_h.py             fig_closed_form        Theorem IV.3
S4  exp_recovery.py      --                     time-domain mode recovery
S5  exp_g.py             fig_runtime            computational cost vs STVMD
"""
