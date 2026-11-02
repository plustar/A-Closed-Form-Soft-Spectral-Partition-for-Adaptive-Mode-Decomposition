# experiments/soft_partition_paper/run_all.py
"""
Run all experiments for "A Closed-Form Soft Spectral Partition for Adaptive Mode
Decomposition" and save figures to the paper's figure directory.

Scripts (selected with --only by their letter), the experiment label in the paper
(S = supplementary material), and the statement each figure supports:

  a  A   fig_vmd_agreement      Rem. VI.5 -- the VMD ADMM fixed point
  b  S1  figS_wiener_identity   Rem. IV.5 -- implementation unit test
  c  B   fig_lorentzian_limits  Cor. IV.9, Rem. IV.10 -- the two limits of the family
  d  C   fig_mmse_shrinkage     Thm. V.2, Rem. V.4 -- MMSE and the shrinkage factor
  e  S2  fig_alpha_estimation   Cor. V.5 -- the FWHM estimator
  f  -   fig_synthetic_suite    hard vs soft on the synthetic suite, with the
         boundary measurement of Experiment D
  g  S5  fig_runtime            Table II, Alg. 1 -- stage-separated cost (slow)
  h  S3  fig_closed_form        Thm. IV.3 -- the closed form is the minimiser
                                (unique where P > 0)

Experiment C' (fig_robustness) and S4 have their own scripts: exp_robustness.py,
exp_separation_sweep.py, plot_robustness_3panel.py, exp_resolution.py and
exp_recovery.py.

File names come from figures.py; see that module for the registry and for
``--latex`` / ``--check``.

Usage
-----
  python experiments/soft_partition_paper/run_all.py
  python experiments/soft_partition_paper/run_all.py --save_dir /path/to/figs
  python experiments/soft_partition_paper/run_all.py --only a b d g

Every script here is self-contained: the signals are generated from synthetic_suite.py
and from the scripts themselves, so nothing has to be downloaded.

Script g (Experiment S5) runs STVMD at hop = 1 and is slow; run it on its own:
  python experiments/soft_partition_paper/exp_g.py --lengths 256 512 1024 2048 4096
"""

import sys, os, argparse, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # repository root, where the softdamd package lives
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_SAVE = os.path.join(_ROOT, 'outputs/soft_partition')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', default=DEFAULT_SAVE)
    ap.add_argument('--only', nargs='*', default=None,
                    choices=list('abcdefgh'),
                    help='Run only these experiments')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    run = set(args.only) if args.only else set('abcdefgh')

    exps = {
        'a': ('exp_a', 'fig_vmd_agreement      Rem. VI.5 (VMD ADMM fixed point)'),
        'b': ('exp_b', 'figS_wiener_identity   Rem. IV.5 [unit test, supplementary]'),
        'c': ('exp_c', 'fig_lorentzian_limits  Cor. IV.9, Rem. IV.10'),
        'd': ('exp_d', 'fig_mmse_shrinkage     Thm. V.2, Rem. V.4'),
        'e': ('exp_e', 'fig_alpha_estimation   Cor. V.5'),
        'f': ('exp_d_synthetic', 'fig_synthetic_suite   boundary measurement'),
        'g': ('exp_g', 'fig_runtime            Table II, Alg. 1  [slow: runs STVMD]'),
        'h': ('exp_h', 'fig_closed_form        Thm. IV.3'),
    }

    total_t0 = time.perf_counter()

    for key in sorted(run):
        mod_name, desc = exps[key]
        print(f'\n{"="*60}')
        print(f'Script {key.upper()}: {desc}')
        print('='*60)
        t0 = time.perf_counter()

        mod = __import__(
            f'soft_partition_paper.{mod_name}',
            fromlist=[mod_name]
        )

        mod.main(save_dir=args.save_dir)
        print(f'  → done in {time.perf_counter()-t0:.1f}s')

    total = time.perf_counter() - total_t0
    print(f'\nAll experiments done in {total:.1f}s')
    print(f'Figures saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
