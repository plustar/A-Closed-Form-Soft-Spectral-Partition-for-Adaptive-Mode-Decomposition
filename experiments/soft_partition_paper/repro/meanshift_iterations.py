# experiments/soft_partition_paper/repro/meanshift_iterations.py
"""
Meanshift iteration counts in the Experiment S5 pipeline (Table II note, Section VII-B).

The detection stage runs Meanshift from S seeds per frame; each iteration of a seed
costs O(F).  This script wraps softdamd.detect.meanshift.shift_seeds (in memory only;
the library is not modified) and uses its iteration output to record, per frame, the
number of seeds, the iterations of every seed, and the length of the loop (the slowest seed), and compares the soft
extraction cost O(F K_t) with the detection cost O(F * sum of seed iterations).
Signal and pipeline: exactly those of exp_g.py.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE)),
           os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import softdamd.detect.meanshift as M               # noqa: E402

RECORD = []


_ORIGINAL = M.shift_seeds


def _counted(seeds, freqs, weights, h, max_iter, tol, seed_idx):
    """The library routine, with its per-seed iteration counts recorded."""
    iters = np.zeros(len(seeds), dtype=np.int64)
    centers = _ORIGINAL(seeds, freqs, weights, h, max_iter, tol, seed_idx, iters=iters)
    RECORD.append(iters)
    return centers


def main():
    M.shift_seeds = _counted
    import exp_g as g
    for n in (256, 2048):
        RECORD.clear()
        res = g.build_damd().run(g.make_signal(n), g.FS)
        k_t = np.asarray(res.n_modes, dtype=float)
        seeds = np.array([r.size for r in RECORD])
        total = np.array([r.sum() for r in RECORD])
        loop = np.array([r.max() for r in RECORD])
        print(f"N = {n}: frames {len(RECORD)}, seeds per frame: median {np.median(seeds):.0f}"
              f", mean {seeds.mean():.1f}")
        print(f"   iterations per seed: mean {total.sum() / seeds.sum():.1f}")
        print(f"   loop length (slowest seed) per frame: median {np.median(loop):.0f}, "
              f"95th percentile {np.percentile(loop, 95):.0f}, max {loop.max()}")
        print(f"   soft / detection operation count  sum K_t / sum(seed iterations) = "
              f"{k_t.sum() / total.sum():.3f}   (mean K_t = {k_t.mean():.2f})")


if __name__ == '__main__':
    main()
