# experiments/soft_partition_paper/repro/fwhm_bound_scan.py
"""
Continuous scan of the FWHM estimator error over the bin count n and the centre offset
(Corollary V.5; Experiment S2, supplementary Appendix C and Fig. S2).

For an isolated, noiseless Lorentzian sampled on a unit grid with n = 2*beta/dw bins
across the FWHM, the estimator counts the bins with P >= max(P)/2 and returns
alpha_hat = 2/(m*dw)^2.  The scan checks
  (1) that |alpha_hat/alpha - 1| never exceeds (1 - 1/n)^-2 - 1,
  (2) the largest n at which the worst case over the centre offset exceeds 10 %,
  (3) the mean error over the offset against the first-order value 4 f (1-f) / n,
      f = frac(n).
"""
import numpy as np


def errors(n, offsets):
    beta = n / 2.0
    alpha = 1.0 / (2.0 * beta ** 2)
    w = np.arange(-4000, 4001, dtype=float)
    out = []
    for c in offsets:
        P = 1.0 / (1.0 + (w - c) ** 2 / beta ** 2)
        m = np.sum(P >= P.max() / 2.0)
        out.append(abs(2.0 / m ** 2 / alpha - 1.0))
    return np.array(out)


def main():
    offs = np.linspace(0.0, 1.0, 401, endpoint=False)
    ns = np.arange(2.0, 40.0, 0.05)
    worst, mean = [], []
    for n in ns:
        e = errors(n, offs)
        worst.append(e.max())
        mean.append(e.mean())
    worst, mean = np.array(worst), np.array(mean)
    bound = (1.0 - 1.0 / ns) ** -2 - 1.0
    f = ns - np.floor(ns)
    first = 4.0 * f * (1.0 - f) / ns
    print(f"bound violated anywhere: {bool(np.any(worst > bound + 1e-12))}")
    print(f"largest n with worst-case error > 10 %: {ns[worst > 0.10].max():.2f}")
    sel = ns >= 10
    print(f"n >= 10: max |mean - 4f(1-f)/n| = {np.max(np.abs(mean - first)[sel]):.4f}; "
          f"mean error <= 1/n everywhere: {bool(np.all(mean[sel] <= 1 / ns[sel] + 1e-12))}")
    for n in (5.5, 11.4, 22.0, 25.6):
        e = errors(n, offs)
        print(f"  n = {n:5.1f}: worst {e.max():.4f}, mean {e.mean():.4f}, "
              f"4f(1-f)/n {4 * (n % 1) * (1 - n % 1) / n:.4f}, 1/n {1 / n:.4f}")


if __name__ == '__main__':
    main()
