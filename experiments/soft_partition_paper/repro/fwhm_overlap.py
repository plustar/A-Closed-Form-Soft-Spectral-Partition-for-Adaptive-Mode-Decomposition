# experiments/soft_partition_paper/repro/fwhm_overlap.py
"""
FWHM estimator (Corollary V.5) with a neighbouring mode and with centre errors
(Experiment S2, "Overlap and centre error", supplementary Appendix C).

Noiseless, on the grid of exp_e.py: w = k/256, k = 0..128 (n_fft = 256), mode centre
0.30.  A second equal-amplitude Lorentzian is placed s FWHM away; the band of the mode
is cut at the Voronoi midpoint of the (assumed) centres, and the estimator counts the
bins at or above half the band maximum.  Both sides of the neighbour are reported,
because the grid position makes the two cases differ.
"""
import numpy as np

W0 = 0.30
W = np.arange(129) / 256.0


def lor(w, c, a):
    return 1.0 / (1.0 + 2.0 * a * (w - c) ** 2)


def err(P, band_mask, alpha):
    band = P[band_mask]
    m = int((band >= band.max() / 2.0).sum())
    return abs(2.0 / (m / 256.0) ** 2 / alpha - 1.0)


def main():
    print("1. Neighbour at s FWHM, band to the Voronoi midpoint of the true centres")
    for alpha in (1000.0, 5000.0):
        F = 2.0 / np.sqrt(2.0 * alpha)
        iso = err(lor(W, W0, alpha), np.ones(W.size, bool), alpha)
        for side, name in ((+1, 'above'), (-1, 'below')):
            row = []
            for s in (3.0, 2.0, 1.5):
                c2 = W0 + side * s * F
                P = lor(W, W0, alpha) + lor(W, c2, alpha)
                row.append(err(P, np.abs(W - W0) < np.abs(W - c2), alpha))
            print(f"   n = {F * 256:5.2f}, neighbour {name}: isolated {100 * iso:5.1f}%   "
                  + "   ".join(f"s={s}: {100 * e:5.1f}%" for s, e in zip((3, 2, 1.5), row)))
    print("\n2. s = 3, n = 11.45, both assumed centres displaced by d FWHM (common shift)")
    alpha = 1000.0
    F = 2.0 / np.sqrt(2.0 * alpha)
    for side, name in ((+1, 'above'), (-1, 'below')):
        c2 = W0 + side * 3.0 * F
        P = lor(W, W0, alpha) + lor(W, c2, alpha)
        out = []
        for d in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            a1, a2 = W0 + d * F, c2 + d * F
            out.append(f"{d:+.1f}: {100 * err(P, np.abs(W - a1) < np.abs(W - a2), alpha):6.1f}%")
        print(f"   neighbour {name}:  " + "  ".join(out))


if __name__ == '__main__':
    main()
