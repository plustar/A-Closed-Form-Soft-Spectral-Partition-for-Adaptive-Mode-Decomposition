"""Pure-tone check of the SSQ-STFT frequency axis (Table IV, supplementary Appendix H).

(1) The SSQ-STFT rows are the DFT grid k·fs/n_fft, also for odd n_fft.
(2) Energy centroid of a stationary tone, in DFT bins, at the Experiment D settings.
Run from the repository root:  python3 <this file>
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(os.path.dirname(_HERE))   # experiments/
_ROOT = os.path.dirname(_EXP)                     # repository root, where softdamd lives
for _p in (_ROOT, _EXP, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from softdamd import SSQSTFT


def ssq_power(x, fs, nfft):
    m = SSQSTFT(n_fft=nfft, hop=1).forward(x, fs)
    P = (np.abs(m.coef) ** 2)[:, nfft:-nfft]
    return P.sum(axis=1), m.freqs_hz

for fs, nfft in ((220., 110), (250., 125), (500., 250)):
    t = np.arange(int(4 * fs)) / fs
    errs = []
    for frac in (0.1, 0.17, 0.23, 0.31, 0.37):
        f0 = frac * fs
        P, fh = ssq_power(np.cos(2 * np.pi * f0 * t), fs, nfft)
        k = int(np.argmax(P)); s = slice(max(k - 3, 0), k + 4)
        errs.append(((P[s] * fh[s]).sum() / P[s].sum() - f0) / (fs / nfft))
    print(f"fs={fs:.0f} n_fft={nfft}: grid step {fh[1] - fh[0]:.4f} Hz, DFT {fs / nfft:.4f} Hz;"
          " tone centroid error (DFT bins): " + " ".join(f"{e:+.2f}" for e in errs))

fs, nfft = 250., 125
t = np.arange(1000) / fs
for f0 in (40., 100.):
    P, fh = ssq_power(np.cos(2 * np.pi * f0 * t), fs, nfft)
    k = int(np.argmax(P))
    print(f"n_fft=125, {f0:.0f} Hz tone: peak bin {k} labelled {fh[k]:.3f} Hz "
          f"(DFT frequency of that bin {k * fs / nfft:.3f} Hz)")
