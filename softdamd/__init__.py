"""softdamd — closed-form soft spectral partition for adaptive mode decomposition.

Implements Algorithm 1 of the paper for single-channel signals:

    𝒯            softdamd.tf        STFT, SSQ-STFT, CWT, SSQ-CWT
    Step 1        softdamd.detect    per-frame mean-shift detection (DAMD)
    Step 2        softdamd.extract   hard Voronoi / Lorentzian soft masks, FWHM α̂
    end to end    SoftDAMD(...).run(x, fs) -> Decomposition

Baselines: softdamd.baselines.VMD, STVMD.  Conventions: softdamd.units.
"""
from .tf import STFT, SSQSTFT, CWT, SSQCWT, TFMap, make_transform
from .detect import MeanShift, Partition
from .extract import (lorentzian_weights, voronoi_weights, fwhm_alpha,
                      frame_weights, extract, mode_energy, reconstruct)
from .pipeline import SoftDAMD, Decomposition

__version__ = '1.0.0'

__all__ = ['SoftDAMD', 'Decomposition',
           'STFT', 'SSQSTFT', 'CWT', 'SSQCWT', 'TFMap', 'make_transform',
           'MeanShift', 'Partition',
           'lorentzian_weights', 'voronoi_weights', 'fwhm_alpha',
           'frame_weights', 'extract', 'mode_energy', 'reconstruct']
