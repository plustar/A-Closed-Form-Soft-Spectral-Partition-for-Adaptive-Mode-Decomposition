"""Algorithm 1 end to end: transform → detect → extract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .detect.meanshift import MeanShift
from .detect.partition import Partition
from .extract import Alpha, extract, frame_weights, mode_energy, reconstruct
from .tf.base import TFMap, Transform
from .tf.stft import STFT


@dataclass(frozen=True, eq=False)
class Decomposition:
    """Result of :meth:`SoftDAMD.run`.

    Attributes
    ----------
    tfmap : TFMap          𝒯(ω, t) of the signal
    partition : Partition  detected centres ω*_{k,t}, K_t per frame
    alpha : default α of the soft mask (number, per-mode array, or 'fwhm')

    Frame indices k are local to each frame: they are not associated across frames.
    """
    tfmap: TFMap
    partition: Partition
    alpha: Alpha = 2000.0

    @property
    def freqs(self) -> np.ndarray:
        return self.tfmap.freqs

    @property
    def n_modes(self) -> np.ndarray:
        return self.partition.n_modes

    def weights(self, t: int, mode: str = 'soft', alpha: Optional[Alpha] = None) -> np.ndarray:
        """s_k(ω) of frame t, (K_t, F)."""
        return frame_weights(self.tfmap, self.partition, t, mode,
                             self.alpha if alpha is None else alpha)

    def modes(self, t: int, mode: str = 'soft', alpha: Optional[Alpha] = None) -> np.ndarray:
        """U_k(ω, t), (K_t, F) complex."""
        return extract(self.tfmap, self.partition, t, mode,
                       self.alpha if alpha is None else alpha)

    def energy(self, t: int, mode: str = 'soft', alpha: Optional[Alpha] = None) -> np.ndarray:
        """E_k = Σ_ω s_k(ω)|𝒯(ω, t)|², (K_t,)."""
        return mode_energy(self.tfmap, self.partition, t, mode,
                           self.alpha if alpha is None else alpha)

    def reconstruct(self, mode: str = 'soft', alpha: Optional[Alpha] = None) -> np.ndarray:
        """Inverse transform of Σ_k U_k (a check of the partition of unity)."""
        return reconstruct(self.tfmap, self.partition, mode,
                           self.alpha if alpha is None else alpha)


class SoftDAMD:
    """DAMD with soft spectral partition (Algorithm 1).

    Parameters
    ----------
    transform : time-frequency transform 𝒯 (default STFT, n_fft 256, hop 1)
    detector : mode detector (default :class:`MeanShift` with its defaults)
    alpha : α of the soft mask, manuscript convention; 'fwhm' estimates α̂_k
    """

    def __init__(self, transform: Optional[Transform] = None,
                 detector: Optional[MeanShift] = None, alpha: Alpha = 2000.0) -> None:
        self.transform = transform if transform is not None else STFT()
        self.detector = detector if detector is not None else MeanShift()
        self.alpha = alpha

    def run(self, x, fs: float = 1.0) -> Decomposition:
        tfmap = self.transform.forward(x, fs)
        partition = self.detector.detect(tfmap.power(), tfmap.freqs)
        return Decomposition(tfmap=tfmap, partition=partition, alpha=self.alpha)

    def __repr__(self) -> str:
        return f'SoftDAMD({self.transform!r}, {self.detector!r}, alpha={self.alpha!r})'
