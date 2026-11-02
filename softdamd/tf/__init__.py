"""Time-frequency transforms 𝒯 used in the paper."""
from .base import TFMap, Transform, as_signal
from .stft import STFT, SSQSTFT, make_window, derivative_window
from .cwt import CWT, SSQCWT

TRANSFORMS = {'stft': STFT, 'ssq_stft': SSQSTFT, 'cwt': CWT, 'ssq_cwt': SSQCWT}


def make_transform(name: str, **kwargs) -> Transform:
    """Transform by name: 'stft', 'ssq_stft', 'cwt' or 'ssq_cwt'."""
    try:
        return TRANSFORMS[name](**kwargs)
    except KeyError:
        raise ValueError(f'unknown transform {name!r}; choose from {sorted(TRANSFORMS)}') from None


__all__ = ['TFMap', 'Transform', 'STFT', 'SSQSTFT', 'CWT', 'SSQCWT',
           'make_transform', 'make_window', 'derivative_window', 'as_signal']
