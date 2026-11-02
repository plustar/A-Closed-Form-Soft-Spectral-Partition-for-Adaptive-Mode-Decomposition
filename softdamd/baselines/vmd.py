"""
Baselines of the paper (single channel):
- VMD: Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014)
- STVMD: Short-Time VMD (Jia et al., 2025)

Both share the same ADMM kernel (following the reference implementation of [1]),
differing only in the transform and
in whether the centre frequencies vary with time.

Conventions of the reference implementation are kept: the ADMM denominator is
1 + alpha (omega - omega_k)^2 on the axis omega = k / F of the (mirror-padded) FFT,
and ``center_freqs`` are on that axis.  See :mod:`softdamd.units` for conversions.
Modes are returned as (K, T).

References
----------
[1] K. Dragomiretskiy and D. Zosso, "Variational Mode Decomposition,"
    IEEE TSP, vol. 62, no. 3, pp. 531-544, 2014.
[2] H. Jia et al., "Short-time Variational Mode Decomposition,"
    Signal Processing, vol. 238, 110203, 2026.
"""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import scipy.signal
from scipy.fft import rfft, irfft
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class VMDResult:
    """Container for VMD decomposition results."""
    modes       : np.ndarray   # (K, T) time-domain mode signals
    center_freqs: np.ndarray   # (K,) or (K, N) centre frequencies (normalised)
    fs          : float        # sample rate in Hz
    n_iterations: int          # ADMM iterations until convergence
    converged   : bool         # whether convergence criterion was met
    method      : str = 'VMD'  # algorithm identifier
    n_fft       : int  = 0     # STFT window length (STVMD only)
    residual    : float = 0.0  # reconstruction residual (STVMD only)


# =============================================================================
# VMD Base Class
# =============================================================================

class VMDBase(ABC):
    """
    Abstract base class for VMD-family algorithms.
    
    Parameters
    ----------
    K : int
        Number of modes to extract
    alpha : float, default=50
        Bandwidth constraint (higher = narrower)
    tau : float, default=1e-5
        Lagrangian multiplier step
    tol : float, default=1e-9
        Convergence tolerance
    maxiters : int, default=500
        Maximum ADMM iterations
    """
    
    def __init__(
        self,
        K: int,
        alpha: float = 50,
        tau: float = 1e-5,
        tol: float = 1e-9,
        maxiters: int = 500
    ):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.maxiters = maxiters
    
    @abstractmethod
    def decompose(self, signal: np.ndarray, fs: float = 1.0, **kwargs) -> VMDResult:
        """Decompose signal into modes."""
        pass
    
    def __call__(self, signal: np.ndarray, fs: float = 1.0, **kwargs) -> VMDResult:
        return self.decompose(signal, fs=fs, **kwargs)
    
    def _admm_optimize(
        self,
        f_hat: np.ndarray,
        freqs: np.ndarray,
        omega_init: Optional[np.ndarray],
        dynamic: bool,
        verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """
        Core ADMM optimization loop.
        
        Parameters
        ----------
        f_hat : ndarray (1, F) or (1, F, N)
            Frequency-domain signal
        freqs : ndarray (F,)
            Normalized frequency axis [0, 1]
        omega_init : ndarray, optional
            Initial center frequencies
        dynamic : bool
            Whether omega varies across frames
        verbose : bool
            Print progress
        
        Returns
        -------
        u_hat : ndarray
            Mode spectra
        omega : ndarray (K,) or (K, N)
            Final center frequencies
        n_iter : int
            Iterations used
        converged : bool
        """
        # Determine dimensions
        if f_hat.ndim == 2:
            C, F = f_hat.shape
            N = 1
            is_stft = False
            f_hat = f_hat[:, :, np.newaxis]
        else:
            C, F, N = f_hat.shape
            is_stft = True
        
        K = self.K
        alpha = self.alpha * np.ones(K)
        
        # Initialize omega
        if dynamic and is_stft:
            omega_plus = np.zeros([2, K, N])
            omega_shape = 'dynamic'
        else:
            omega_plus = np.zeros([2, K])
            omega_shape = 'global'
        
        # Set initial frequencies
        if omega_init is not None:
            omega_plus[0] = omega_init
        else:
            for k in range(K):
                if omega_shape == 'dynamic':
                    omega_plus[0, k, :] = k / K
                else:
                    omega_plus[0, k] = k / K
        
        # Initialize mode spectra and Lagrangian
        u_hat_plus = np.zeros([2, C, F, K, N], dtype=complex)
        sum_uk = np.zeros([C, F, N], dtype=complex)
        lambda_hat = np.zeros([2, C, F, N], dtype=complex)
        
        n_iter = 0
        converged = False
        
        for iter_idx in range(self.maxiters - 1):
            n_iter = iter_idx + 1
            curr = iter_idx % 2
            next_ = (iter_idx + 1) % 2
            
            # Update modes
            for k in range(K):
                if k == 0:
                    sum_uk = (u_hat_plus[curr, :, :, K-1, :] + 
                             sum_uk - u_hat_plus[curr, :, :, 0, :])
                else:
                    sum_uk = (u_hat_plus[next_, :, :, k-1, :] + 
                             sum_uk - u_hat_plus[curr, :, :, k, :])
                
                if omega_shape == 'dynamic':
                    omega_k = omega_plus[curr, k, :].reshape(1, 1, N)
                else:
                    omega_k = omega_plus[curr, k]
                
                freqs_bc = freqs.reshape(1, -1, 1)
                denom = 1 + alpha[k] * (freqs_bc - omega_k) ** 2
                
                u_hat_plus[next_, :, :, k, :] = (
                    f_hat - sum_uk - lambda_hat[curr] / 2
                ) / (denom + 1e-15)
            
            # Update center frequencies
            if omega_shape == 'dynamic':
                for k in range(K):
                    for n in range(N):
                        power = np.sum(np.abs(u_hat_plus[next_, :, :, k, n]) ** 2, axis=0)
                        num = np.sum(freqs * power)
                        den = np.sum(power)
                        if den > 1e-12:
                            omega_plus[next_, k, n] = num / (den + 1e-15)
                        else:
                            omega_plus[next_, k, n] = omega_plus[curr, k, n]
            else:
                for k in range(K):
                    power = np.sum(np.abs(u_hat_plus[next_, :, :, k, :]) ** 2, axis=(0, 2))
                    num = np.sum(freqs * power)
                    den = np.sum(power)
                    if den > 1e-12:
                        omega_plus[next_, k] = num / (den + 1e-15)
                    else:
                        omega_plus[next_, k] = omega_plus[curr, k]
            
            # Update Lagrangian
            residual = np.sum(u_hat_plus[next_], axis=2) - f_hat
            lambda_hat[next_] = lambda_hat[curr] + self.tau * residual
            
            # Convergence check
            delta = u_hat_plus[0] - u_hat_plus[1]
            uDiff = np.real(np.vdot(delta, delta)) / delta.size
            
            if verbose and iter_idx % 100 == 0:
                logger.info(f"  Iter {iter_idx}: diff = {uDiff:.2e}")
            
            if uDiff < self.tol and iter_idx > 2:
                converged = True
                if verbose:
                    logger.info(f"  Converged at iteration {iter_idx}")
                break
        
        # Extract final results
        final = n_iter % 2
        u_hat = u_hat_plus[final]
        omega = omega_plus[final]
        
        # Sort by frequency
        if omega_shape == 'dynamic':
            sort_idx = np.argsort(np.mean(omega, axis=1))
            omega = omega[sort_idx, :]
        else:
            sort_idx = np.argsort(omega)
            omega = omega[sort_idx]
        
        u_hat = u_hat[:, :, sort_idx, :]
        
        if not is_stft:
            u_hat = u_hat[:, :, :, 0]
        
        return u_hat, omega, n_iter, converged


def _as_row(signal: np.ndarray) -> np.ndarray:
    """(T,) signal -> (1, T) row used internally by the ADMM kernel."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"single-channel signal of shape (T,) expected, got {signal.shape}")
    return signal.reshape(1, -1)


# =============================================================================
# VMD Implementation
# =============================================================================

class VMD(VMDBase):
    """
    Variational Mode Decomposition (single channel).
    
    Decomposes a signal into K band-limited modes by ADMM on the
    mirror-padded one-sided spectrum.
    
    Parameters
    ----------
    K : int
        Number of modes to extract
    alpha : float, default=50
        Bandwidth constraint (higher = narrower)
    tau : float, default=1e-5
        Lagrangian multiplier step
    tol : float, default=1e-9
        Convergence tolerance
    maxiters : int, default=500
        Maximum iterations
    
    Examples
    --------
    >>> result = VMD(K=5)(x, fs=256)
    >>> result.modes.shape      # (5, len(x))
    """
    
    def decompose(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        omega_init: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> VMDResult:
        """
        Decompose signal into K modes.
        
        Parameters
        ----------
        signal : ndarray (T,)
            Input signal
        fs : float
            Sampling frequency in Hz
        omega_init : ndarray (K,), optional
            Initial center frequencies
        verbose : bool
            Print progress
        
        Returns
        -------
        VMDResult
        """
        signal = _as_row(signal)
        T = signal.shape[1]
        
        # Mirror padding
        pad_len = T // 2
        signal_padded = np.concatenate([
            signal[:, pad_len:0:-1],
            signal,
            signal[:, -2:-pad_len-2:-1]
        ], axis=1)
        
        T_padded = signal_padded.shape[1]
        
        # FFT
        f_hat = rfft(signal_padded, axis=1)
        F = f_hat.shape[1]
        freqs = np.arange(F) / F
        
        # ADMM optimization
        u_hat, omega, n_iter, converged = self._admm_optimize(
            f_hat, freqs, omega_init, dynamic=False, verbose=verbose
        )
        
        # Inverse FFT
        u_hat = np.transpose(u_hat, (2, 0, 1))  # (K, 1, F)
        modes_padded = irfft(u_hat, n=T_padded, axis=2).real
        
        # Remove padding
        modes = modes_padded[:, :, pad_len:-pad_len]
        
        return VMDResult(
            modes=modes[:, 0, :],
            center_freqs=omega,
            fs=fs,
            n_iterations=n_iter,
            converged=converged,
            method='VMD'
        )


# =============================================================================
# STVMD Implementation
# =============================================================================

class STVMD(VMDBase):
    """
    Short-Time Variational Mode Decomposition.
    
    Extends VMD to non-stationary signals using STFT.
    
    Parameters
    ----------
    K : int
        Number of modes
    n_fft : int, default=64
        STFT window length
    alpha : float, default=50
        Bandwidth constraint
    tau : float, default=1e-5
        Lagrangian multiplier step
    tol : float, default=1e-9
        Convergence tolerance
    maxiters : int, default=1000
        Maximum iterations
    window : str, default='hamming'
        Window function
    
    Examples
    --------
    >>> stvmd = STVMD(K=5, n_fft=64)
    >>> result = stvmd(x, fs=256, dynamic=True)
    """
    
    def __init__(
        self,
        K: int,
        n_fft: int = 64,
        alpha: float = 50,
        tau: float = 1e-5,
        tol: float = 1e-9,
        maxiters: int = 1000,
        window: str = 'hamming'
    ):
        super().__init__(K, alpha, tau, tol, maxiters)
        self.n_fft = n_fft
        self.hop_len = 1
        self.window = scipy.signal.get_window(window, n_fft, fftbins=False)
        self.pad_left = (n_fft - 1) // 2 + ((n_fft - 1) % 2)
        self.pad_right = (n_fft - 1) // 2
    
    def decompose(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        dynamic: bool = False,
        omega_init: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> VMDResult:
        """
        Decompose signal using STVMD.
        
        Parameters
        ----------
        signal : ndarray (T,)
            Input signal
        fs : float
            Sampling frequency
        dynamic : bool, default=False
            Time-varying frequencies
        omega_init : ndarray, optional
            Initial frequencies
        verbose : bool
            Print progress
        
        Returns
        -------
        VMDResult
        """
        signal = _as_row(signal)
        T = signal.shape[1]
        
        # Pad and STFT
        signal_padded = np.pad(signal, [[0, 0], [self.pad_left, self.pad_right]], 
                               mode='reflect')
        f_hat = self._stft(signal_padded)
        
        F = f_hat.shape[1]
        freqs = np.arange(F) / F
        
        # ADMM
        u_hat, omega, n_iter, converged = self._admm_optimize(
            f_hat, freqs, omega_init, dynamic=dynamic, verbose=verbose
        )
        
        # Inverse STFT
        modes = self._istft(u_hat, T)
        
        # Residual
        reconstructed = np.sum(modes, axis=0)
        residual = np.mean((signal - reconstructed[:, :T]) ** 2)
        
        method = 'STVMD-dynamic' if dynamic else 'STVMD'
        
        return VMDResult(
            modes=modes[:, 0, :T],
            center_freqs=omega,
            fs=fs,
            n_iterations=n_iter,
            converged=converged,
            method=method,
            n_fft=self.n_fft,
            residual=residual
        )
    
    def _stft(self, signal: np.ndarray) -> np.ndarray:
        """Compute STFT with hop_len=1."""
        C, T = signal.shape
        N = T - self.n_fft + 1
        
        frames = np.zeros((C, self.n_fft, N))
        for i in range(N):
            frames[:, :, i] = signal[:, i:i+self.n_fft]
        
        frames *= self.window.reshape(1, -1, 1)
        f_hat = rfft(frames, axis=1)
        
        return f_hat
    
    def _istft(self, u_hat: np.ndarray, T_orig: int) -> np.ndarray:
        """Compute inverse STFT with overlap-add."""
        C, F, K, N = u_hat.shape
        T_padded = T_orig + self.pad_left + self.pad_right
        
        modes = np.zeros((K, C, T_padded))
        
        # Window normalization
        wn = np.zeros(T_padded)
        w_sq = self.window ** 2
        for i in range(N):
            wn[i:i+self.n_fft] += w_sq
        wn[wn < 1e-10] = 1.0
        
        for k in range(K):
            frames = irfft(u_hat[:, :, k, :], n=self.n_fft, axis=1).real
            
            mode_k = np.zeros((C, T_padded))
            for i in range(N):
                mode_k[:, i:i+self.n_fft] += frames[:, :, i] * self.window
            
            mode_k /= wn
            modes[k] = mode_k
        
        # Remove padding
        if self.pad_right > 0:
            modes = modes[:, :, self.pad_left:-self.pad_right]
        else:
            modes = modes[:, :, self.pad_left:]
        
        return modes


# =============================================================================
# Quick Functions
# =============================================================================

def vmd(signal: np.ndarray, K: int, fs: float = 1.0, **kwargs) -> VMDResult:
    """
    Quick VMD decomposition.
    
    Parameters
    ----------
    signal : ndarray (T,)
    K : int
        Number of modes
    fs : float
        Sampling frequency
    **kwargs
        Additional arguments to VMD
    
    Returns
    -------
    VMDResult
    """
    return VMD(K=K, **kwargs)(signal, fs=fs)


def stvmd(signal: np.ndarray, K: int, fs: float = 1.0, 
          n_fft: int = 64, dynamic: bool = False, **kwargs) -> VMDResult:
    """
    Quick STVMD decomposition.
    
    Parameters
    ----------
    signal : ndarray (T,)
    K : int
        Number of modes
    fs : float
        Sampling frequency
    n_fft : int
        STFT window size
    dynamic : bool
        Time-varying frequencies
    **kwargs
        Additional arguments to STVMD
    
    Returns
    -------
    VMDResult
    """
    return STVMD(K=K, n_fft=n_fft, **kwargs)(signal, fs=fs, dynamic=dynamic)
