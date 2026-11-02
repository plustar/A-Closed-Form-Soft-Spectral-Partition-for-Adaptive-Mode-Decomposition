# experiments/common/plot_utils.py
"""
Shared plotting utilities for the paper experiments.

Every function takes (ax, result, fs) with ``result`` a softdamd Decomposition.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════
# Shared config
# ═══════════════════════════════════════════════════════════════════

CMAP_DEFAULT = 'viridis'
MARKER_BASE_SIZE = 120
MARKER_COLOR = '#B22222'
MARKER_ALPHA = 0.4



def setup_font_config(fontsize: int = 10):
    """Configure matplotlib to use Times New Roman (TeX Gyre Termes).

    TeX Gyre Termes is metrically compatible with Times New Roman and
    is used when the system does not have Times New Roman installed.
    All paper figures call this function first.
    """
    import matplotlib.font_manager as fm
    # Prefer actual Times New Roman if available, else TeX Gyre Termes
    available = {f.name for f in fm.fontManager.ttflist}
    if 'Times New Roman' in available:
        serif_font = 'Times New Roman'
    elif 'TeX Gyre Termes' in available:
        serif_font = 'TeX Gyre Termes'
    else:
        serif_font = 'DejaVu Serif'

    matplotlib.rcParams.update({
        # Font family
        'font.family'       : 'serif',
        'font.serif'        : [serif_font, 'Times New Roman',
                               'TeX Gyre Termes', 'DejaVu Serif'],
        # Sizes
        'font.size'         : fontsize,
        'axes.titlesize'    : fontsize,
        'axes.labelsize'    : fontsize,
        'xtick.labelsize'   : fontsize - 1,
        'ytick.labelsize'   : fontsize - 1,
        'legend.fontsize'   : fontsize - 2,
        # Math font — use stix for Times-compatible math
        'mathtext.fontset'  : 'stix',
        'mathtext.rm'       : serif_font,
        # Figure quality
        'figure.dpi'        : 150,
        'savefig.dpi'       : 300,
        'savefig.bbox'      : 'tight',
        'pdf.fonttype'      : 42,   # embed fonts in PDF
        'ps.fonttype'       : 42,
    })


# ═══════════════════════════════════════════════════════════════════
# Axis helpers (result is a softdamd Decomposition)
# ═══════════════════════════════════════════════════════════════════

def _freq_axis(result, fs):
    """Frequency axis in Hz (linear for the STFT family, log-spaced for SSQ-CWT)."""
    return np.asarray(result.tfmap.freqs_hz)


def _center_to_hz(center_val, fs):
    """Convert center from [0, 0.5] cycles/sample to Hz."""
    return float(center_val) * fs


def _is_log_freq(result):
    return result.tfmap.scale == 'nonlinear'


def _apply_freq_scale(ax, result):
    """Log y-scale for the wavelet family."""
    if _is_log_freq(result):
        ax.set_yscale('log')


def _hop_length(result):
    return result.tfmap.hop


# ═══════════════════════════════════════════════════════════════════
# plot_tf_spectrogram
# ═══════════════════════════════════════════════════════════════════

def plot_tf_spectrogram(ax, result, fs, *, title=None,
                        cmap=CMAP_DEFAULT, add_colorbar=True):
    """TF energy as pcolormesh."""
    tf_energy = np.abs(result.tfmap.coef)  # (F, T)
    F, T = tf_energy.shape
    hop = _hop_length(result)
    time_points = np.arange(T) * hop / fs
    freq_points = _freq_axis(result, fs)

    im = ax.pcolormesh(time_points, freq_points, tf_energy,
                       cmap=cmap, shading='gouraud', rasterized=True)
    if add_colorbar:
        plt.colorbar(im, ax=ax, label='Energy')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    _apply_freq_scale(ax, result)
# ═══════════════════════════════════════════════════════════════════

def plot_tf_centers(ax, result, fs, *, title=None, mode='hard'):
    """Scatter of detected mode centres; dot size ∝ the energy the ``mode`` mask
    allocates to each centre, E_k = Σ_ω s_k(ω)|𝒯(ω,t)|²."""
    p = result.partition
    tf_map = result.tfmap.coef                        # (F, T)
    n_modes = np.asarray(p.n_modes)
    T_frames = len(n_modes)
    F = tf_map.shape[0]
    hop = _hop_length(result)
    freq_points = _freq_axis(result, fs)
    offsets = np.concatenate([[0], np.cumsum(n_modes)]).astype(np.int64)

    total = int(n_modes.sum())
    times = np.empty(total)
    freqs_hz = np.empty(total)
    energies = np.empty(total)
    for t in range(T_frames):
        sl = slice(int(offsets[t]), int(offsets[t + 1]))
        times[sl] = t * hop / fs
        freqs_hz[sl] = p.centers[sl] * fs
        if n_modes[t]:
            energies[sl] = result.energy(t, mode)

    if total == 0:
        if title is not None:
            ax.set_title(title)
        return

    max_e = energies.max()
    normalized = energies / max_e if max_e > 0 else np.ones(total)


    ax.scatter(times, freqs_hz,
               s=normalized * MARKER_BASE_SIZE,
               c=MARKER_COLOR,
               alpha=MARKER_ALPHA,
               edgecolors='none',
               rasterized=True)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    time_points = np.arange(tf_map.shape[1]) * hop / fs
    ax.set_xlim(time_points[0], time_points[-1])
    ax.set_ylim(freq_points[0], freq_points[-1])
    _apply_freq_scale(ax, result)


# ═══════════════════════════════════════════════════════════════════
# Energy-stratified scatter
# ═══════════════════════════════════════════════════════════════════

ENERGY_COLORS = ['#D0D0D0', '#A8D8EA', '#4A90D9', '#E8873A', '#B22222']

def plot_tf_energy_stratified(ax, result, fs, *, title=None, n_quantiles=5, mode='soft'):
    """Scatter of mode centres coloured by quintile of the energy the ``mode`` mask
    allocates to them (soft: Lorentzian weights at the result's α)."""
    p = result.partition
    tf_map = result.tfmap.coef                        # (F, T)
    n_modes_arr = np.asarray(p.n_modes)
    T_frames = len(n_modes_arr)
    F = tf_map.shape[0]
    hop = _hop_length(result)
    freq_points = _freq_axis(result, fs)
    offsets = np.concatenate([[0], np.cumsum(n_modes_arr)]).astype(np.int64)

    total = int(n_modes_arr.sum())
    times = np.empty(total)
    freqs_hz = np.empty(total)
    energies = np.empty(total)
    for t in range(T_frames):
        sl = slice(int(offsets[t]), int(offsets[t + 1]))
        times[sl] = t * hop / fs
        freqs_hz[sl] = p.centers[sl] * fs
        if n_modes_arr[t]:
            energies[sl] = result.energy(t, mode)

    if total == 0:
        if title is not None:
            ax.set_title(title)
        return

    max_e = energies.max()
    normalized = energies / max_e if max_e > 0 else np.ones(total)
    scatter_sizes = normalized * MARKER_BASE_SIZE

    pos = energies[energies > 0]
    if len(pos) == 0:
        return
    edges = np.percentile(pos, np.linspace(0, 100, n_quantiles + 1))
    colors = ENERGY_COLORS[:n_quantiles]


    for qi in range(n_quantiles):
        lo = edges[qi]
        hi = edges[qi + 1] if qi < n_quantiles - 1 else np.inf
        mask = (energies >= lo) & ((energies < hi) | (qi == n_quantiles - 1))
        if mask.sum() == 0:
            continue
        ax.scatter(times[mask], freqs_hz[mask], s=scatter_sizes[mask],
                   c=colors[qi], alpha=MARKER_ALPHA,
                   edgecolors='none', rasterized=True)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    time_points = np.arange(tf_map.shape[1]) * hop / fs
    ax.set_xlim(time_points[0], time_points[-1])
    ax.set_ylim(freq_points[0], freq_points[-1])
    _apply_freq_scale(ax, result)
