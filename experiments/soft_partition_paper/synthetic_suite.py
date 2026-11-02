# experiments/soft_partition_paper/synthetic_suite.py
"""
Four synthetic multi-component signals, used wherever this repository needs a
non-trivial time-varying spectrum.

Each generator returns ``(signal, fs, transform)`` -- the same contract the pipeline
expects -- and is fully determined by its seed, so every number computed from the suite
is reproducible from this file alone. Nothing here is a model of, or fitted to, any
measured recording; the four are chosen because they exercise different parts of the
detector and of the partition:

  harmonic_switch     a harmonic stack whose fundamental steps mid-record.  Many
                      closely spaced low-frequency components on a logarithmic axis;
                      the mode count changes at the step.
  crossing_chirps     two chirps that cross, with a stationary tone nearby.  Centres
                      merge and separate again, so the Voronoi boundaries sweep across
                      the band and the two allocations differ most where they meet.
  intermittent_tones  three narrow-band components that switch on and off at different
                      times, one with a slow wobble.  The mode count steps between one
                      and three, and the Voronoi boundaries appear and vanish with it.
  moving_formants     a glottal-type pulse train shaped by three resonances that move,
                      with two silences.  Dense harmonics under a moving envelope, and
                      frames with no signal at all.

Sampling rates and lengths differ across the four on purpose: one frequency bin is then
a different physical width in each, which is exactly the caveat the boundary
measurement has to be read with.
"""

import numpy as np

__all__ = ['harmonic_switch', 'crossing_chirps', 'intermittent_tones',
           'moving_formants', 'SUITE']


def _pink(rng, n, scale):
    """Noise with a 1/f magnitude spectrum, unit variance before scaling."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(spec.size)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return scale * x / x.std()


def harmonic_switch(seed=0):
    """10 Hz fundamental with two harmonics for 5 s, then 13 Hz for 5 s.

    fs = 250 Hz, 2500 samples.  Each component is amplitude-modulated at its own slow
    rate, so the energy ordering of the harmonics changes during the record.
    """
    fs, n = 250.0, 2500
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    half = n // 2
    f0 = np.where(np.arange(n) < half, 10.0, 13.0)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    x = np.zeros(n)
    for k, (amp, mod_hz, ph) in enumerate([(1.00, 0.17, 0.0),
                                           (0.45, 0.23, 1.1),
                                           (0.20, 0.31, 2.3)], start=1):
        env = amp * (1.0 + 0.3 * np.sin(2 * np.pi * mod_hz * t + ph))
        x += env * np.cos((k * phase) + ph)
    x += _pink(rng, n, 0.12)
    return x.astype(np.float64), fs, 'ssq_cwt'


def crossing_chirps(seed=1):
    """Two linear chirps crossing at mid-record, plus a stationary tone at 62 Hz.

    fs = 220 Hz, 660 samples.  The chirps sweep 25 -> 85 Hz and 85 -> 25 Hz, so they
    coincide at 55 Hz halfway through and the detector sees three modes, then two,
    then three again.
    """
    fs, n = 220.0, 660
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    T = t[-1]
    up = 25.0 + (85.0 - 25.0) * t / T
    dn = 85.0 - (85.0 - 25.0) * t / T
    x = (1.00 * np.cos(2 * np.pi * np.cumsum(up) / fs)
         + 0.80 * np.cos(2 * np.pi * np.cumsum(dn) / fs + 0.7)
         + 0.50 * np.cos(2 * np.pi * 62.0 * t + 1.9))
    x += rng.standard_normal(n) * 0.05
    return x.astype(np.float64), fs, 'ssq_stft'


def intermittent_tones(seed=2):
    """Three narrow-band components that switch on and off at different times.

    fs = 250 Hz, 1000 samples.  A 28 Hz component with a slow +-3 Hz wobble runs
    throughout; a 64 Hz component is present in two bursts (0.4-1.6 s and 2.6-3.6 s)
    with a short fade at each edge; a 105 Hz component ramps up from 1.0 s and stops at
    3.0 s.  The mode count a detector should find steps between one and three.
    """
    fs, n = 250.0, 1000
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs

    def gate(t0, t1, fade=0.04):
        g = np.clip((t - t0) / fade, 0, 1) * np.clip((t1 - t) / fade, 0, 1)
        return g

    f_low = 28.0 + 3.0 * np.sin(2 * np.pi * 0.25 * t)
    x = 1.0 * np.cos(2 * np.pi * np.cumsum(f_low) / fs)
    x += 0.8 * (gate(0.4, 1.6) + gate(2.6, 3.6)) * np.cos(2 * np.pi * 64.0 * t + 0.9)
    ramp = np.clip((t - 1.0) / 1.0, 0, 1) * gate(1.0, 3.0, fade=0.02)
    x += 0.7 * ramp * np.cos(2 * np.pi * 105.0 * t + 2.1)
    x += rng.standard_normal(n) * 0.03
    return x.astype(np.float64), fs, 'ssq_stft'


def moving_formants(seed=3):
    """A pulse train whose fundamental glides, shaped by three moving resonances.

    fs = 500 Hz, 1000 samples.  Two silences split the record into three segments; the
    fundamental glides 18 -> 26 Hz and the three resonance centres move with it, so the
    harmonics stay dense while the envelope that selects them travels upwards.
    """
    fs, n = 500.0, 1000
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    T = t[-1]
    f0 = 18.0 + (26.0 - 18.0) * t / T
    phase = 2 * np.pi * np.cumsum(f0) / fs
    centres = [(60.0, 95.0), (130.0, 175.0), (200.0, 230.0)]
    gains = (1.0, 0.6, 0.35)
    x = np.zeros(n)
    for h in range(1, 13):                       # harmonics inside the band
        fh = h * f0
        env = np.zeros(n)
        for (c0, c1), g in zip(centres, gains):
            c = c0 + (c1 - c0) * t / T
            env += g * np.exp(-0.5 * ((fh - c) / 28.0) ** 2)
        x += env * np.cos(h * phase + 0.3 * h)
    gate = np.ones(n)
    gate[330:400] = 0.0                          # two silences
    gate[700:760] = 0.0
    x *= gate
    x += rng.standard_normal(n) * 0.02
    return x.astype(np.float64), fs, 'ssq_stft'


# (generator, transform title for the figure, label)
SUITE = [
    (harmonic_switch,    'SSQ-CWT',  'Harmonic stack'),
    (crossing_chirps,    'SSQ-STFT', 'Crossing chirps'),
    (intermittent_tones, 'SSQ-STFT', 'Intermittent tones'),
    (moving_formants,    'SSQ-STFT', 'Moving formants'),
]


if __name__ == '__main__':
    for fn, _title, label in SUITE:
        sig, fs, tf = fn()
        print(f'{label:20s} fs={fs:6.1f}  N={sig.size:5d}  '
              f'duration={sig.size / fs:5.2f} s  rms={sig.std():.3f}  {tf}')
