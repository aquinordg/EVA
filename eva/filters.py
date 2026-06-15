"""
Signal processing filters for EEG preprocessing.

Each filter class exposes a single ``apply(data, sfreq)`` method:

    data  : np.ndarray, shape (n_channels, n_samples), in SI units (volts)
    sfreq : float, sampling frequency in Hz

All operations are stateless; instances can be reused across recordings.
Filters are designed to be chained in sequence:

    DCDetrend -> ButterworthFilter -> NotchFilter -> AverageReference -> SoftClipper

This ordering is scientifically motivated:
  1. DC removal stabilises downstream IIR filter initialisation.
  2. Bandpass applied before notch narrows the spectrum first.
  3. Average reference after filtering avoids propagating channel artefacts.
  4. Soft clipping last to attenuate any residual transient amplitudes.

Amplitude units
---------------
MNE returns data in volts (SI).  All amplitude thresholds in this module
must therefore be specified in **volts** (e.g. 100 µV = 100e-6 V).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# 1. DC offset removal
# ---------------------------------------------------------------------------

class DCDetrend:
    """
    Remove the DC offset from each channel by subtracting its temporal mean.

    Must be applied before any IIR or FIR filter to prevent spectral leakage
    and filter-initialisation transients caused by a non-zero mean.
    """

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        return data - data.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Zero-phase Butterworth bandpass / highpass / lowpass
# ---------------------------------------------------------------------------

@dataclass
class ButterworthFilter:
    """
    Zero-phase Butterworth filter (highpass, lowpass, or bandpass).

    Uses second-order sections (SOS) internally via
    ``scipy.signal.sosfiltfilt``, which is numerically far more stable than
    the transfer-function ``(b, a)`` form — especially when the cutoff
    frequency is very low relative to the sampling rate (e.g. 0.1 Hz at
    1024 Hz, normalised frequency ≈ 0.0002).  The ``(b, a)`` form produces
    catastrophic coefficient overflow in those cases; SOS avoids it entirely.

    ``sosfiltfilt`` applies forward-backward filtering (zero phase) exactly
    as ``filtfilt`` does.

    Parameters
    ----------
    l_freq : float or None
        High-pass cutoff (Hz).  ``None`` disables the high-pass stage.
    h_freq : float or None
        Low-pass cutoff (Hz).  ``None`` disables the low-pass stage.
    order : int
        Filter order for each direction (effective order = 2 × order after
        sosfiltfilt).  Order 4 gives a roll-off of −80 dB/decade per
        direction.
    """

    l_freq: Optional[float] = 1.0
    h_freq: Optional[float] = 40.0
    order: int = 4

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        nyq = sfreq / 2.0
        out = data.copy()

        if self.l_freq is not None and self.h_freq is not None:
            sos = sp_signal.butter(
                self.order,
                [self.l_freq / nyq, self.h_freq / nyq],
                btype="bandpass",
                output="sos",
            )
        elif self.l_freq is not None:
            sos = sp_signal.butter(
                self.order, self.l_freq / nyq, btype="high", output="sos"
            )
        elif self.h_freq is not None:
            sos = sp_signal.butter(
                self.order, self.h_freq / nyq, btype="low", output="sos"
            )
        else:
            return out

        for ch in range(data.shape[0]):
            out[ch] = sp_signal.sosfiltfilt(sos, data[ch])
        return out


# ---------------------------------------------------------------------------
# 3. Notch filter (power-line interference)
# ---------------------------------------------------------------------------

@dataclass
class NotchFilter:
    """
    Zero-phase IIR notch filter for power-line harmonic removal.

    The notch bandwidth is ``freq / quality_factor`` Hz.  A Q of 30 at
    60 Hz yields a −3 dB bandwidth of 2 Hz, which is narrow enough to
    preserve adjacent EEG frequencies.

    Parameters
    ----------
    freq : float
        Notch centre frequency (Hz).  Use 60 Hz (Americas/Asia) or
        50 Hz (Europe/Africa/most of Asia).
    quality_factor : float
        Q-factor controlling the notch width.
    """

    freq: float = 60.0
    quality_factor: float = 30.0

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        b, a = sp_signal.iirnotch(self.freq, self.quality_factor, sfreq)
        sos  = sp_signal.tf2sos(b, a)
        out  = np.empty_like(data)
        for ch in range(data.shape[0]):
            out[ch] = sp_signal.sosfiltfilt(sos, data[ch])
        return out


# ---------------------------------------------------------------------------
# 4. Average reference (Common Average Reference — CAR)
# ---------------------------------------------------------------------------

class AverageReference:
    """
    Re-reference to the instantaneous mean across all retained channels.

    CAR subtracts the spatial mean at each time sample, attenuating
    noise and artefacts common to all electrodes (e.g. slow drifts,
    movement artefacts) while preserving channel-specific activity.
    Applied after spectral filtering to avoid propagating channel-level
    artefacts into the reference estimate.
    """

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        return data - data.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# 5. Soft clipping (smooth amplitude limiter)
# ---------------------------------------------------------------------------

@dataclass
class SoftClipper:
    """
    Piecewise smooth amplitude limiter based on hyperbolic tangent saturation.

    Amplitudes within ``[-threshold, +threshold]`` pass through **unchanged**
    (linear region).  Amplitudes beyond the threshold are continuously
    compressed toward a soft asymptote rather than hard-clipped to zero,
    preventing the spectral discontinuities and artificial flat regions that
    hard zeroing introduces.

    Transfer function
    -----------------
    For |x| ≤ threshold:
        y = x                              (identity — signal preserved)

    For |x| > threshold:
        excess = |x| − threshold
        y = sign(x) × (threshold + threshold × tanh(excess / threshold))

    Properties
    ----------
    - C¹ continuous at ±threshold (derivative = 1 from both sides).
    - Monotonically increasing; relative ordering of amplitudes is preserved.
    - Soft asymptote at ≈ 2 × threshold as |x| -> ∞.

    Parameters
    ----------
    threshold : float
        Amplitude ceiling in **volts** (SI units as returned by MNE).
        Example: 100 µV -> ``threshold=100e-6``.
    """

    threshold: float = 100e-6  # 100 µV in volts

    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        abs_x = np.abs(data)
        excess = np.maximum(abs_x - self.threshold, 0.0)
        compressed_excess = self.threshold * np.tanh(excess / self.threshold)
        return np.sign(data) * (np.minimum(abs_x, self.threshold) + compressed_excess)
