"""
Signal quality metrics for per-channel and per-recording EEG evaluation.

All functions operate on plain NumPy arrays, keeping them independent of
MNE objects.  The canonical data shape is (n_channels, n_samples).

Metrics
-------
- MAE / MSE              : error magnitude between raw and processed signals
- SNR                    : signal-to-noise ratio in dB (variance-based, informational)
- ADC clipping fraction  : fraction of samples stuck at the exact min/max value
- PaLOSi                 : Parallel LOg Spectra index — recording-level
                           spectral homogeneity (scalar, [0, 1])
- Log-Spectra Deviation  : per-channel deviation from ensemble median log-PSD
- Spectral entropy       : normalised frequency-domain entropy
- Hjorth params          : activity, mobility, complexity — time-domain

References
----------
[1] PaLOSi: Hu et al. (2025) NeuroImage https://doi.org/10.1016/j.neuroimage.2025.121247
[2] Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def mae(raw: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error per channel.

    Parameters
    ----------
    raw, processed : (n_channels, n_samples)

    Returns
    -------
    (n_channels,) array in the same units as the input.
    """
    return np.mean(np.abs(raw - processed), axis=-1)


def mse(raw: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Mean Squared Error per channel.

    Parameters
    ----------
    raw, processed : (n_channels, n_samples)

    Returns
    -------
    (n_channels,) array in squared input units.
    """
    return np.mean((raw - processed) ** 2, axis=-1)


def snr_db(raw: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Signal-to-Noise Ratio per channel, in decibels.

    Defined as:

        SNR = 10 x log10(Var(raw) / Var(noise))

    where ``noise = raw - processed`` is the signal component removed
    by the filter chain.  A high positive value means the filter removed
    a small fraction of the signal power (low distortion).

    Parameters
    ----------
    raw, processed : (n_channels, n_samples)
        ``raw`` should be the DC-detrended signal (before bandpass/notch),
        not the unprocessed original, to avoid DC contaminating the variance.

    Returns
    -------
    (n_channels,) array in dB.
    """
    noise = raw - processed
    return 10.0 * np.log10(
        (np.var(raw, axis=-1) + 1e-30) / (np.var(noise, axis=-1) + 1e-30)
    )


def adc_clipping_fraction(data: np.ndarray) -> np.ndarray:
    """
    Per-channel fraction of samples at the exact minimum or maximum value.

    When an amplifier's voltage range is exceeded, the ADC saturates and
    multiple consecutive samples are stored at the same extreme value.
    A fraction above ~0.1% is a reliable indicator of ADC saturation
    during recording, regardless of amplifier model or voltage range.

    Parameters
    ----------
    data : (n_channels, n_samples)
        Raw or DC-detrended EEG signal. DC detrending preserves the
        clipping pattern because all stuck samples shift by the same offset.

    Returns
    -------
    (n_channels,) float array — fraction in [0, 1]; higher means more clipping.
    """
    n = data.shape[-1]
    result = np.empty(data.shape[0])
    for i in range(data.shape[0]):
        ch = data[i]
        min_val, max_val = ch.min(), ch.max()
        result[i] = int(np.sum((ch == min_val) | (ch == max_val))) / n
    return result


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------

def compute_psd(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch power spectral density estimate for all channels.

    Parameters
    ----------
    data    : (n_channels, n_samples)
    sfreq   : sampling frequency (Hz)
    nperseg : Welch segment length (samples)

    Returns
    -------
    freqs : (n_freqs,) in Hz
    psd   : (n_channels, n_freqs) in V^2/Hz
    """
    freqs, psd = sp_signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd


def palosi(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> float:
    """
    PaLOSi (Parallel LOg Spectra index) for a recording — scalar in [0, 1].

    Measures the proportion of total cross-spectral power captured by the
    dominant spatial component at each frequency, summed across the analysis
    band.  A value near 1 means all channels are spectrally homogeneous
    (dominated by a single spatial pattern), which is a sign of dominant
    artefact or over-preprocessing.  Well-preprocessed, spatially diverse
    EEG recordings typically yield values well below 0.5.

    This implements the eigenvalue-proportion approach from Hu et al. (2025):

        PaLOSi = sum_f [ lambda_1(f) ] / total_power

    where lambda_1(f) is the largest eigenvalue of the cross-spectral matrix
    at frequency f, and total_power = sum_f trace(C(f)).

    Note: uses per-frequency eigendecomposition rather than CPC for practical
    tractability without external toolboxes; the measured quantity is the same.

    Parameters
    ----------
    data    : (n_channels, n_samples) — processed signal; requires >= 2 channels
    sfreq   : sampling frequency (Hz)
    nperseg : FFT segment length (samples)
    fmin    : lower frequency bound (Hz); should match the high-pass cutoff
    fmax    : upper frequency bound (Hz); should match the low-pass cutoff

    Returns
    -------
    float in [0, 1] — higher is worse.

    References
    ----------
    [1] Hu et al. (2025) NeuroImage https://doi.org/10.1016/j.neuroimage.2025.121247
    """
    n_ch, n_samp = data.shape
    if n_ch < 2:
        return 0.0

    step = nperseg // 2
    win = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    n_freq_sel = int(freq_mask.sum())
    if n_freq_sel == 0:
        return 0.0

    # Build all windowed segments in one shot using advanced indexing:
    # seg_idx[s, k] = start_of_segment_s + k  → shape (n_segs, nperseg)
    n_segs = (n_samp - nperseg) // step + 1
    seg_idx = step * np.arange(n_segs)[:, None] + np.arange(nperseg)[None, :]
    # segments: (n_ch, n_segs, nperseg) — applies Hann window in-place
    segments = data[:, seg_idx] * win[None, None, :]

    # Batch FFT over last axis → (n_ch, n_segs, n_freq_sel)
    X = np.fft.rfft(segments, axis=-1)[:, :, freq_mask]

    # Rearrange to (n_freq_sel, n_ch, n_segs) for batched matmul
    X = X.transpose(2, 0, 1)

    # Batched cross-spectral matrix: C[f] = X[f] @ X[f].conj().T / n_segs
    # Shape: (n_freq_sel, n_ch, n_ch) — Hermitian at each frequency
    C = np.matmul(X, X.conj().swapaxes(-1, -2)) / n_segs

    # Total power = sum of diagonal (auto-spectra) across selected frequencies
    diag_idx = np.arange(n_ch)
    ssd = float(np.sum(C[:, diag_idx, diag_idx].real))
    if ssd < 1e-30:
        return 0.0

    # Sum of dominant eigenvalue across frequencies (C is Hermitian -> real eigenvalues)
    first_eig_sum = sum(
        float(np.linalg.eigvalsh(C[f])[-1])   # eigvalsh returns ascending order
        for f in range(n_freq_sel)
    )
    return first_eig_sum / ssd


def log_spectra_deviation(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
) -> np.ndarray:
    """
    Log-Spectra Deviation per channel.

    Measures how far each channel's log-power spectrum deviates from the
    median log-power spectrum across the channel ensemble.  A high score
    indicates the channel is a spectral outlier relative to its neighbours,
    which is consistent with artefact contamination or electrode malfunction.

    Definition
    ----------
        LSD(ch) = mean_f |log PSD_ch(f) - median_ch log PSD(f)|

    A score of 0 means the channel is identical to the ensemble median;
    values above ~2 warrant manual inspection.

    Note: this is a simplified per-channel variant inspired by the spectral
    parallelism concept in PaLOSi.  For the recording-level homogeneity
    index, use :func:`palosi`.

    Parameters
    ----------
    data    : (n_channels, n_samples) — processed signal; requires >= 2 channels
    sfreq   : sampling frequency (Hz)
    nperseg : Welch segment length

    Returns
    -------
    (n_channels,) array — lower is better.
    """
    _, psd = compute_psd(data, sfreq, nperseg)
    log_psd = np.log(psd + 1e-30)
    median_log = np.median(log_psd, axis=0)
    return np.mean(np.abs(log_psd - median_log), axis=-1)


def spectral_entropy(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
) -> np.ndarray:
    """
    Normalised spectral entropy per channel (range [0, 1]).

    A value near 0 indicates power concentrated in narrow bands, which may
    signal narrow-band artefacts (e.g. residual line noise) or electrode
    bridging.  A value near 1 corresponds to a spectrally flat (white-noise)
    distribution.

    Parameters
    ----------
    data    : (n_channels, n_samples)
    sfreq   : sampling frequency (Hz)
    nperseg : Welch segment length

    Returns
    -------
    (n_channels,) array.
    """
    _, psd = compute_psd(data, sfreq, nperseg)
    p = psd / (psd.sum(axis=-1, keepdims=True) + 1e-30)
    entropy = -np.sum(p * np.log(p + 1e-30), axis=-1)
    return entropy / np.log(psd.shape[-1])


# ---------------------------------------------------------------------------
# Hjorth parameters
# ---------------------------------------------------------------------------

def hjorth_parameters(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hjorth activity, mobility, and complexity per channel.

    These three time-domain features characterise signal power, mean
    frequency, and bandwidth without requiring a Fourier transform.

    Definitions
    -----------
    Activity   = Var(x)                             signal power proxy
    Mobility   = std(dx/dt) / std(x)               mean frequency proxy
    Complexity = Mobility(d^2x/dt^2) / Mobility(x) bandwidth / shape proxy

    Parameters
    ----------
    data : (n_channels, n_samples)

    Returns
    -------
    activity   : (n_channels,)
    mobility   : (n_channels,)
    complexity : (n_channels,)

    References
    ----------
    [2] Hjorth (1970), Electroencephalography and Clinical Neurophysiology.
    """
    dx = np.diff(data, axis=-1)
    d2x = np.diff(dx, axis=-1)

    activity = np.var(data, axis=-1)
    mobility = np.std(dx, axis=-1) / (np.std(data, axis=-1) + 1e-30)
    mob_dx = np.std(d2x, axis=-1) / (np.std(dx, axis=-1) + 1e-30)
    complexity = mob_dx / (mobility + 1e-30)

    return activity, mobility, complexity


# ---------------------------------------------------------------------------
# Multi-criteria channel diagnostics
# ---------------------------------------------------------------------------

@dataclass
class QualityConfig:
    """
    Multi-criterion quality gate for individual EEG channels.

    Each criterion produces a boolean flag; the overall channel status
    is determined by how many flags are raised:

    ======== ============================================================
    Status   Criteria
    ======== ============================================================
    good     No flags raised.
    warning  Exactly one flag raised — channel is borderline.
    bad      Two or more flags raised — channel likely artefact-dominated.
    ======== ============================================================

    All amplitude parameters are in **volts** (MNE SI units).

    Parameters
    ----------
    log_spectra_dev_threshold : float
        Maximum acceptable Log-Spectra Deviation score.  Channels above
        this are flagged as spectral outliers.
    flat_std_threshold : float
        Channels with std below this value (V) are flagged as flat or dead.
        Default 0.1 uV = 100e-9 V.
    high_amplitude_threshold : float
        Channels with peak |amplitude| above this value (V) are flagged.
        Default 150 uV = 150e-6 V.
    adc_clip_fraction_threshold : float
        Channels where more than this fraction of samples are stuck at the
        exact min or max value are flagged as ADC-saturated.
        Default 0.001 (0.1%).
    """

    log_spectra_dev_threshold: float = 2.0
    flat_std_threshold: float = 100e-9         # 0.1 uV in volts
    high_amplitude_threshold: float = 150e-6   # 150 uV in volts
    adc_clip_fraction_threshold: float = 0.001 # 0.1%

    def evaluate(
        self,
        ch_name: str,
        raw_ch: np.ndarray,
        processed_ch: np.ndarray,
        sfreq: float,
        log_spectra_dev: float = 0.0,
        adc_clip_frac: float = 0.0,
    ) -> Dict:
        """
        Run all diagnostic tests for a single channel.

        Parameters
        ----------
        ch_name          : electrode label
        raw_ch           : (n_samples,) DC-detrended raw signal
        processed_ch     : (n_samples,) fully processed signal
        sfreq            : sampling frequency (Hz)
        log_spectra_dev  : pre-computed Log-Spectra Deviation for this channel,
                           obtained by calling ``log_spectra_deviation(all_channels,
                           sfreq)`` in :func:`evaluate_all_channels`.  Requires the
                           full channel ensemble; computing on a single channel
                           always yields 0 and must not be done here.
        adc_clip_frac    : pre-computed ADC clipping fraction for this channel,
                           obtained from :func:`adc_clipping_fraction` on the raw
                           data before DC detrending.

        Returns
        -------
        dict with metric values, individual flags, and an overall ``status``.
        SNR is included as an informational metric but does not contribute to
        the channel status — near-zero SNR is expected for clean EEG where
        most signal energy already lies within the filter passband.
        """
        snr_val = float(snr_db(raw_ch[None], processed_ch[None])[0])
        mae_val = float(mae(raw_ch[None], processed_ch[None])[0])
        mse_val = float(mse(raw_ch[None], processed_ch[None])[0])
        entropy_val = float(spectral_entropy(processed_ch[None], sfreq)[0])

        activity, mobility, complexity = hjorth_parameters(processed_ch[None])
        activity_val = float(activity[0])
        mobility_val = float(mobility[0])
        complexity_val = float(complexity[0])

        std_val  = float(np.std(processed_ch))
        peak_val = float(np.max(np.abs(processed_ch)))

        flag_flat             = std_val < self.flat_std_threshold
        flag_high_amplitude   = peak_val > self.high_amplitude_threshold
        flag_spectral_outlier = log_spectra_dev > self.log_spectra_dev_threshold
        flag_adc_clipping     = adc_clip_frac > self.adc_clip_fraction_threshold

        n_flags = sum([flag_flat, flag_high_amplitude,
                       flag_spectral_outlier, flag_adc_clipping])
        if n_flags == 0:
            status = "good"
        elif n_flags == 1:
            status = "warning"
        else:
            status = "bad"

        return {
            "channel":              ch_name,
            "status":               status,
            "snr_db":               snr_val,
            "mae_V":                mae_val,
            "mse_V2":               mse_val,
            "log_spectra_dev":      log_spectra_dev,
            "spectral_entropy":     entropy_val,
            "hjorth_activity":      activity_val,
            "hjorth_mobility":      mobility_val,
            "hjorth_complexity":    complexity_val,
            "std_V":                std_val,
            "peak_V":               peak_val,
            "adc_clip_frac":        adc_clip_frac,
            "flag_flat":            flag_flat,
            "flag_high_amplitude":  flag_high_amplitude,
            "flag_spectral_outlier": flag_spectral_outlier,
            "flag_adc_clipping":    flag_adc_clipping,
        }


def detect_bad_channels(
    ch_names: List[str],
    data: np.ndarray,
    sfreq: float,
    flat_std_threshold: float = 100e-9,
    high_amplitude_threshold: float = 150e-6,
    log_spectra_dev_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect bad channels in raw (unprocessed) EEG data.

    Uses three criteria applicable to raw data (no processed reference needed):

    ==================  ===================================================
    Flag                Criterion
    ==================  ===================================================
    flat                ``std(channel) < flat_std_threshold``
    high_amplitude      ``max(|channel|) > high_amplitude_threshold``
    spectral_outlier    log-spectra deviation above threshold
    ==================  ===================================================

    Status is assigned by flag count — same rule as :class:`QualityConfig`:
    ``good`` (0 flags), ``warning`` (1 flag), ``bad`` (2+ flags).

    Parameters
    ----------
    ch_names                  : electrode labels, length n_channels
    data                      : (n_channels, n_samples) raw EEG in volts
    sfreq                     : sampling frequency (Hz)
    flat_std_threshold        : std ceiling for flat/dead channel flag (V)
    high_amplitude_threshold  : peak ceiling for high-amplitude flag (V)
    log_spectra_dev_threshold : LSD ceiling for spectral-outlier flag

    Returns
    -------
    pd.DataFrame indexed by channel name with columns:
        status, std_V, peak_V, log_spectra_dev, spectral_entropy,
        hjorth_activity, hjorth_mobility, hjorth_complexity,
        flag_flat, flag_high_amplitude, flag_spectral_outlier
    """
    lsd      = log_spectra_deviation(data, sfreq)
    se       = spectral_entropy(data, sfreq)
    act, mob, cplx = hjorth_parameters(data)

    rows = []
    for i, ch in enumerate(ch_names):
        std_val  = float(np.std(data[i]))
        peak_val = float(np.max(np.abs(data[i])))
        lsd_val  = float(lsd[i])

        flag_flat     = std_val  < flat_std_threshold
        flag_high_amp = peak_val > high_amplitude_threshold
        flag_spectral = lsd_val  > log_spectra_dev_threshold

        n_flags = sum([flag_flat, flag_high_amp, flag_spectral])
        status  = "good" if n_flags == 0 else ("warning" if n_flags == 1 else "bad")

        rows.append({
            "channel":               ch,
            "status":                status,
            "std_V":                 std_val,
            "peak_V":                peak_val,
            "log_spectra_dev":       lsd_val,
            "spectral_entropy":      float(se[i]),
            "hjorth_activity":       float(act[i]),
            "hjorth_mobility":       float(mob[i]),
            "hjorth_complexity":     float(cplx[i]),
            "flag_flat":             flag_flat,
            "flag_high_amplitude":   flag_high_amp,
            "flag_spectral_outlier": flag_spectral,
        })

    return pd.DataFrame(rows).set_index("channel")


def evaluate_all_channels(
    ch_names: List[str],
    raw_data: np.ndarray,
    processed_data: np.ndarray,
    sfreq: float,
    diagnostics: Optional[QualityConfig] = None,
) -> pd.DataFrame:
    """
    Run :class:`QualityConfig` for every channel and return a DataFrame.

    Log-Spectra Deviation is computed once over the full channel ensemble
    (as required by its definition) and then distributed to the per-channel
    ``evaluate`` calls.

    Parameters
    ----------
    ch_names       : list of electrode names, length n_channels
    raw_data       : (n_channels, n_samples) DC-detrended raw signal
    processed_data : (n_channels, n_samples) fully processed signal
    sfreq          : sampling frequency (Hz)
    diagnostics    : :class:`QualityConfig` instance; default thresholds
                     are used if ``None``.

    Returns
    -------
    pd.DataFrame indexed by channel name.
    """
    if diagnostics is None:
        diagnostics = QualityConfig()

    # Both LSD and ADC clipping require the full channel ensemble.
    lsd_scores  = log_spectra_deviation(processed_data, sfreq)
    clip_fracs  = adc_clipping_fraction(raw_data)

    rows = [
        diagnostics.evaluate(
            name, raw_data[i], processed_data[i], sfreq,
            log_spectra_dev=float(lsd_scores[i]),
            adc_clip_frac=float(clip_fracs[i]),
        )
        for i, name in enumerate(ch_names)
    ]
    return pd.DataFrame(rows).set_index("channel")
