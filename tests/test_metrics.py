"""Unit tests for eva.metrics."""

import numpy as np
import pandas as pd
import pytest

from eva.metrics import (
    QualityConfig,
    detect_bad_channels,
    evaluate_all_channels,
    hjorth_parameters,
    log_spectra_deviation,
    mae,
    mse,
    palosi,
    snr_db,
    spectral_entropy,
)

SFREQ = 500.0
N_CH = 4
N_SAMP = 2000


@pytest.fixture
def signal():
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_CH, N_SAMP)) * 20e-6


@pytest.fixture
def signal_and_processed(signal):
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((N_CH, N_SAMP)) * 2e-6
    return signal, signal + noise


class TestMAE:
    def test_identical_signals_zero(self, signal):
        np.testing.assert_allclose(mae(signal, signal), 0.0, atol=1e-20)

    def test_shape(self, signal_and_processed):
        raw, proc = signal_and_processed
        assert mae(raw, proc).shape == (N_CH,)

    def test_non_negative(self, signal_and_processed):
        raw, proc = signal_and_processed
        assert np.all(mae(raw, proc) >= 0)


class TestMSE:
    def test_identical_signals_zero(self, signal):
        np.testing.assert_allclose(mse(signal, signal), 0.0, atol=1e-40)

    def test_shape(self, signal_and_processed):
        raw, proc = signal_and_processed
        assert mse(raw, proc).shape == (N_CH,)

    def test_non_negative(self, signal_and_processed):
        raw, proc = signal_and_processed
        assert np.all(mse(raw, proc) >= 0)


class TestSNRdB:
    def test_identical_signals_high_snr(self, signal):
        assert np.all(snr_db(signal, signal) > 60)

    def test_shape(self, signal_and_processed):
        raw, proc = signal_and_processed
        assert snr_db(raw, proc).shape == (N_CH,)

    def test_known_ratio(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((1, 10_000)) * 10.0
        noise = rng.standard_normal((1, 10_000)) * 1.0
        result = snr_db(sig, sig - noise)
        assert result[0] > 15


class TestHjorthParameters:
    def test_output_shapes(self, signal):
        act, mob, cplx = hjorth_parameters(signal)
        assert act.shape == (N_CH,)
        assert mob.shape == (N_CH,)
        assert cplx.shape == (N_CH,)

    def test_activity_non_negative(self, signal):
        act, _, _ = hjorth_parameters(signal)
        assert np.all(act >= 0)

    def test_mobility_non_negative(self, signal):
        _, mob, _ = hjorth_parameters(signal)
        assert np.all(mob >= 0)

    def test_constant_signal_zero_activity(self):
        data = np.ones((2, 1000)) * 5e-6
        act, _, _ = hjorth_parameters(data)
        np.testing.assert_allclose(act, 0.0, atol=1e-20)


class TestSpectralEntropy:
    def test_output_shape(self, signal):
        assert spectral_entropy(signal, SFREQ).shape == (N_CH,)

    def test_range(self, signal):
        result = spectral_entropy(signal, SFREQ)
        assert np.all(result >= 0) and np.all(result <= 1.0)

    def test_white_noise_high_entropy(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((2, 5000)) * 20e-6
        assert np.all(spectral_entropy(noise, SFREQ) > 0.8)


class TestPaLOSi:
    def test_range(self, signal):
        assert 0.0 <= palosi(signal, SFREQ) <= 1.0

    def test_single_channel_returns_zero(self, signal):
        assert palosi(signal[:1], SFREQ) == 0.0


class TestLogSpectraDeviation:
    def test_output_shape(self, signal):
        assert log_spectra_deviation(signal, SFREQ).shape == (N_CH,)

    def test_non_negative(self, signal):
        assert np.all(log_spectra_deviation(signal, SFREQ) >= 0)


class TestQualityConfig:
    def test_status_values(self):
        rng = np.random.default_rng(0)
        ch = rng.standard_normal(N_SAMP) * 20e-6
        result = QualityConfig().evaluate("Cz", ch, ch * 0.99, SFREQ)
        assert result["status"] in {"good", "warning", "bad"}
        assert result["channel"] == "Cz"
        assert "snr_db" in result

    def test_flat_channel_flagged(self):
        flat = np.ones(N_SAMP) * 1e-9
        result = QualityConfig().evaluate("Fz", flat, flat * 0.99, SFREQ)
        assert result["flag_flat"] is True

    def test_high_amplitude_flagged(self):
        rng = np.random.default_rng(1)
        ch = rng.standard_normal(N_SAMP) * 500e-6
        result = QualityConfig().evaluate("C3", ch, ch * 0.99, SFREQ)
        assert result["flag_high_amplitude"] is True


class TestDetectBadChannels:
    def test_returns_dataframe(self, signal):
        ch_names = [f"CH{i}" for i in range(N_CH)]
        df = detect_bad_channels(ch_names, signal, SFREQ)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ch_names

    def test_required_columns(self, signal):
        ch_names = [f"CH{i}" for i in range(N_CH)]
        df = detect_bad_channels(ch_names, signal, SFREQ)
        for col in ("status", "std_V", "peak_V", "flag_flat",
                    "flag_high_amplitude", "flag_spectral_outlier"):
            assert col in df.columns

    def test_flat_channel_detected(self):
        data = np.zeros((2, N_SAMP))
        data[0] = 1e-9  # nearly flat
        df = detect_bad_channels(["flat", "normal"], data, SFREQ)
        assert df.loc["flat", "flag_flat"] == True

    def test_high_amplitude_detected(self):
        rng = np.random.default_rng(2)
        data = rng.standard_normal((2, N_SAMP)) * 20e-6
        data[0] = rng.standard_normal(N_SAMP) * 500e-6  # high amplitude
        df = detect_bad_channels(["high", "normal"], data, SFREQ)
        assert df.loc["high", "flag_high_amplitude"] == True


class TestEvaluateAllChannels:
    def test_returns_dataframe_with_correct_index(self, signal):
        ch_names = [f"CH{i}" for i in range(N_CH)]
        df = evaluate_all_channels(ch_names, signal, signal * 0.99, SFREQ)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ch_names
        assert "status" in df.columns
