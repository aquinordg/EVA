"""Unit tests for eva.filters."""

import numpy as np
import pytest

from eva.filters import (
    AverageReference,
    ButterworthFilter,
    DCDetrend,
    NotchFilter,
    SoftClipper,
)

SFREQ = 500.0
N_CH = 4
N_SAMP = 2000


@pytest.fixture
def white_noise():
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_CH, N_SAMP)) * 20e-6


@pytest.fixture
def dc_offset_signal(white_noise):
    offsets = np.array([10e-6, -5e-6, 20e-6, -15e-6])[:, None]
    return white_noise + offsets


class TestDCDetrend:
    def test_removes_mean(self, dc_offset_signal):
        out = DCDetrend().apply(dc_offset_signal, SFREQ)
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-15)

    def test_output_shape(self, white_noise):
        out = DCDetrend().apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape

    def test_zero_mean_signal_unchanged(self, white_noise):
        data = white_noise - white_noise.mean(axis=1, keepdims=True)
        out = DCDetrend().apply(data, SFREQ)
        np.testing.assert_allclose(out, data, atol=1e-15)


class TestButterworthFilter:
    def test_bandpass_removes_dc(self, white_noise):
        data = white_noise + 50e-6
        out = ButterworthFilter(l_freq=1.0, h_freq=40.0).apply(data, SFREQ)
        assert np.abs(out.mean()) < np.abs(data.mean()) * 0.01

    def test_output_shape(self, white_noise):
        out = ButterworthFilter(l_freq=1.0, h_freq=40.0).apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape

    def test_noop_when_both_none(self, white_noise):
        out = ButterworthFilter(l_freq=None, h_freq=None).apply(white_noise, SFREQ)
        np.testing.assert_array_equal(out, white_noise)

    def test_highpass_only(self, white_noise):
        out = ButterworthFilter(l_freq=1.0, h_freq=None).apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape

    def test_lowpass_only(self, white_noise):
        out = ButterworthFilter(l_freq=None, h_freq=40.0).apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape


class TestNotchFilter:
    def test_attenuates_at_notch_frequency(self):
        t = np.arange(int(SFREQ * 5)) / SFREQ
        sine_60 = np.sin(2 * np.pi * 60 * t)
        data = np.stack([sine_60, sine_60])
        out = NotchFilter(freq=60.0).apply(data, SFREQ)
        rms_before = np.sqrt(np.mean(data ** 2))
        rms_after = np.sqrt(np.mean(out ** 2))
        assert rms_after < rms_before * 0.1

    def test_output_shape(self, white_noise):
        out = NotchFilter().apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape


class TestAverageReference:
    def test_zero_cross_channel_mean(self, white_noise):
        out = AverageReference().apply(white_noise, SFREQ)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-15)

    def test_output_shape(self, white_noise):
        out = AverageReference().apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape


class TestSoftClipper:
    def test_passthrough_below_threshold(self):
        threshold = 100e-6
        data = np.array([[50e-6, -50e-6, 0.0, 30e-6]])
        out = SoftClipper(threshold=threshold).apply(data, SFREQ)
        np.testing.assert_allclose(out, data, atol=1e-15)

    def test_compresses_above_threshold(self):
        threshold = 100e-6
        data = np.array([[200e-6, -300e-6, 500e-6]])
        out = SoftClipper(threshold=threshold).apply(data, SFREQ)
        assert np.all(np.abs(out) < np.abs(data))

    def test_soft_asymptote(self):
        threshold = 100e-6
        data = np.array([[1e-3, -1e-3]])  # 10× threshold
        out = SoftClipper(threshold=threshold).apply(data, SFREQ)
        assert np.all(np.abs(out) < 2.1 * threshold)

    def test_sign_preserved(self):
        threshold = 100e-6
        data = np.array([[200e-6, -200e-6]])
        out = SoftClipper(threshold=threshold).apply(data, SFREQ)
        assert out[0, 0] > 0
        assert out[0, 1] < 0

    def test_output_shape(self, white_noise):
        out = SoftClipper().apply(white_noise, SFREQ)
        assert out.shape == white_noise.shape


