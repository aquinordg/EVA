"""Unit tests for eva.preprocess helpers, eva.sync, eva.convert, and eva.optimizer."""

from __future__ import annotations

import numpy as np
import pytest

SFREQ = 500.0
N_CH = 4
N_SAMP = 2000
N_EPOCHS = 10


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_CH, N_SAMP)) * 20e-6


@pytest.fixture
def minimal_h5(tmp_path):
    """Create a minimal .h5 file with /eeg/ group matching preprocess() output."""
    import h5py

    path = tmp_path / "subject.h5"
    with h5py.File(path, "w") as f:
        eeg = f.create_group("eeg")
        eeg.create_dataset("data",   data=np.zeros((N_EPOCHS, N_CH, 250), dtype=np.float32))
        eeg.create_dataset("labels", data=np.ones(N_EPOCHS, dtype=np.int32))
        eeg.create_dataset("ch_names",    data=np.array([f"CH{i}" for i in range(N_CH)],
                                                         dtype=object))
        eeg.create_dataset("label_names", data=np.array(["stim"], dtype=object))
        eeg.create_dataset("label_codes", data=np.array([1], dtype=np.int32))
        meta = f.create_group("metadata")
        meta.attrs["sfreq"] = np.float32(SFREQ)
        meta.attrs["tmin"]  = np.float32(0.0)
        meta.attrs["tmax"]  = np.float32(0.5)
    return path


# ---------------------------------------------------------------------------
# convert()
# ---------------------------------------------------------------------------

class TestConvert:
    def test_missing_file_raises(self, tmp_path):
        from eva import convert
        with pytest.raises(FileNotFoundError):
            convert(tmp_path / "nonexistent.vhdr")

    def test_unsupported_extension_raises(self, tmp_path):
        from eva import convert
        fake = tmp_path / "recording.txt"
        fake.write_text("not an eeg file")
        with pytest.raises(ValueError, match="Cannot detect format"):
            convert(fake)

    def test_unknown_input_type_raises(self, tmp_path):
        from eva import convert
        fake = tmp_path / "recording.vhdr"
        fake.write_text("")
        with pytest.raises(ValueError, match="Unknown input_type"):
            convert(fake, input_type="unsupported_format")


# ---------------------------------------------------------------------------
# preprocess() — internal helpers (no real EEG file required)
# ---------------------------------------------------------------------------

class TestBuildChain:
    def test_default_chain_length(self):
        from eva.preprocess import _build_chain
        # DCDetrend + Butterworth + Notch + AverageRef + SoftClipper = 5
        chain = _build_chain(1.0, 40.0, 4, 60.0, 100e-6, True, True)
        assert len(chain) == 5

    def test_no_notch_shortens_chain(self):
        from eva.preprocess import _build_chain
        chain = _build_chain(1.0, 40.0, 4, None, 100e-6, True, True)
        assert len(chain) == 4

    def test_no_avg_ref_shortens_chain(self):
        from eva.preprocess import _build_chain
        chain = _build_chain(1.0, 40.0, 4, 60.0, 100e-6, False, True)
        assert len(chain) == 4

    def test_no_soft_clip_shortens_chain(self):
        from eva.preprocess import _build_chain
        chain = _build_chain(1.0, 40.0, 4, 60.0, 100e-6, True, False)
        assert len(chain) == 4

    def test_minimal_chain_length(self):
        from eva.preprocess import _build_chain
        # DCDetrend + Butterworth only
        chain = _build_chain(1.0, 40.0, 4, None, 100e-6, False, False)
        assert len(chain) == 2


class TestApplyChain:
    def test_output_shapes(self, synthetic_data):
        from eva.preprocess import _build_chain, _apply_chain
        chain = _build_chain(1.0, 40.0, 4, None, 100e-6, True, True)
        raw_det, processed = _apply_chain(chain, synthetic_data, SFREQ)
        assert raw_det.shape == synthetic_data.shape
        assert processed.shape == synthetic_data.shape

    def test_removes_dc(self, synthetic_data):
        from eva.preprocess import _build_chain, _apply_chain
        data_with_dc = synthetic_data + 50e-6
        chain = _build_chain(1.0, 40.0, 4, None, 100e-6, True, True)
        _, processed = _apply_chain(chain, data_with_dc, SFREQ)
        assert np.abs(processed.mean()) < np.abs(data_with_dc.mean()) * 0.01

    def test_raw_detrend_independent_of_full_chain(self, synthetic_data):
        from eva.preprocess import _build_chain, _apply_chain
        chain = _build_chain(1.0, 40.0, 4, 60.0, 100e-6, True, True)
        raw_det, processed = _apply_chain(chain, synthetic_data, SFREQ)
        assert not np.allclose(raw_det, processed)


class TestPreprocessErrors:
    def test_missing_file_raises(self, tmp_path):
        from eva import preprocess
        with pytest.raises(FileNotFoundError):
            preprocess(tmp_path / "ghost.fif")

    def test_unsupported_extension_raises(self, tmp_path):
        from eva import preprocess
        fake = tmp_path / "data.wav"
        fake.write_bytes(b"")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            preprocess(fake)


# ---------------------------------------------------------------------------
# sync()
# ---------------------------------------------------------------------------

class TestSync:
    def test_no_data_raises(self, minimal_h5):
        from eva import sync
        with pytest.raises(ValueError, match="at least one"):
            sync(minimal_h5)

    def test_missing_file_raises(self, tmp_path):
        from eva import sync
        with pytest.raises(FileNotFoundError):
            sync(tmp_path / "ghost.h5", behavioral={"rt": np.ones(N_EPOCHS)})

    def test_no_eeg_group_raises(self, tmp_path):
        import h5py
        from eva import sync
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_group("other")
        with pytest.raises(ValueError, match="no '/eeg/'"):
            sync(path, behavioral={"rt": np.ones(5)})

    def test_write_behavioral(self, minimal_h5):
        import h5py
        from eva import sync
        rt = np.linspace(0.3, 0.9, N_EPOCHS)
        result = sync(minimal_h5, behavioral={"rt": rt})
        assert result == minimal_h5
        with h5py.File(minimal_h5, "r") as f:
            np.testing.assert_allclose(f["behavioral/rt"][:], rt.astype(np.float32), atol=1e-6)

    def test_write_physio_with_sfreq(self, minimal_h5):
        import h5py
        from eva import sync
        ecg = np.zeros((N_EPOCHS, 500))
        sync(minimal_h5, physio={"ecg": ecg}, physio_sfreq=1000.0)
        with h5py.File(minimal_h5, "r") as f:
            assert "physio/ecg" in f
            assert f["physio/ecg"].attrs["sfreq"] == pytest.approx(1000.0)

    def test_overwrite_false_raises(self, minimal_h5):
        import h5py
        from eva import sync
        rt = np.ones(N_EPOCHS)
        sync(minimal_h5, behavioral={"rt": rt})
        with pytest.raises(ValueError, match="already exists"):
            sync(minimal_h5, behavioral={"rt": rt}, overwrite=False)

    def test_overwrite_true_replaces(self, minimal_h5):
        import h5py
        from eva import sync
        sync(minimal_h5, behavioral={"rt": np.ones(N_EPOCHS)})
        new_rt = np.full(N_EPOCHS, 0.5, dtype=np.float32)
        sync(minimal_h5, behavioral={"rt": new_rt}, overwrite=True)
        with h5py.File(minimal_h5, "r") as f:
            np.testing.assert_allclose(f["behavioral/rt"][:], new_rt, atol=1e-6)

    def test_shape_mismatch_raises(self, minimal_h5):
        from eva import sync
        wrong = np.ones(N_EPOCHS + 5)
        with pytest.raises(ValueError, match="rows but"):
            sync(minimal_h5, behavioral={"rt": wrong})

    def test_returns_path(self, minimal_h5):
        from eva import sync
        from pathlib import Path
        result = sync(minimal_h5, behavioral={"acc": np.ones(N_EPOCHS)})
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# find_best_params()
# ---------------------------------------------------------------------------

class TestFindBestParams:
    @pytest.fixture
    def minimal_grid(self):
        return {
            "l_freq":             [1.0],
            "h_freq":             [40.0],
            "filter_order":       [4],
            "notch_freq":         [None],
            "artifact_threshold": [100e-6],
            "use_avg_ref":        [True],
            "use_soft_clip":      [True],
        }

    def test_returns_dict(self, synthetic_data, minimal_grid):
        from eva.optimizer import find_best_params
        result = find_best_params(synthetic_data, SFREQ, grid=minimal_grid)
        assert isinstance(result, dict)

    def test_expected_keys(self, synthetic_data, minimal_grid):
        from eva.optimizer import find_best_params
        result = find_best_params(synthetic_data, SFREQ, grid=minimal_grid)
        for key in ("l_freq", "h_freq", "filter_order", "notch_freq",
                    "artifact_threshold", "use_avg_ref", "use_soft_clip"):
            assert key in result

    def test_single_config_selected(self, synthetic_data, minimal_grid):
        from eva.optimizer import find_best_params
        result = find_best_params(synthetic_data, SFREQ, grid=minimal_grid)
        assert result["l_freq"] == 1.0
        assert result["h_freq"] == 40.0
        assert result["filter_order"] == 4
