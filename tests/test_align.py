"""Tests for eva.align_veca."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SFREQ     = 500.0
N_SECONDS = 60
N_CH      = 4
CH_NAMES  = ["Fz", "Cz", "Pz", "Oz"]

# Recording starts at this wall-clock time (matches vmrk timestamp below)
RECORDING_START = "20260626_143000.000"   # VECA format
VMRK_TS         = "20260626143000000000"  # .vmrk format (20 chars)


def _make_vmrk(path: Path, timestamp: str = VMRK_TS) -> None:
    path.write_text(
        textwrap.dedent(f"""\
            Brain Vision Data Exchange Marker File, Version 1.0
            ; Data created by BrainVision Recorder

            [Common Infos]
            Codepage=UTF-8
            DataFile=session.eeg

            [Marker Infos]
            ; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,
            ; <Size in data points>, <Channel number (0 = marker is related to all channels)>,
            ; <Date (YYYYMMDDHHmmssμμμμμμ)>

            Mk1=New Segment,,1,1,0,{timestamp}
            Mk2=Stimulus,S  1,500,1,0
        """),
        encoding="utf-8",
    )


def _make_vhdr(path: Path, eeg_file: str = "session.eeg") -> None:
    path.write_text(
        textwrap.dedent(f"""\
            Brain Vision Data Exchange Header File Version 1.0

            [Common Infos]
            DataFile={eeg_file}
            MarkerFile={path.with_suffix('.vmrk').name}
            DataFormat=BINARY
            DataOrientation=MULTIPLEXED
            NumberOfChannels={N_CH}
            SamplingInterval={int(1e6 / SFREQ)}

            [Binary Infos]
            BinaryFormat=INT_16

            [Channel Infos]
            Ch1=Fz,,0.1
            Ch2=Cz,,0.1
            Ch3=Pz,,0.1
            Ch4=Oz,,0.1
        """),
        encoding="utf-8",
    )


def _make_eeg(path: Path) -> None:
    """Write a minimal INT_16 MULTIPLEXED binary EEG file."""
    n_samples = int(SFREQ * N_SECONDS)
    data = np.zeros((N_CH, n_samples), dtype=np.int16)
    path.write_bytes(data.T.tobytes())


def _make_csv(path: Path, rows: list[dict] | None = None) -> None:
    if rows is None:
        rows = [
            {"participant_id": "ABCDEF", "trial_start": "20260626_143005.000",
             "trial_end": "20260626_143013.000", "feature": "vr_att", "value": 0.75},
            {"participant_id": "ABCDEF", "trial_start": "20260626_143020.000",
             "trial_end": "20260626_143028.000", "feature": "vr_abs", "value": 0.50},
        ]
    header = "participant_id,trial_start,trial_end,feature,value"
    lines  = [header] + [
        f"{r['participant_id']},{r['trial_start']},{r['trial_end']},{r['feature']},{r['value']}"
        for r in rows
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def session(tmp_path):
    """Return (vhdr_path, csv_path) for a valid synthetic session."""
    vhdr = tmp_path / "session.vhdr"
    vmrk = tmp_path / "session.vmrk"
    eeg  = tmp_path / "session.eeg"
    csv  = tmp_path / "VECA_ABCDEF_20260626_143000.csv"

    _make_vmrk(vmrk)
    _make_vhdr(vhdr)
    _make_eeg(eeg)
    _make_csv(csv)

    return vhdr, csv


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAlignVecaHappyPath:
    def test_returns_raw_and_dataframe(self, session):
        import mne
        import pandas as pd
        from eva import align_veca

        vhdr, csv = session
        raw, trials = align_veca(vhdr, csv)

        assert isinstance(raw, mne.io.BaseRaw)
        assert isinstance(trials, pd.DataFrame)

    def test_trials_columns(self, session):
        from eva import align_veca

        _, trials = align_veca(*session)
        assert set(trials.columns) == {"feature", "value", "onset_s", "duration_s", "onset_sample"}

    def test_onset_s_correct(self, session):
        from eva import align_veca

        _, trials = align_veca(*session)
        # vr_att starts at 143005 — recording starts at 143000 → onset = 5 s
        att = trials[trials["feature"] == "vr_att"].iloc[0]
        assert abs(att["onset_s"] - 5.0) < 1e-3

    def test_duration_s_correct(self, session):
        from eva import align_veca

        _, trials = align_veca(*session)
        att = trials[trials["feature"] == "vr_att"].iloc[0]
        assert abs(att["duration_s"] - 8.0) < 1e-3

    def test_onset_sample_correct(self, session):
        from eva import align_veca

        _, trials = align_veca(*session)
        att = trials[trials["feature"] == "vr_att"].iloc[0]
        # onset_s=5.0, sfreq=500 → sample=2500
        assert att["onset_sample"] == int(round(5.0 * SFREQ))

    def test_annotations_injected(self, session):
        from eva import align_veca

        raw, trials = align_veca(*session)
        descriptions = list(raw.annotations.description)
        assert "vr_att" in descriptions
        assert "vr_abs" in descriptions

    def test_n_annotations_matches_trials(self, session):
        from eva import align_veca

        raw, trials = align_veca(*session)
        veca_annots = [a for a in raw.annotations.description if a.startswith("vr_")]
        assert len(veca_annots) == len(trials)


# ---------------------------------------------------------------------------
# File-not-found errors
# ---------------------------------------------------------------------------

class TestAlignVecaFileMissing:
    def test_missing_vhdr_raises(self, tmp_path):
        from eva import align_veca

        csv = tmp_path / "dummy.csv"
        csv.write_text("a,b\n1,2")
        with pytest.raises(FileNotFoundError, match="session.vhdr"):
            align_veca(tmp_path / "session.vhdr", csv)

    def test_missing_csv_raises(self, tmp_path):
        from eva import align_veca

        vhdr = tmp_path / "session.vhdr"
        vmrk = tmp_path / "session.vmrk"
        eeg  = tmp_path / "session.eeg"
        _make_vmrk(vmrk)
        _make_vhdr(vhdr)
        _make_eeg(eeg)

        with pytest.raises(FileNotFoundError, match="VECA.csv"):
            align_veca(vhdr, tmp_path / "VECA.csv")

    def test_missing_vmrk_raises(self, tmp_path):
        from eva import align_veca

        vhdr = tmp_path / "session.vhdr"
        eeg  = tmp_path / "session.eeg"
        csv  = tmp_path / "VECA.csv"
        _make_vhdr(vhdr)
        _make_eeg(eeg)
        _make_csv(csv)

        with pytest.raises(FileNotFoundError, match=".vmrk"):
            align_veca(vhdr, csv)


# ---------------------------------------------------------------------------
# CSV column validation
# ---------------------------------------------------------------------------

class TestAlignVecaCsvColumns:
    def test_missing_columns_raises(self, tmp_path):
        from eva import align_veca

        vhdr = tmp_path / "session.vhdr"
        vmrk = tmp_path / "session.vmrk"
        eeg  = tmp_path / "session.eeg"
        csv  = tmp_path / "VECA.csv"

        _make_vmrk(vmrk)
        _make_vhdr(vhdr)
        _make_eeg(eeg)
        # CSV without required columns
        csv.write_text("feature,value\nvr_att,0.5", encoding="utf-8")

        with pytest.raises(ValueError, match="missing columns"):
            align_veca(vhdr, csv)

    def test_error_lists_missing_column_names(self, tmp_path):
        from eva import align_veca

        vhdr = tmp_path / "session.vhdr"
        vmrk = tmp_path / "session.vmrk"
        eeg  = tmp_path / "session.eeg"
        csv  = tmp_path / "VECA.csv"

        _make_vmrk(vmrk)
        _make_vhdr(vhdr)
        _make_eeg(eeg)
        csv.write_text("feature,value\nvr_att,0.5", encoding="utf-8")

        with pytest.raises(ValueError, match="trial_start"):
            align_veca(vhdr, csv)


# ---------------------------------------------------------------------------
# Trials outside recording window
# ---------------------------------------------------------------------------

class TestAlignVecaWindowValidation:
    def test_trial_before_recording_raises(self, tmp_path):
        from eva import align_veca

        vhdr = tmp_path / "session.vhdr"
        vmrk = tmp_path / "session.vmrk"
        eeg  = tmp_path / "session.eeg"
        csv  = tmp_path / "VECA.csv"

        _make_vmrk(vmrk)
        _make_vhdr(vhdr)
        _make_eeg(eeg)
        # Trial starts 10 s BEFORE recording
        _make_csv(csv, rows=[{
            "participant_id": "ABCDEF",
            "trial_start": "20260626_142950.000",
            "trial_end":   "20260626_142958.000",
            "feature": "vr_att",
            "value": 0.5,
        }])

        with pytest.raises(ValueError, match="outside recording window"):
            align_veca(vhdr, csv)

    def test_trial_after_recording_raises(self, tmp_path):
        from eva import align_veca

        vhdr = tmp_path / "session.vhdr"
        vmrk = tmp_path / "session.vmrk"
        eeg  = tmp_path / "session.eeg"
        csv  = tmp_path / "VECA.csv"

        _make_vmrk(vmrk)
        _make_vhdr(vhdr)
        _make_eeg(eeg)
        # Trial starts 200 s into a 60 s recording
        _make_csv(csv, rows=[{
            "participant_id": "ABCDEF",
            "trial_start": "20260626_143200.000",
            "trial_end":   "20260626_143208.000",
            "feature": "vr_att",
            "value": 0.5,
        }])

        with pytest.raises(ValueError, match="outside recording window"):
            align_veca(vhdr, csv)


# ---------------------------------------------------------------------------
# .vmrk timestamp parsing
# ---------------------------------------------------------------------------

class TestParseVmrkStart:
    def test_standard_20char_timestamp(self, tmp_path):
        from eva.align import _parse_vmrk_start

        vmrk = tmp_path / "session.vmrk"
        _make_vmrk(vmrk, timestamp="20260626143000000000")
        vhdr = tmp_path / "session.vhdr"
        vhdr.write_text("")

        result = _parse_vmrk_start(vhdr)
        assert result == datetime(2026, 6, 26, 14, 30, 0)

    def test_14char_timestamp_padded(self, tmp_path):
        from eva.align import _parse_vmrk_start

        vmrk = tmp_path / "session.vmrk"
        _make_vmrk(vmrk, timestamp="20260626143000")
        vhdr = tmp_path / "session.vhdr"
        vhdr.write_text("")

        result = _parse_vmrk_start(vhdr)
        assert result == datetime(2026, 6, 26, 14, 30, 0)

    def test_no_segment_marker_returns_none(self, tmp_path):
        from eva.align import _parse_vmrk_start

        vmrk = tmp_path / "session.vmrk"
        vmrk.write_text("Brain Vision Data Exchange Marker File\n\nMk1=Stimulus,S1,500,1,0\n")
        vhdr = tmp_path / "session.vhdr"
        vhdr.write_text("")

        assert _parse_vmrk_start(vhdr) is None

    def test_missing_vmrk_raises(self, tmp_path):
        from eva.align import _parse_vmrk_start

        vhdr = tmp_path / "session.vhdr"
        vhdr.write_text("")

        with pytest.raises(FileNotFoundError, match=".vmrk"):
            _parse_vmrk_start(vhdr)
