"""Align a VECA-EEG trial CSV with a BrainVision Recorder recording.

Usage
-----
>>> from eva import align_veca, preprocess, sync
>>> raw, trials = align_veca("session.vhdr", "VECA_ABCDEF_20260626_143000.csv")
>>> h5 = preprocess(raw, epoch_tmin=0.0, epoch_tmax=8.0)
>>> sync(h5, behavioral={"score": trials["value"].values})
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import mne
import numpy as np

logger = logging.getLogger(__name__)

_VECA_TS_FORMAT = "%Y%m%d_%H%M%S.%f"
_VMRK_TS_FORMAT = "%Y%m%d%H%M%S%f"

# Matches: Mk<n>=New Segment,<pos>,<size>,<channel>,<date20chars>
_VMRK_SEGMENT_RE = re.compile(r"^Mk\d+=New Segment,.*?(\d{14,20})\s*$")


def align_veca(
    vhdr_path: Union[str, Path],
    csv_path: Union[str, Path],
) -> Tuple[mne.io.BaseRaw, "pd.DataFrame"]:
    """
    Load a BrainVision recording and align VECA-EEG trial timestamps.

    Injects trial annotations into the returned ``mne.Raw`` object so that
    :func:`preprocess` can epoch around each cognitive task.  Both files must
    come from the same session: they share the Windows system clock as their
    common time base.

    Parameters
    ----------
    vhdr_path
        Path to the BrainVision header file (``.vhdr``).  The companion
        ``.vmrk`` marker file must be in the same directory.
    csv_path
        Path to the VECA-EEG results CSV (``VECA_<ID>_<timestamp>.csv``).
        Required columns: ``trial_start``, ``trial_end``, ``feature``,
        ``value``.  Timestamp format: ``YYYYMMDD_HHMMSS.fff``.

    Returns
    -------
    raw : mne.io.BaseRaw
        Preloaded BrainVision recording with VECA trial annotations injected.
    trials : pandas.DataFrame
        One row per trial with columns:
        ``feature``, ``value``, ``onset_s``, ``duration_s``, ``onset_sample``.

    Raises
    ------
    FileNotFoundError
        If ``vhdr_path``, its companion ``.vmrk``, or ``csv_path`` is missing.
    ValueError
        If the ``.vmrk`` contains no parseable ``New Segment`` timestamp, if
        required CSV columns are absent, or if any trial falls outside the
        recording window.

    Examples
    --------
    >>> raw, trials = align_veca("VECA_EEG.vhdr", "VECA_ABCDEF_ts.csv")
    >>> h5 = preprocess(raw, epoch_tmin=0.0, epoch_tmax=8.0)
    >>> sync(h5, behavioral={"score": trials["value"].values})
    """
    import pandas as pd

    vhdr_path = Path(vhdr_path).resolve()
    csv_path  = Path(csv_path).resolve()

    for p in (vhdr_path, csv_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    raw   = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    recording_duration = raw.n_times / sfreq

    recording_start = _parse_vmrk_start(vhdr_path)
    if recording_start is None:
        raise ValueError(
            f"Could not find 'New Segment' timestamp in "
            f"'{vhdr_path.with_suffix('.vmrk')}'. "
            "Ensure the file was recorded with BrainVision Recorder."
        )

    trials = pd.read_csv(csv_path)
    _validate_csv_columns(trials, csv_path)

    trials["_start_dt"] = pd.to_datetime(trials["trial_start"], format=_VECA_TS_FORMAT)
    trials["_end_dt"]   = pd.to_datetime(trials["trial_end"],   format=_VECA_TS_FORMAT)

    rec_ts = pd.Timestamp(recording_start)
    trials["onset_s"]      = (trials["_start_dt"] - rec_ts).dt.total_seconds()
    trials["duration_s"]   = (trials["_end_dt"] - trials["_start_dt"]).dt.total_seconds()
    trials["onset_sample"] = (trials["onset_s"] * sfreq).round().astype(int)

    out_of_window = trials[
        (trials["onset_s"] < 0) | (trials["onset_s"] > recording_duration)
    ]
    if len(out_of_window):
        raise ValueError(
            f"Trials outside recording window ({recording_duration:.1f} s): "
            f"{out_of_window['feature'].tolist()}. "
            "Check that the CSV and .vhdr belong to the same session."
        )

    annotations = mne.Annotations(
        onset=trials["onset_s"].values,
        duration=trials["duration_s"].values,
        description=trials["feature"].values,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + annotations)
    logger.info("Injected %d VECA annotations into raw (sfreq=%.1f Hz).", len(trials), sfreq)

    out = trials[["feature", "value", "onset_s", "duration_s", "onset_sample"]].copy()
    return raw, out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_vmrk_start(vhdr_path: Path) -> datetime | None:
    """Return the ``New Segment`` datetime from the companion .vmrk file."""
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    if not vmrk_path.exists():
        raise FileNotFoundError(f".vmrk file not found: {vmrk_path}")

    with open(vmrk_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _VMRK_SEGMENT_RE.match(line.strip())
            if m:
                raw_ts = m.group(1).ljust(20, "0")[:20]
                try:
                    return datetime.strptime(raw_ts, _VMRK_TS_FORMAT)
                except ValueError:
                    continue
    return None


def _validate_csv_columns(df: "pd.DataFrame", path: Path) -> None:
    required = {"trial_start", "trial_end", "feature", "value"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"VECA CSV '{path.name}' is missing columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}."
        )
