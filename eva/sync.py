"""Synchronise behavioral and physiological data with preprocessed EEG epochs.

Adds data to an existing .h5 file produced by preprocess(), keeping everything
for one subject in a single file ready for downstream ML pipelines.

HDF5 structure after sync()
----------------------------
subject01.h5
  /eeg/
    data          (n_epochs, n_channels, n_times)  float32   — written by preprocess()
    labels        (n_epochs,)                       int32
    ch_names      (n_channels,)                     str
    label_names   (n_classes,)                      str
    label_codes   (n_classes,)                      int32
  /behavioral/
    <key>         (n_epochs,) or (n_epochs, n_features)       — written by sync()
  /physio/
    <key>         (n_epochs, n_times) or (n_epochs, n_ch, n_times)
                  attribute: sfreq (Hz) — if physio_sfreq provided
  /metadata/
    sfreq, tmin, tmax   — written by preprocess()

Usage
-----
>>> from eva import sync
>>> import numpy as np

>>> # Behavioral only
>>> sync("subject01.h5", behavioral={"rt": rt_array, "accuracy": acc_array})

>>> # Physiological only
>>> sync("subject01.h5",
...      physio={"ecg": ecg_array},
...      physio_sfreq=1000.0)

>>> # Both at once
>>> sync("subject01.h5",
...      behavioral={"rt": rt_array, "accuracy": acc_array},
...      physio={"ecg": ecg_array, "emg": emg_array},
...      physio_sfreq={"ecg": 1000.0, "emg": 2000.0})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def sync(
    path: Union[str, Path],
    *,
    behavioral: Optional[Dict[str, np.ndarray]] = None,
    physio: Optional[Dict[str, np.ndarray]] = None,
    physio_sfreq: Optional[Union[float, Dict[str, float]]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Add behavioral and/or physiological data to a preprocessed .h5 file.

    At least one of *behavioral* or *physio* must be provided.
    All arrays must share the same first dimension (n_epochs) as the EEG data
    already stored in the file.

    Parameters
    ----------
    path
        Path to the .h5 file produced by ``preprocess()``.
    behavioral
        Dict mapping signal names to arrays.  Each array must have shape
        ``(n_epochs,)`` or ``(n_epochs, n_features)``.
        Examples: ``{"rt": array, "accuracy": array}``.
    physio
        Dict mapping signal names to arrays.  Each array must have shape
        ``(n_epochs, n_times)`` or ``(n_epochs, n_channels, n_times)``.
        Examples: ``{"ecg": array, "emg": array}``.
    physio_sfreq
        Sampling rate(s) for physiological signals, stored as an attribute
        on each dataset.  Pass a single float (same rate for all signals)
        or a dict ``{signal_name: sfreq}`` for per-signal rates.
    overwrite
        When ``False`` (default), raises an error if a signal already exists
        in the file.  When ``True``, replaces existing signals silently.

    Returns
    -------
    Path
        Path to the updated .h5 file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file was not created by ``preprocess()``, if no data is
        provided, or if array shapes are incompatible with the stored EEG.
    """
    import h5py

    if behavioral is None and physio is None:
        raise ValueError("Provide at least one of 'behavioral' or 'physio'.")

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sfreq_map = _build_sfreq_map(physio or {}, physio_sfreq)

    with h5py.File(path, "a") as f:
        if "eeg" not in f:
            raise ValueError(
                f"'{path.name}' has no '/eeg/' group. "
                "It must be created by preprocess() before calling sync()."
            )
        n_epochs = f["eeg/labels"].shape[0]

        if behavioral is not None:
            _write_signals(f, "behavioral", behavioral, n_epochs, overwrite, sfreq_map={})

        if physio is not None:
            _write_signals(f, "physio", physio, n_epochs, overwrite, sfreq_map=sfreq_map)

    logger.info(
        "Synced '%s'  behavioral=%s  physio=%s",
        path.name,
        list(behavioral or {}),
        list(physio or {}),
    )
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sfreq_map(
    physio: Dict[str, np.ndarray],
    physio_sfreq: Optional[Union[float, Dict[str, float]]],
) -> Dict[str, float]:
    if physio_sfreq is None:
        return {}
    if isinstance(physio_sfreq, (int, float)):
        return {k: float(physio_sfreq) for k in physio}
    return {k: float(v) for k, v in physio_sfreq.items()}


def _write_signals(
    f,
    group_name: str,
    signals: Dict[str, np.ndarray],
    n_epochs: int,
    overwrite: bool,
    sfreq_map: Dict[str, float],
) -> None:
    grp = f.require_group(group_name)

    for key, arr in signals.items():
        arr = np.asarray(arr)
        if arr.shape[0] != n_epochs:
            raise ValueError(
                f"'{group_name}/{key}' has {arr.shape[0]} rows but "
                f"EEG data has {n_epochs} epochs. Arrays must align on axis 0."
            )

        if key in grp:
            if overwrite:
                del grp[key]
            else:
                raise ValueError(
                    f"'{group_name}/{key}' already exists in '{f.filename}'. "
                    "Use overwrite=True to replace it."
                )

        arr = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr
        dset = grp.create_dataset(key, data=arr)

        if key in sfreq_map:
            dset.attrs["sfreq"] = sfreq_map[key]
