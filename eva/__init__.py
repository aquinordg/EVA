"""
EVA — EEG data Validation and preprocessing Assistant.

Quick start
-----------
>>> from eva import convert, preprocess, sync

# Step 1 — normalise format (BrainVision, EDF, BDF, EEGLAB -> .fif)
>>> convert("subject01.vhdr")

# Step 2 — preprocess (filter -> reference -> clip -> epochs -> .h5 + report)
>>> preprocess("subject01.fif", report=True)

# Step 3 — synchronise behavioral and/or physiological data
>>> sync("subject01.h5",
...      behavioral={"rt": rt_array, "accuracy": acc_array},
...      physio={"ecg": ecg_array},
...      physio_sfreq=1000.0)

# Steps 1–2 can be skipped: pass any supported file or mne.Raw to preprocess()
>>> preprocess("subject01.vhdr", optimize=True, report=True)

# VECA-EEG integration: align trial CSV with BrainVision recording
>>> from eva import align_veca
>>> raw, trials = align_veca("session.vhdr", "VECA_ABCDEF_ts.csv")
>>> preprocess(raw, epoch_tmin=0.0, epoch_tmax=8.0)
"""

from .convert import convert
from .preprocess import preprocess
from .sync import sync
from .metrics import QualityConfig
from .align import align_veca

__all__ = ["convert", "preprocess", "sync", "QualityConfig", "align_veca"]
__version__ = "1.1.1"
