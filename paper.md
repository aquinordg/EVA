---
title: 'EVA: A Python Library for EEG Preprocessing, Quality Assessment, and Multimodal Synchronisation'
tags:
  - Python
  - EEG
  - neuroscience
  - signal processing
  - BrainVision
  - multimodal
authors:
  - name: Roberto Douglas Guimarães de Aquino
    orcid: 0000-0002-8486-8354
    affiliation: 1
affiliations:
  - name: Universidade de São Paulo, São Paulo, Brazil
    index: 1
date: 15 June 2026
bibliography: paper.bib
---

# Summary

EVA (EEG data Validation and preprocessing Assistant) is an open-source
Python library that converts raw EEG recordings into preprocessed,
quality-assessed, and labelled epoch arrays ready for downstream
statistical analysis or machine learning. The public API exposes three
functions that map one-to-one onto the stages of a typical EEG
workflow: `convert()` normalises any supported input format (BrainVision,
EDF, BDF, EEGLAB) to MNE `.fif`; `preprocess()` applies a fixed,
scientifically motivated filter chain to the continuous signal, segments
it into epochs aligned to event annotations, and saves the result as a
compressed HDF5 file; and `sync()` appends per-epoch behavioural and
physiological co-variables to the same file, producing a single
self-contained artefact per participant. An optional grid-search
optimiser (`find_best_params()`) identifies the filter configuration that
maximises a composite signal quality score before full processing.

# Statement of need

Preprocessing raw EEG data into analysis-ready epochs involves a repeated
sequence of decisions — frequency cutoffs, reference scheme, artefact
thresholds — that are commonly implemented ad hoc in laboratory-specific
scripts. This fragmentation hinders reproducibility: two researchers
applying nominally identical steps to the same recording may obtain
different results due to implementation details that are rarely reported
in full [@Luck2014].

Established toolboxes such as MNE-Python [@Gramfort2013] and EEGLAB
[@Delorme2004] offer comprehensive environments for EEG analysis. However,
their generality comes at a cost: assembling a reproducible, batch-ready
preprocessing pipeline requires substantial configuration and domain
expertise. EVA addresses this gap by providing a minimal, opinionated
API surface — three functions covering conversion, preprocessing, and
multimodal data fusion — built on top of MNE's I/O and epoch extraction.

The primary target audience is researchers in cognitive neuroscience and
brain-computer interface development who need a reproducible preprocessing
step that integrates naturally with multimodal acquisition systems. EVA
is particularly suited to paradigms where EEG is acquired alongside eye
tracking or physiological signals and event markers are broadcast via Lab
Streaming Layer (LSL) and recorded into `.vmrk` annotation files. EVA
was developed to support VECA-EEG [@VECAEEG], a Unity 6 virtual reality
platform for cognitive assessment in which LSL markers emitted by the
stimulus system serve directly as EVA's epoch class labels — with no
remapping required.

# Installation

EVA requires Python 3.10 or later. It can be installed directly from the
repository:

```bash
pip install eva-eeg
```

Core dependencies are MNE-Python [@Gramfort2013], SciPy [@Virtanen2020],
NumPy [@Harris2020], pandas, h5py, matplotlib, and tqdm.

# Usage

A complete preprocessing workflow for one participant requires three
function calls:

```python
from eva import convert, preprocess, sync
import numpy as np

# Step 1 — convert to .fif and inspect channel quality
convert("sub-01.vhdr", report=True)

# Step 2 — filter, epoch, and save to HDF5
preprocess("sub-01.fif", l_freq=1.0, h_freq=40.0,
           notch_freq=60.0, report=True)

# Step 3 — attach per-epoch behavioural co-variables
rt  = np.load("sub-01_rt.npy")   # shape (n_epochs,)
acc = np.load("sub-01_acc.npy")
sync("sub-01.h5", behavioral={"rt": rt, "accuracy": acc},
     physio={"ecg": ecg_array}, physio_sfreq=1000.0)
```

Steps 1 and 2 can be collapsed: `preprocess()` accepts any supported
file format directly, converting internally without writing an
intermediate `.fif`. The `optimize=True` flag triggers the grid-search
optimiser before filtering.

# Features

## Filter chain

`preprocess()` applies five sequential steps to the continuous signal:

1. **DC detrend** — subtracts the temporal mean per channel, stabilising
   IIR filter initialisation.
2. **Zero-phase Butterworth bandpass** — default 1–40 Hz, order 4;
   implemented via `scipy.signal.sosfiltfilt` for numerical stability
   at any cutoff-to-sampling-rate ratio [@Virtanen2020].
3. **IIR notch** — default 60 Hz (configurable for 50 Hz mains), Q = 30.
4. **Common Average Reference (CAR)** — subtracts the instantaneous
   spatial mean, attenuating artefacts shared across all electrodes.
5. **Soft clipper** — hyperbolic-tangent limiter; amplitudes within the
   threshold (default 100 µV) pass through unchanged while excess
   amplitude is continuously compressed, avoiding the spectral
   discontinuities introduced by hard zeroing.

All parameters are keyword arguments to `preprocess()` and can be
overridden without modifying source code.

## Quality assessment

EVA provides two levels of quality assessment. Before any filtering,
`convert(report=True)` runs `detect_bad_channels()` on the raw signal
and generates an HTML report. Each channel is evaluated against three
criteria: flatness (std below 100 nV), high amplitude (peak above
150 µV), and spectral outlier score (Log-Spectra Deviation above 2.0).
Channels meeting two or more criteria are flagged as *bad* and can be
excluded via `channel_picks` in the subsequent `preprocess()` call.

After filtering, `preprocess()` computes per-channel diagnostics
(SNR, spectral entropy, Hjorth parameters [@Hjorth1970]) and the
recording-level PaLOSi index [@Hu2025], a measure of cross-spectral
homogeneity that detects over-preprocessing and dominant artefacts.
All metrics are exported to the HTML report and companion CSV files.

## Pipeline optimiser

`find_best_params(data, sfreq)` performs an exhaustive grid search over
candidate values for bandpass cutoffs, filter order, notch frequency,
artefact threshold, and optional processing steps. Each configuration is
scored by a composite metric:

$$\text{score} = \alpha \cdot \overline{\text{SNR}}_{\text{dB}} - (1 - \alpha) \cdot |\text{PaLOSi} - 0.45|$$

where $\alpha$ (default 0.5) balances reconstruction fidelity against
spectral quality. The penalty term drives the pipeline toward the centre
of the ideal PaLOSi range [0.3, 0.6] identified by Hu et al. [@Hu2025].
The returned parameter dictionary unpacks directly into `preprocess()`.

## Output format

Each recording produces a single HDF5 file (gzip level 4) with a fixed
group structure (`/eeg/`, `/behavioral/`, `/physio/`, `/metadata/`).
This layout is compatible with h5py, NumPy, PyTorch, and TensorFlow
data loaders without additional parsing. The `sync()` function extends
the file in-place, so all data for one participant travel as a single
artefact through the analysis pipeline.

# Tests and continuous integration

EVA ships with a test suite of 70 unit tests covering all public
functions and internal helpers (`tests/test_filters.py`,
`test_metrics.py`, `test_pipeline.py`). A GitHub Actions workflow runs
the full suite on Python 3.10, 3.11, and 3.12 on every push and pull
request. Contribution guidelines are documented in `CONTRIBUTING.md`.

# Limitations

EVA is intentionally scoped to the preprocessing stage. It does not
perform blind source separation (e.g. ICA), automatic epoch rejection
beyond soft amplitude clipping, sensor-level source localisation, or
time-frequency decomposition. It is designed for event-related paradigms
with discrete annotations; continuous or resting-state analyses require
downstream tooling such as MNE-Python or EEGLAB. The exhaustive grid
search in `find_best_params()` scales as the product of all candidate
lists and may be slow for large grids or long recordings; users should
reduce the search space via the `grid` parameter when needed.

# Acknowledgements

This work was supported by the Fundação de Amparo à Pesquisa do Estado
de São Paulo (FAPESP), grant 2024/17032-7.

# References
