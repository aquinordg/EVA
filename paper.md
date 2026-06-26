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
optimiser (`find_best_params()`) identifies the filter configuration whose
PaLOSi is closest to the ideal range centre before full processing.

# Statement of need

Preprocessing raw EEG data into analysis-ready epochs involves a repeated
sequence of decisions — frequency cutoffs, reference scheme, artefact
thresholds — that are commonly implemented ad hoc in laboratory-specific
scripts. This fragmentation hinders reproducibility: two researchers
applying nominally identical steps to the same recording may obtain
different results due to implementation details that are rarely reported
in full [@Luck2014].

Established toolboxes such as MNE-Python [@Gramfort2013] and EEGLAB
[@Delorme2004] offer comprehensive environments for EEG analysis, but
their generality places the burden of pipeline assembly on the researcher:
choosing filter parameters, selecting a reference scheme, deciding on
artefact rejection thresholds, and wiring the output to downstream formats
all require domain expertise and produce lab-specific code that is rarely
shared in full. Specialised tools narrow this gap for specific subtasks —
`autoreject` [@Jas2017] optimises epoch rejection thresholds via
cross-validation, and MNE-BIDS [@Appelhoff2019] provides a
configuration-driven batch workflow for BIDS-organised datasets — but
neither addresses data-driven filter selection, hardware-level quality
flags, or multimodal data fusion into a single analysis-ready file.

EVA makes three contributions that are absent from existing tools:

1. **Data-driven filter optimisation.** `find_best_params()` searches a
   configurable grid of preprocessing configurations and selects the one
   whose PaLOSi score [@Hu2025] is closest to the centre of the empirically
   validated ideal range [0.3, 0.6]. This replaces trial-and-error filter
   tuning with a principled, dataset-specific criterion.

2. **Hardware saturation detection.** The `flag_adc_clipping` quality flag
   identifies channels where a measurable fraction of samples reach the
   exact minimum or maximum of the ADC range — a hardware-level artefact
   that SNR-based metrics systematically miss, because clipped signals can
   have normal spectral energy while their waveforms are distorted.

3. **Multimodal epoch fusion.** `sync()` appends per-epoch behavioural and
   physiological co-variables to the same HDF5 file produced by
   `preprocess()`, yielding a single self-contained artefact per participant
   with a fixed group structure (`/eeg/`, `/behavioral/`, `/physio/`,
   `/metadata/`) that loads directly into h5py, NumPy, PyTorch, and
   TensorFlow without additional parsing.

The primary target audience is researchers in cognitive neuroscience and
brain-computer interface development who acquire EEG alongside eye
tracking or physiological signals and broadcast event markers via Lab
Streaming Layer (LSL). EVA was developed to support VECA-EEG [@VECAEEG],
a Unity 6 virtual reality platform for cognitive assessment in which LSL
markers emitted by the stimulus system serve directly as EVA's epoch class
labels — with no remapping required.

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

After filtering, `preprocess()` computes per-channel diagnostics —
SNR (informational only), ADC clipping fraction, spectral entropy,
Hjorth parameters [@Hjorth1970] — and the recording-level PaLOSi index
[@Hu2025], a measure of cross-spectral homogeneity that detects
over-preprocessing and dominant artefacts. Four boolean flags determine
channel status: `flag_flat`, `flag_high_amplitude`, `flag_adc_clipping`,
and `flag_spectral_outlier`. All metrics are exported to the HTML report
and companion CSV files.

## Pipeline optimiser

`find_best_params(data, sfreq)` performs an exhaustive grid search over
candidate values for bandpass cutoffs, filter order, notch frequency,
artefact threshold, and optional processing steps. Each configuration is
scored by:

$$\text{score} = -|\text{PaLOSi} - 0.45|$$

The penalty drives the pipeline toward 0.45, the centre of the ideal
PaLOSi range [0.3, 0.6] identified by Hu et al. [@Hu2025]. SNR is not
used in scoring: for clean EEG, SNR ≈ 0 dB is the expected outcome
(most signal energy already lies within the passband), making it
uninformative as an optimisation criterion. The returned parameter
dictionary unpacks directly into `preprocess()`.

## Output format

Each recording produces a single HDF5 file (gzip level 4) with a fixed
group structure (`/eeg/`, `/behavioral/`, `/physio/`, `/metadata/`).
This layout is compatible with h5py, NumPy, PyTorch, and TensorFlow
data loaders without additional parsing. The `sync()` function extends
the file in-place, so all data for one participant travel as a single
artefact through the analysis pipeline.

# Tests and continuous integration

EVA ships with a test suite of 94 unit tests covering all public
functions and internal helpers (`tests/test_filters.py`,
`test_metrics.py`, `test_pipeline.py`, `test_align.py`). A GitHub
Actions workflow runs the full suite on Python 3.10, 3.11, and 3.12 on
every push and pull request. Contribution guidelines are documented in
`CONTRIBUTING.md`.

# Validation

EVA's default filter chain was validated against three public EEG datasets
using the script provided in `scripts/validation.py`.

**SSVEP (Nakanishi et al., MNE sample dataset).** After applying the
default filter chain (1–40 Hz bandpass, 60 Hz notch, CAR, soft clipper)
to two subjects, the mean ratio of peak power at the stimulus frequencies
(12 Hz and 15 Hz) relative to the detrended raw signal was 2.31,
confirming that the steady-state responses are preserved and sharpened
rather than attenuated.

**PhysioNet EEG Motor Movement/Imagery (EEGMMI).** Resting-state
recordings from five subjects (eyes-open and eyes-closed runs) were
processed and scored with PaLOSi. All ten recordings fell within the ideal
range [0.3, 0.6] (mean PaLOSi = 0.53), indicating that the default
configuration achieves the spectral homogeneity expected of
well-preprocessed EEG on a standard clinical dataset.

**MOABB BCI Competition IV 2a (motor imagery).** Training runs from five
subjects were processed with the 50 Hz notch variant (European mains). All
five subjects yielded PaLOSi within [0.3, 0.6] (mean = 0.55), achieving
100% pass rate against the pre-specified threshold of ≥ 50%.

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
