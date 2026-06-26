[![PyPI](https://img.shields.io/pypi/v/eva-eeg)](https://pypi.org/project/eva-eeg/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# EVA — EEG data Validation and preprocessing Assistant

EVA is a Python library for preprocessing EEG (electroencephalography) recordings. It accepts recordings in all formats supported by MNE-Python — BrainVision `.vhdr`, EDF, BDF, EEGLAB `.set`, GDF, EGI `.mff`, Neuroscan `.cnt`, Nihon Kohden `.eeg`, Persyst `.lay`, CURRY `.cdt`, Nicolet, and MNE-native `.fif` — applies a configurable filter chain, evaluates per-channel and recording-level signal quality, saves processed epochs as HDF5 archives (`.h5`), and generates self-contained HTML reports — all with a three-function API.

EVA is built on top of [MNE-Python](https://mne.tools), an open-source library for EEG/MEG (magnetoencephalography) analysis.

---

## Installation

```bash
pip install eva-eeg
```

**Requirements:** Python 3.10+, MNE-Python, NumPy, SciPy, pandas, h5py.

---

## Workflow overview

EVA follows a three-step pipeline:

```
Raw file (.vhdr / .edf / ...)
    │
    ▼  convert()   — normalise format, inspect channel quality
    │
    ▼  preprocess() — filter → epoch → save .h5  +  HTML report
    │
    ▼  sync()      — attach behavioural / physiological data to the .h5
    │
    ▼  .h5 file ready for your ML / analysis pipeline
```

---

## Quick start

```python
from eva import convert, preprocess, sync
import numpy as np

# Step 1 — convert to .fif and generate a bad-channel report
convert("subject01.vhdr", report=True)

# Step 2 — apply the filter chain, extract epochs, save to .h5
preprocess("subject01.fif")

# Step 3 (optional) — attach per-epoch behavioural measurements
# rt_array and acc_array must have one value per epoch (1-D numpy arrays)
rt_array  = np.array([0.42, 0.38, 0.51, ...])   # reaction time in seconds
acc_array = np.array([1, 0, 1, ...])             # accuracy (1 = correct)
sync("subject01.h5", behavioral={"rt": rt_array, "accuracy": acc_array})
```

Output files land next to the source file by default:
- `subject01.fif` — after `convert()`
- `subject01.h5` — after `preprocess()`
- `subject01_report/report.html` — HTML quality report

---

## API

### `convert(path, *, ...)`

Converts any MNE-supported EEG format to MNE `.fif`. Optionally runs
bad-channel detection and generates an HTML report. **No channels are
removed automatically** — the report is for inspection only; you decide
which channels to exclude.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | str / Path | — | Input file path |
| `input_type` | str | `"auto"` | `"auto"` detects from extension. Explicit: `"brainvision"`, `"edf"`, `"bdf"`, `"eeglab"`, `"gdf"`, `"egi"`, `"cnt"`, `"nihon"`, `"persyst"`, `"curry"`, `"nicolet"`. Use explicit value for ambiguous extensions (e.g. `input_type="egi"` for `.raw`, `input_type="curry"` for `.dat`). |
| `output` | str / Path | same dir as input | Destination `.fif` path |
| `channel_picks` | list[str] | `None` (keep all) | Channel names to keep, e.g. `["Fz", "Cz", "Pz"]` |
| `report` | bool | `False` | Generate HTML bad-channel report |
| `report_dir` | str / Path | same dir as input | Where to save the report folder |
| `flat_std_threshold` | float | `100e-9` (0.1 µV) | Flag channels with std below this value |
| `high_amplitude_threshold` | float | `150e-6` (150 µV) | Flag channels with peaks above this value |
| `log_spectra_dev_threshold` | float | `2.0` | Flag channels whose spectrum deviates more than this many times the median |

```python
from eva import convert

convert("subject01.vhdr")                      # basic conversion, no report

convert("subject01.vhdr", report=True)         # conversion + bad-channel report

convert("subject01.vhdr", report=True,
        flat_std_threshold=100e-9,
        high_amplitude_threshold=150e-6,
        log_spectra_dev_threshold=2.0)

convert("subject01.vhdr",
        output="data/s01.fif",
        channel_picks=["Fz", "Cz", "Pz"])
```

### `preprocess(source, *, ...)`

Applies the filter chain, extracts stimulus-locked epochs, and saves the
result as a compressed `.h5` archive. Accepts a file path or an `mne.Raw`
object directly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | str / Path / mne.Raw | — | Input file or pre-loaded Raw object |
| `l_freq` | float | `1.0` | High-pass cutoff in Hz |
| `h_freq` | float | `40.0` | Low-pass cutoff in Hz |
| `filter_order` | int | `4` | Butterworth filter order per direction |
| `notch_freq` | float / None | `60.0` | Notch frequency in Hz; `None` disables it |
| `artifact_threshold` | float | `100e-6` (100 µV) | Soft-clip ceiling in volts |
| `use_avg_ref` | bool | `True` | Apply Common Average Reference (CAR) |
| `use_soft_clip` | bool | `True` | Apply soft amplitude clipping |
| `epoch_tmin` | float | `0.0` | Epoch start in seconds relative to each event |
| `epoch_tmax` | float | `1.0` | Epoch end in seconds relative to each event |
| `channel_picks` | list[str] | `None` (keep all) | Channel names to keep |
| `diagnostics` | QualityConfig | `None` (defaults) | Custom quality thresholds |
| `optimize` | bool | `False` | Run grid search to find best filter parameters first |
| `report` | bool | `True` | Generate HTML quality report |
| `output` | str / Path | same dir as source | Destination `.h5` path |
| `report_dir` | str / Path | same dir as source | Where to save the report folder |

```python
from eva import preprocess

preprocess("subject01.fif")                    # all defaults, report generated

preprocess("subject01.fif", optimize=True)     # auto-tune filter parameters

preprocess("subject01.fif", report=False)      # skip report (faster)

preprocess("subject01.fif",
           l_freq=0.5,
           h_freq=30.0,
           epoch_tmin=-0.2,       # 200 ms before each event
           epoch_tmax=0.8,        # 800 ms after each event
           output="results/subject01.h5",
           report_dir="results/reports/subject01")

# Pass an already-loaded MNE Raw object
import mne
raw = mne.io.read_raw_brainvision("subject01.vhdr", preload=True)
preprocess(raw, optimize=True)
```

### `sync(path, *, ...)`

Adds behavioural and/or physiological signals to an existing `.h5` file.
Each array must have the same number of rows as there are epochs in the file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | str / Path | — | Path to the `.h5` file from `preprocess()` |
| `behavioral` | dict[str, ndarray] | `None` | Per-epoch measurements; each array shape `(n_epochs,)` or `(n_epochs, n_features)` |
| `physio` | dict[str, ndarray] | `None` | Physiological signals; each array shape `(n_epochs, n_times)` or `(n_epochs, n_ch, n_times)` |
| `physio_sfreq` | float / dict | `None` | Sampling rate(s) for `physio` signals in Hz; single float or `{"ecg": 1000.0, ...}` |
| `overwrite` | bool | `False` | Replace existing keys instead of raising an error |

```python
from eva import sync
import numpy as np

# Attach only behavioural data
sync("subject01.h5",
     behavioral={"rt": rt_array, "accuracy": acc_array})

# Attach physiological data recorded on a separate device
# physio_sfreq is the sampling rate of the physiological signal (Hz)
sync("subject01.h5",
     physio={"ecg": ecg_array},   # ECG = electrocardiogram
     physio_sfreq=1000.0)

# Both at once; each physio signal can have a different sampling rate
sync("subject01.h5",
     behavioral={"rt": rt_array},
     physio={"ecg": ecg_array, "emg": emg_array},   # EMG = electromyogram
     physio_sfreq={"ecg": 1000.0, "emg": 2000.0})

# Replace previously saved data
sync("subject01.h5", behavioral={"rt": corrected_rt}, overwrite=True)
```

### `align_veca(vhdr_path, csv_path)`

Aligns a VECA-EEG trial CSV with a BrainVision recording, using the
Windows system clock as the common time base. Injects trial annotations
into the returned `mne.Raw` object so that `preprocess()` can epoch
around each cognitive task.

| Parameter | Type | Description |
|---|---|---|
| `vhdr_path` | str / Path | Path to the BrainVision header file (`.vhdr`). The companion `.vmrk` must be in the same directory. |
| `csv_path` | str / Path | Path to the VECA-EEG results CSV (`VECA_<ID>_<timestamp>.csv`). Required columns: `trial_start`, `trial_end`, `feature`, `value`. |

**Returns** `(raw, trials)`:
- `raw` — `mne.io.BaseRaw` with VECA trial annotations injected
- `trials` — `DataFrame` with columns `feature`, `value`, `onset_s`, `duration_s`, `onset_sample`

```python
from eva import align_veca, preprocess, sync

# 1. Align CSV timestamps with the BrainVision recording
raw, trials = align_veca("VECA_XK4TW2.vhdr",
                          "VECA_XK4TW2_20260626_143000.csv")

# 2. Preprocess and epoch (one epoch per trial, 0–8 s window)
preprocess(raw, epoch_tmin=0.0, epoch_tmax=8.0)
# Output: VECA_XK4TW2.h5

# 3. Attach per-trial VECA scores to the .h5
sync("VECA_XK4TW2.h5", behavioral={"score": trials["value"].values})
```

**How the alignment works:** `align_veca()` reads the `New Segment`
timestamp from the `.vmrk` file (recording start, µs resolution) and
computes `onset_s = (trial_start_datetime − recording_start).total_seconds()`
for each row in the CSV. This method does not require an active LSL
connection during recording or analysis.

**Raises:**
- `FileNotFoundError` — if `.vhdr`, `.vmrk`, or CSV is missing
- `ValueError` — if required CSV columns are absent, or if any trial falls outside the recording window

---

### `QualityConfig` — custom channel quality thresholds

`QualityConfig` is a configuration object that controls how EVA classifies
each channel as **good**, **warning**, or **bad**. It holds four thresholds;
a channel raises one flag per threshold it exceeds, and the final status
depends on the number of flags:

| Flags raised | Status | Meaning |
|---|---|---|
| 0 | **good** | Channel is clean |
| 1 | **warning** | One criterion exceeded — worth inspecting |
| ≥ 2 | **bad** | Channel is likely artefact-dominated or dead |

You only need `QualityConfig` when the defaults do not fit your data. For
example, if your amplifier produces higher baseline amplitudes and the
default 150 µV threshold is flagging healthy channels as bad, raise it:

| Parameter | Default | What triggers the flag |
|---|---|---|
| `flat_std_threshold` | `100e-9` (0.1 µV) | Channel std below this — dead electrode |
| `high_amplitude_threshold` | `150e-6` (150 µV) | Peak amplitude above this — large artefact |
| `log_spectra_dev_threshold` | `2.0` | Spectrum deviates more than this × the median |
| `adc_clip_fraction_threshold` | `0.001` (0.1%) | Fraction of samples stuck at exact min/max above this — ADC saturation |

```python
from eva import preprocess, QualityConfig

# Example: loosen amplitude threshold for a high-impedance setup
cfg = QualityConfig(
    high_amplitude_threshold=300e-6,         # accept up to 300 µV
    adc_clip_fraction_threshold=0.005,       # tolerate up to 0.5% clipping
)
preprocess("subject01.fif", diagnostics=cfg)
```

If you do not pass `diagnostics=`, EVA uses the defaults listed above.

---

## Filter chain

The following steps are applied in order. Each class is independent and
exposes `apply(data, sfreq) -> ndarray`, where `data` has shape
`(n_channels, n_samples)` and values are in **volts** (as returned by MNE).

| Step | Class | What it does |
|---|---|---|
| DC removal | `DCDetrend` | Subtracts the per-channel mean to remove electrode offset |
| Bandpass | `ButterworthFilter` | Zero-phase IIR (Infinite Impulse Response) filter; keeps only frequencies between `l_freq` and `h_freq` |
| Notch | `NotchFilter` | Removes power-line interference (default: 60 Hz) |
| Average reference | `AverageReference` | Common Average Reference (CAR) — subtracts the mean across all channels at each time point to suppress spatially diffuse noise |
| Soft clip | `SoftClipper` | Smoothly limits extreme amplitude spikes using a tanh curve; less disruptive than hard zeroing |

**Default parameters:** `l_freq=1.0 Hz`, `h_freq=40.0 Hz`, `order=4`, `notch=60.0 Hz`, `threshold=100 µV`.

Both `ButterworthFilter` and `NotchFilter` use the second-order sections
(SOS) representation internally. This keeps the filters numerically stable
for any combination of cutoff frequency and sampling rate — including cases
like `l_freq=0.1 Hz` at `sfreq=1024 Hz` where the standard direct-form
implementation overflows.

**Using the filters directly** (independent of the full pipeline):

```python
import mne
from eva.filters import (
    DCDetrend, ButterworthFilter, NotchFilter, AverageReference, SoftClipper
)

raw = mne.io.read_raw_brainvision("subject01.vhdr", preload=True)
data  = raw.get_data()    # shape: (n_channels, n_samples), in volts
sfreq = raw.info["sfreq"] # sampling frequency in Hz

steps = [
    DCDetrend(),
    ButterworthFilter(l_freq=1.0, h_freq=40.0, order=4),
    NotchFilter(freq=60.0),
    AverageReference(),
    SoftClipper(threshold=100e-6),   # 100 µV expressed in volts
]
for step in steps:
    data = step.apply(data, sfreq)
```

---

## Quality metrics

### Per-channel metrics

EVA computes the following metrics for each channel and stores them in the
report and in `channel_quality.csv`:

| Metric | What it measures |
|---|---|
| **SNR — Signal-to-Noise Ratio (dB)** | How much the filter changed the signal: `10·log₁₀(Var(raw) / Var(raw − processed))`. Reported for information only — values near 0 are normal for clean EEG where most signal energy already lies within the passband. |
| **ADC clipping fraction** | Fraction of samples stuck at the exact minimum or maximum value, indicating amplifier saturation during recording. Values above ~0.1% are a reliable indicator of ADC clipping. |
| **Log-Spectra Deviation** | How much a channel's power spectrum deviates from the median spectrum across all channels. High values point to outlier channels (artefact-dominated or dead). |
| **Spectral entropy** | How flat (broadband) the channel's spectrum is. White noise has entropy ≈ 1; a clean EEG with dominant alpha rhythm has lower entropy. |
| **Hjorth activity** | Signal variance — a simple proxy for signal power. |
| **Hjorth mobility** | Ratio of the derivative's standard deviation to the signal's. Relates to the mean frequency. |
| **Hjorth complexity** | How much the signal's waveform complexity changes over time. |

Each channel is assigned a status based on the number of quality flags raised:

| Status | Flags | Meaning |
|---|---|---|
| **good** | 0 | All criteria within thresholds |
| **warning** | 1 | One criterion exceeded — worth inspecting |
| **bad** | ≥ 2 | Likely artefact-dominated or dead electrode |

Flags: `flag_flat` (nearly zero variance — dead electrode), `flag_high_amplitude`
(extreme peaks — movement or sweat artefact), `flag_adc_clipping` (ADC saturation),
`flag_spectral_outlier` (spectrum far from channel ensemble).

### Recording-level metric — PaLOSi

**PaLOSi** (Parallel LOg Spectra index, Hu et al. 2025) summarises the
preprocessing quality of the whole recording with a single number between
0 and 1. It measures how much the cross-channel spectral structure is
dominated by a single spatial component: too little filtering leaves
broadband noise across all channels (low PaLOSi); too much filtering
erases genuine neural activity (high PaLOSi).

| PaLOSi range | Interpretation |
|---|---|
| < 0.3 | Insufficient filtering — residual noise still dominates |
| **0.3 – 0.6** | **Ideal — good balance between denoising and signal preservation** |
| > 0.6 | Over-filtered — too much signal structure has been removed |

The HTML report shows a colour-coded PaLOSi card with an explanatory
message. Reference: [Hu et al. (2025) *NeuroImage* 121247](https://doi.org/10.1016/j.neuroimage.2025.121247).

---

## Optimiser

When `optimize=True`, EVA tries every combination in the default grid
below, scores each with the formula:

```
score = α × mean_SNR  −  (1 − α) × |PaLOSi − 0.45|
```

**α (alpha)** is a weight between 0 and 1 that you set via the `alpha`
parameter (default `0.5`). It controls the balance between two goals:

- **α = 1.0** → optimise for SNR only (maximise noise removed)
- **α = 0.0** → optimise for PaLOSi only (target the ideal [0.3, 0.6] range)
- **α = 0.5** → equal weight to both (recommended starting point)

The `|PaLOSi − 0.45|` term penalises any departure from the centre of
the ideal PaLOSi range (0.45 = midpoint of [0.3, 0.6]). The winning
configuration is applied automatically, overriding any filter parameters
you passed explicitly.

Parameters searched (default grid):

| Parameter | Candidates | `None` means |
|---|---|---|
| High-pass cutoff (`l_freq`) | `None`, 0.5, 1.0, 2.0 Hz | no high-pass filter |
| Low-pass cutoff (`h_freq`) | 30.0, 40.0, 50.0 Hz | — |
| Filter order | 4, 6 | — |
| Notch frequency (`notch_freq`) | `None`, 50.0, 60.0 Hz | no notch filter |
| Soft-clip threshold | 75, 100, 150 µV | — |
| Use soft clip (`use_soft_clip`) | `True`, `False` | — |
| Use CAR (`use_avg_ref`) | `True`, `False` | — |

Total combinations: 4 × 3 × 2 × 3 × 3 × 2 × 2 = **864 configurations**.

```python
from eva import preprocess
preprocess("subject01.fif", optimize=True)          # alpha=0.5 (default)
preprocess("subject01.fif", optimize=True, alpha=0.8)  # favour SNR
```

---

## Output format (.h5)

Each recording is saved as a single `.h5` file (HDF5 format, readable with
[h5py](https://www.h5py.org/) or any HDF5-compatible tool):

```
subject01.h5
  /eeg/
    data        (n_epochs, n_channels, n_times)  float32  gzip-compressed
    labels      (n_epochs,)                       int32    integer event codes
    ch_names    (n_channels,)                     str      electrode names
    label_names (n_classes,)                      str      condition names
    label_codes (n_classes,)                      int32    codes matching labels
  /behavioral/
    rt          (n_epochs,)                                reaction time, etc.
    accuracy    (n_epochs,)
  /physio/
    ecg         (n_epochs, n_times)               attr: sfreq (Hz)
  /metadata/
    attrs: sfreq, tmin, tmax
```

**Reading the file:**

```python
import h5py
import numpy as np

with h5py.File("subject01.h5", "r") as f:
    data      = f["eeg/data"][:]        # (n_epochs, n_channels, n_times)
    labels    = f["eeg/labels"][:]      # integer class codes per epoch
    ch_names  = f["eeg/ch_names"][:].astype(str)
    sfreq     = f["metadata"].attrs["sfreq"]
    tmin      = f["metadata"].attrs["tmin"]

print(data.shape, np.unique(labels))
```

---

## Validation

EVA was tested on three independent public datasets covering different EEG
paradigms.

| Dataset | N | Paradigm | Test | Result |
|---|---|---|---|---|
| MNE SSVEP (Nakanishi et al.) | 2 subjects, 32 ch, 1000 Hz | SSVEP (Steady-State Visual Evoked Potential) — visual flicker at 12 and 15 Hz | Peak power preserved ≥ 75% at both frequencies after filtering | **PASS** — mean ratio 0.797 |
| PhysioNet EEGMMI | 5 subjects, 64 ch, 160 Hz | Resting state (eyes open / closed) | PaLOSi in [0.3, 0.6] for ≥ 50% of recordings | **PASS** — 50% in range |
| MOABB BCI Competition IV 2a | 5 subjects, 22 ch, 250 Hz | Motor imagery — 4 classes (BCI, Brain-Computer Interface) | PaLOSi in [0.3, 0.6] for ≥ 50% of trials | **PASS** — 97% in range, mean PaLOSi 0.540 |

> **Note on motor imagery and band power:** Applying Common Average Reference
> to motor imagery data substantially reduces the absolute power in the mu
> (8–12 Hz) and beta (18–25 Hz) bands because CAR removes correlated
> broadband noise shared across channels. This is the intended behaviour —
> the neural structure is preserved (confirmed by PaLOSi in range). A band
> power ratio test is only meaningful for paradigms where the signal is
> externally driven and spectrally narrow (e.g. SSVEP).

---

## Limitations

- **No artefact decomposition (ICA).** Independent Component Analysis (ICA)
  separates ocular (eye blink), muscular, and cardiac artefacts from neural
  activity. EVA does not include ICA. The soft clipper attenuates extreme
  transients but does not remove structured biological artefacts. For studies
  where blink or movement artefacts are a concern, run ICA with MNE after
  `preprocess()`.

- **High-pass cutoff and slow brain responses.** The default `l_freq=1.0 Hz`
  can attenuate low-frequency ERP (Event-Related Potential) components such
  as the P3 or N400 by up to ~30% compared to a 0.1 Hz high-pass. For ERP
  studies, set
  `l_freq=0.1` explicitly — EVA's SOS filter handles this without numerical
  issues.

- **Power-line frequency must be set manually.** The default notch filter
  targets 60 Hz (Americas and most of Asia). European recordings use 50 Hz
  mains frequency and require `notch_freq=50.0`. EVA does not auto-detect
  this.

- **No epoch rejection.** Epochs with extreme amplitudes are soft-clipped,
  not discarded. If your downstream model requires artefact-free epochs,
  apply additional rejection after loading the `.h5` file.

- **CAR requires a full, clean channel set.** Common Average Reference works
  best when artefacts are spread evenly across channels. If one or more
  channels are severely contaminated, their contribution to the mean will
  spread noise to all other channels. Exclude known bad channels with
  `channel_picks` before preprocessing.

- **PaLOSi range was established on resting-state data.** The [0.3, 0.6]
  ideal range (Hu et al. 2025) is defined for eyes-open/closed resting-state
  recordings. Task-driven paradigms such as motor imagery or SSVEP may show
  PaLOSi values slightly above 0.6 without indicating a problem.

---

## Module structure

```
eva/
├── __init__.py    Public API: convert, preprocess, sync, QualityConfig, align_veca
├── align.py       align_veca() — VECA-EEG CSV-to-BrainVision alignment
├── convert.py     Format normalisation to .fif
├── preprocess.py  Filter chain, epoching, .h5 output, HTML report
├── sync.py        Attach behavioural/physio data to an existing .h5
├── optimizer.py   Grid search over filter strategies
├── filters.py     DCDetrend, ButterworthFilter, NotchFilter,
│                  AverageReference, SoftClipper
├── metrics.py     snr_db, adc_clipping_fraction, palosi, spectral_entropy,
│                  hjorth_parameters, QualityConfig, evaluate_all_channels
└── report.py      HTML + CSV report generation
```

---

## Glossary

| Acronym | Full name | Brief description |
|---|---|---|
| **API** | Application Programming Interface | The set of functions a library exposes to the user |
| **BCI** | Brain-Computer Interface | System that translates brain signals directly into computer commands |
| **BDF** | BioSemi Data Format | Binary EEG file format used by BioSemi amplifiers (`.bdf`) |
| **CAR** | Common Average Reference | Referencing scheme that subtracts the instantaneous mean across all channels |
| **ECG** | Electrocardiogram | Recording of the heart's electrical activity |
| **EDF** | European Data Format | Standard binary format for biosignal storage (`.edf`) |
| **EEG** | Electroencephalography | Measurement of brain electrical activity via scalp electrodes |
| **EMG** | Electromyogram | Recording of muscle electrical activity |
| **ERP** | Event-Related Potential | Brain response time-locked to a stimulus or event (e.g. P3, N400) |
| **HDF5** | Hierarchical Data Format 5 | Binary file format for storing large arrays with compression (`.h5`) |
| **ICA** | Independent Component Analysis | Signal decomposition technique used to separate artefacts from neural sources |
| **IIR** | Infinite Impulse Response | Class of digital filter with recursive feedback; efficient but requires zero-phase correction |
| **MEG** | Magnetoencephalography | Measurement of the magnetic fields produced by brain activity |
| **ML** | Machine Learning | — |
| **MOABB** | Mother of All BCI Benchmarks | Open-source Python framework for benchmarking BCI algorithms on public datasets |
| **MNE** | — | Open-source Python library for EEG/MEG analysis ([mne.tools](https://mne.tools)) |
| **PaLOSi** | Parallel LOg Spectra index | Recording-level preprocessing quality metric based on the cross-spectral matrix (Hu et al. 2025) |
| **SNR** | Signal-to-Noise Ratio | Ratio of signal power to noise power, expressed in decibels (dB) |
| **SOS** | Second-Order Sections | Numerically stable representation of IIR filters as a cascade of second-order stages |
| **SSVEP** | Steady-State Visual Evoked Potential | Sustained brain response to a flickering visual stimulus at a fixed frequency |

---

## License

MIT License. See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome. Fork the repository, create a feature branch,
and open a pull request. For questions, contact
[aquinordga@gmail.com](mailto:aquinordga@gmail.com).

---

## Author

Developed by AQUINO, R. D. G.
[![Lattes](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-plataforma-lattes-32.png)](http://lattes.cnpq.br/2373005809061037)
[![ORCID](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-orcid-32.png)](https://orcid.org/0000-0002-8486-8354)
[![Google Scholar](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-google-scholar-32.png)](https://scholar.google.com/citations?user=r5WsvKgAAAAJ&hl)
