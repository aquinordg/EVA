"""Main preprocessing function for EEG recordings.

Usage
-----
Minimal (all defaults):
>>> from eva import preprocess
>>> preprocess("subject01.fif")

Custom parameters with report:
>>> preprocess("subject01.fif", l_freq=0.5, h_freq=30.0,
...            epoch_tmin=-0.2, epoch_tmax=0.8, report=True)

Auto-optimised preprocessing:
>>> preprocess("subject01.fif", optimize=True, report=True)

From an mne.Raw object (skips file loading):
>>> import mne
>>> raw = mne.io.read_raw_brainvision("subject01.vhdr", preload=True)
>>> preprocess(raw, optimize=True, output="results/subject01.h5")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from .filters import AverageReference, ButterworthFilter, DCDetrend, NotchFilter, SoftClipper
from .metrics import QualityConfig, evaluate_all_channels, palosi
from .report import ReportGenerator

logger = logging.getLogger(__name__)

# Supported formats for direct loading (without convert() first).
# Mirrors convert._EXT_TO_TYPE plus .fif (MNE native).
_DIRECT_LOADERS = {
    ".fif":  mne.io.read_raw_fif,
    ".vhdr": mne.io.read_raw_brainvision,
    ".edf":  mne.io.read_raw_edf,
    ".bdf":  mne.io.read_raw_bdf,
    ".set":  mne.io.read_raw_eeglab,
    ".gdf":  mne.io.read_raw_gdf,
    ".mff":  mne.io.read_raw_egi,
    ".cnt":  mne.io.read_raw_cnt,
    ".eeg":  mne.io.read_raw_nihon,
    ".lay":  mne.io.read_raw_persyst,
    ".cdt":  mne.io.read_raw_curry,
}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def preprocess(
    source: Union[str, Path, mne.io.BaseRaw],
    *,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    filter_order: int = 4,
    notch_freq: Optional[float] = 60.0,
    artifact_threshold: float = 100e-6,
    use_avg_ref: bool = True,
    use_soft_clip: bool = True,
    epoch_tmin: float = 0.0,
    epoch_tmax: float = 1.0,
    channel_picks: Optional[List[str]] = None,
    diagnostics: Optional[QualityConfig] = None,
    optimize: bool = False,
    alpha: float = 0.5,
    report: bool = True,
    output: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Preprocess an EEG recording and optionally generate a quality report.

    Parameters
    ----------
    source
        File path (str or Path) or a pre-loaded ``mne.Raw`` object.
        Supported extensions for auto-detection: ``.fif``, ``.vhdr``,
        ``.edf``, ``.bdf``, ``.set``, ``.gdf``, ``.mff``, ``.cnt``
        (Neuroscan), ``.eeg`` (Nihon Kohden), ``.lay`` (Persyst),
        ``.cdt`` (CURRY). For ambiguous extensions pass a pre-loaded
        ``mne.Raw`` object or use :func:`convert` first.
        A plain filename is resolved relative to the current working directory.
    l_freq, h_freq
        Butterworth bandpass cutoffs (Hz).
    filter_order
        Filter order per pass direction (effective order = 2× after zero-phase
        filtering with filtfilt).
    notch_freq
        Power-line notch frequency (Hz). 60 Hz for Americas/Asia, 50 Hz for Europe.
        ``None`` skips the notch filter entirely.
    artifact_threshold
        Soft-clipping ceiling in volts (e.g. ``100e-6`` = 100 µV).
    use_avg_ref
        Apply Common Average Reference after bandpass/notch.
    use_soft_clip
        Apply soft amplitude clipping after average reference.
    epoch_tmin, epoch_tmax
        Epoch window in seconds, relative to event onset.
    channel_picks
        Channel names to retain. ``None`` keeps all EEG channels.
    diagnostics
        Quality thresholds for per-channel evaluation. Uses defaults if None.
    optimize
        When ``True``, runs a grid search over preprocessing strategies and
        applies the best-scoring one (highest SNR + PaLOSi composite score),
        overriding the filter/clipping parameters provided above.
    alpha
        Weight for the optimizer scoring function (0–1). Only used when
        ``optimize=True``. ``alpha=1.0`` maximises SNR; ``alpha=0.0``
        minimises deviation from PaLOSi target (0.45). Default ``0.5``.
    report
        When ``True``, generates an HTML quality report alongside the .h5 output.
    output
        Destination for the .h5 file. Defaults to the same directory as
        *source*, with the same stem and a ``.h5`` extension.
    report_dir
        Directory for the HTML report. Defaults to a ``reports/<stem>``
        folder next to *output*.

    Returns
    -------
    Path
        Location of the saved .h5 file.
    """
    raw, stem, source_dir = _load_source(source, channel_picks)
    sfreq = raw.info["sfreq"]

    if optimize:
        from .optimizer import find_best_params
        logger.info("Running grid search for '%s'...", stem)
        best = find_best_params(raw.get_data(), sfreq, alpha=alpha)
        l_freq             = best["l_freq"]
        h_freq             = best["h_freq"]
        filter_order       = best["filter_order"]
        notch_freq         = best["notch_freq"]
        artifact_threshold = best["artifact_threshold"]
        use_avg_ref        = best["use_avg_ref"]
        use_soft_clip      = best["use_soft_clip"]
        logger.info("Optimisation complete -> %s", best)

    chain = _build_chain(l_freq, h_freq, filter_order, notch_freq,
                         artifact_threshold, use_avg_ref, use_soft_clip)
    raw_data = raw.get_data()
    raw_detrended, processed = _apply_chain(chain, raw_data, sfreq)

    epochs, event_id = _make_epochs(raw, processed, epoch_tmin, epoch_tmax)

    diag = diagnostics or QualityConfig()
    quality_df = evaluate_all_channels(raw.ch_names, raw_detrended, processed, sfreq, diag)
    palosi_score = palosi(processed, sfreq, fmin=l_freq or 1.0, fmax=h_freq or 40.0)

    n_bad  = int((quality_df["status"] == "bad").sum())
    n_warn = int((quality_df["status"] == "warning").sum())
    logger.info(
        "'%s' — %d good  %d warning  %d bad  PaLOSi=%.3f",
        stem, len(raw.ch_names) - n_bad - n_warn, n_warn, n_bad, palosi_score,
    )

    out_path = _resolve_output(output, source_dir, stem)
    _save_h5(out_path, epochs, event_id, sfreq, epoch_tmin, epoch_tmax)

    if report:
        rep_dir = Path(report_dir) if report_dir else out_path.parent / "reports" / stem
        params = dict(
            l_freq=l_freq, h_freq=h_freq, filter_order=filter_order,
            notch_freq=notch_freq, artifact_threshold=artifact_threshold,
            use_avg_ref=use_avg_ref, use_soft_clip=use_soft_clip,
            epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
            channel_picks=channel_picks,
        )
        artefact = dict(
            stem=stem, sfreq=sfreq, ch_names=raw.ch_names,
            raw_arr=raw_data, processed_arr=processed,
            quality_df=quality_df, palosi_recording=palosi_score,
            event_id=event_id, epochs_proc=epochs,
            n_epochs=len(epochs), tmin=epoch_tmin, tmax=epoch_tmax,
        )
        ReportGenerator(rep_dir, params).generate([artefact])

    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_source(
    source: Union[str, Path, mne.io.BaseRaw],
    channel_picks: Optional[List[str]],
) -> Tuple[mne.io.BaseRaw, str, Path]:
    """Load raw from a file path or pass through an mne.Raw object."""
    if isinstance(source, (str, Path)):
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        loader = _DIRECT_LOADERS.get(path.suffix.lower())
        if loader is None:
            supported = ", ".join(_DIRECT_LOADERS)
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {supported}."
            )
        raw = loader(str(path), preload=True, verbose=False)
        stem = path.stem
        source_dir = path.parent
    else:
        raw = source
        if not raw.preload:
            raw.load_data(verbose=False)
        fnames = getattr(raw, "filenames", ())
        stem = Path(fnames[0]).stem if fnames else "recording"
        source_dir = Path(fnames[0]).parent if fnames else Path.cwd()

    if channel_picks is not None:
        raw.pick(channel_picks)
    else:
        raw.pick("eeg")

    return raw, stem, source_dir


def _build_chain(
    l_freq, h_freq, filter_order, notch_freq,
    artifact_threshold, use_avg_ref, use_soft_clip,
) -> list:
    steps = [DCDetrend()]
    steps.append(ButterworthFilter(l_freq=l_freq, h_freq=h_freq, order=filter_order))
    if notch_freq is not None:
        steps.append(NotchFilter(freq=notch_freq))
    if use_avg_ref:
        steps.append(AverageReference())
    if use_soft_clip:
        steps.append(SoftClipper(threshold=artifact_threshold))
    return steps


def _apply_chain(
    chain: list,
    data: np.ndarray,
    sfreq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dc_detrended_raw, fully_processed) — both used for quality metrics."""
    raw_detrended = DCDetrend().apply(data, sfreq)
    processed = data.copy()
    for step in chain:
        processed = step.apply(processed, sfreq)
    return raw_detrended, processed


def _make_epochs(
    raw: mne.io.BaseRaw,
    processed: np.ndarray,
    tmin: float,
    tmax: float,
) -> Tuple[mne.Epochs, Dict[str, int]]:
    """Extract epochs from the processed signal using the original annotations."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if len(events) == 0:
        raise RuntimeError(
            "No events found in the recording. "
            "Ensure the file contains event annotations."
        )

    raw_proc = mne.io.RawArray(processed, raw.info, verbose=False)
    raw_proc.set_annotations(raw.annotations)
    events_proc, _ = mne.events_from_annotations(raw_proc, event_id=event_id, verbose=False)

    epochs = mne.Epochs(
        raw_proc, events=events_proc, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False,
    )
    return epochs, event_id


def _resolve_output(
    output: Optional[Union[str, Path]],
    source_dir: Path,
    stem: str,
) -> Path:
    if output is not None:
        return Path(output).resolve()
    return source_dir / f"{stem}.h5"


def _save_h5(
    out_path: Path,
    epochs: mne.Epochs,
    event_id: Dict[str, int],
    sfreq: float,
    tmin: float,
    tmax: float,
) -> None:
    import h5py

    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as f:
        eeg = f.create_group("eeg")
        eeg.create_dataset(
            "data",
            data=epochs.get_data().astype(np.float32),
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        eeg.create_dataset("labels", data=epochs.events[:, 2].astype(np.int32))
        eeg.create_dataset("ch_names",    data=np.array(epochs.ch_names,         dtype=object), dtype=str_dt)
        eeg.create_dataset("label_names", data=np.array(list(event_id.keys()),   dtype=object), dtype=str_dt)
        eeg.create_dataset("label_codes", data=np.array(list(event_id.values()), dtype=np.int32))

        meta = f.create_group("metadata")
        meta.attrs["sfreq"] = np.float32(sfreq)
        meta.attrs["tmin"]  = np.float32(tmin)
        meta.attrs["tmax"]  = np.float32(tmax)

    logger.info("Saved %d epochs -> %s", len(epochs), out_path)
