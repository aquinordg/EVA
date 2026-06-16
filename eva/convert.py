"""Convert EEG recordings from any supported format to MNE .fif.

Optionally detects bad channels on the raw signal and generates a quality
report before any preprocessing is applied.

Usage
-----
>>> from eva import convert

# Basic conversion
>>> convert("subject01.vhdr")
PosixPath('.../subject01.fif')

# With bad-channel detection and HTML report (inspection only)
>>> convert("subject01.vhdr", report=True)
PosixPath('.../subject01.fif')

# Custom output and channel selection
>>> convert("subject01.vhdr", output="converted/subject01.fif",
...         channel_picks=["Fz", "Cz", "Pz"], report=True)
PosixPath('.../converted/subject01.fif')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne

from .metrics import compute_psd, detect_bad_channels

logger = logging.getLogger(__name__)

_LOADERS = {
    "brainvision": mne.io.read_raw_brainvision,  # BrainProducts .vhdr
    "edf":         mne.io.read_raw_edf,           # European Data Format
    "bdf":         mne.io.read_raw_bdf,           # BioSemi BDF
    "eeglab":      mne.io.read_raw_eeglab,        # EEGLAB .set
    "gdf":         mne.io.read_raw_gdf,           # General Data Format (OpenBCI, g.tec)
    "egi":         mne.io.read_raw_egi,           # EGI/Philips .mff or .raw
    "cnt":         mne.io.read_raw_cnt,           # Neuroscan .cnt
    "nihon":       mne.io.read_raw_nihon,         # Nihon Kohden .eeg
    "persyst":     mne.io.read_raw_persyst,       # Persyst .lay
    "curry":       mne.io.read_raw_curry,         # CURRY .cdt / .dat
    "nicolet":     mne.io.read_raw_nicolet,       # Nicolet .data
}

# Auto-detection from file extension. Ambiguous extensions (.raw, .dat, .data,
# .mat) are intentionally excluded — pass input_type explicitly for those.
# Note: .eeg maps to Nihon Kohden; BrainVision .eeg is an internal companion
# file and is never opened directly (use the .vhdr header instead).
_EXT_TO_TYPE = {
    ".vhdr": "brainvision",
    ".edf":  "edf",
    ".bdf":  "bdf",
    ".set":  "eeglab",
    ".gdf":  "gdf",
    ".mff":  "egi",
    ".cnt":  "cnt",
    ".eeg":  "nihon",
    ".lay":  "persyst",
    ".cdt":  "curry",
}


def convert(
    path: Union[str, Path],
    *,
    input_type: str = "auto",
    output: Optional[Union[str, Path]] = None,
    channel_picks: Optional[List[str]] = None,
    report: bool = False,
    report_dir: Optional[Union[str, Path]] = None,
    flat_std_threshold: float = 100e-9,
    high_amplitude_threshold: float = 150e-6,
    log_spectra_dev_threshold: float = 2.0,
) -> Path:
    """
    Convert an EEG recording to MNE .fif format.

    Optionally runs bad-channel detection on the raw signal and generates
    an HTML quality report. No channels are removed or modified — the
    report is for inspection only.

    Parameters
    ----------
    path
        Input file. A plain filename is resolved relative to the current
        working directory.
    input_type
        Source format. ``"auto"`` detects from the file extension.
        Explicit values: ``"brainvision"``, ``"edf"``, ``"bdf"``,
        ``"eeglab"``, ``"gdf"``, ``"egi"``, ``"cnt"`` (Neuroscan),
        ``"nihon"`` (Nihon Kohden), ``"persyst"``, ``"curry"``,
        ``"nicolet"``. Use an explicit value for extensions that are
        ambiguous (e.g. ``input_type="egi"`` for EGI ``.raw`` files,
        ``input_type="curry"`` for CURRY ``.dat`` files).
    output
        Destination .fif file. Defaults to the same directory as *path*,
        with the same stem and a ``.fif`` extension.
    channel_picks
        Channel names to keep. ``None`` retains all EEG channels.
    report
        When ``True``, runs bad-channel detection and generates an HTML
        quality report alongside the .fif. No channels are removed or
        marked automatically — the report is for inspection only.
    report_dir
        Directory for the HTML report. Defaults to a ``reports/<stem>``
        folder next to *output*.
    flat_std_threshold
        Channels with std below this value (V) are flagged as flat/dead.
        Default 100 nV = ``100e-9``.
    high_amplitude_threshold
        Channels with peak amplitude above this value (V) are flagged.
        Default 150 µV = ``150e-6``.
    log_spectra_dev_threshold
        Channels with log-spectra deviation above this value are flagged
        as spectral outliers. Default 2.0.

    Returns
    -------
    Path
        Location of the saved .fif file.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if input_type == "auto":
        input_type = _EXT_TO_TYPE.get(path.suffix.lower())
        if input_type is None:
            supported = ", ".join(_EXT_TO_TYPE)
            raise ValueError(
                f"Cannot detect format from '{path.suffix}'. "
                f"Supported extensions: {supported}. "
                "Set input_type explicitly."
            )

    loader = _LOADERS.get(input_type)
    if loader is None:
        raise ValueError(
            f"Unknown input_type '{input_type}'. "
            f"Choose from: {list(_LOADERS)}."
        )

    raw = loader(str(path), preload=True, verbose=False)

    if channel_picks is not None:
        raw.pick(channel_picks)
    else:
        raw.pick("eeg")

    if output is None:
        output = path.with_suffix(".fif")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- Bad-channel detection (report only — no channels are removed) ---
    if report:
        quality_df = detect_bad_channels(
            raw.ch_names,
            raw.get_data(),
            raw.info["sfreq"],
            flat_std_threshold=flat_std_threshold,
            high_amplitude_threshold=high_amplitude_threshold,
            log_spectra_dev_threshold=log_spectra_dev_threshold,
        )
        rep_dir = Path(report_dir) if report_dir else output.parent / "reports" / output.stem
        _generate_report(rep_dir, path.name, raw, quality_df)

    raw.save(str(output), overwrite=True, verbose=False)
    logger.info("Converted '%s' -> '%s'  (%d channels)", path.name, output, len(raw.ch_names))
    return output


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _fig_raw_psd(data: np.ndarray, sfreq: float, out_path: Path) -> None:
    freqs, psd = compute_psd(data, sfreq)
    psd_uv = psd * 1e12  # V²/Hz → µV²/Hz

    fig, ax = plt.subplots(figsize=(10, 4))
    for row in psd_uv:
        ax.semilogy(freqs, row, alpha=0.3, linewidth=0.7, color="steelblue")
    ax.semilogy(freqs, np.median(psd_uv, axis=0),
                color="black", linewidth=1.8, label="Median across channels")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Power spectral density (µV²/Hz)", fontsize=10)
    ax.set_title("Raw Signal — Power by Frequency\n"
                 "Each thin line is one channel; thick black line is the median",
                 fontsize=10)
    ax.set_xlim(0, min(sfreq / 2.0, 120.0))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _quality_table_html(quality_df) -> str:
    STATUS_BADGE = {
        "good":    '<span style="background:#2ecc71;color:#fff;padding:2px 8px;border-radius:4px">Good</span>',
        "warning": '<span style="background:#f39c12;color:#fff;padding:2px 8px;border-radius:4px">Needs review</span>',
        "bad":     '<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px">Bad</span>',
    }
    cols = ["status", "std_V", "peak_V", "log_spectra_dev", "spectral_entropy",
            "flag_flat", "flag_high_amplitude", "flag_spectral_outlier"]
    labels = {
        "status":                "Status",
        "std_V":                 "Std. Dev. (µV)",
        "peak_V":                "Peak Amplitude (µV)",
        "log_spectra_dev":       "Log-Spectra Deviation",
        "spectral_entropy":      "Spectral Entropy",
        "flag_flat":             "Flat / Dead",
        "flag_high_amplitude":   "High Amplitude",
        "flag_spectral_outlier": "Spectral Outlier",
    }
    uv_cols  = {"std_V", "peak_V"}
    flag_cols = {"flag_flat", "flag_high_amplitude", "flag_spectral_outlier"}

    header = "<tr><th>Channel</th>" + "".join(f"<th>{labels[c]}</th>" for c in cols) + "</tr>"
    rows_html = []
    for ch, row in quality_df.iterrows():
        cells = f"<td><strong>{ch}</strong></td>"
        for col in cols:
            val = row[col]
            if col == "status":
                cells += f"<td>{STATUS_BADGE.get(val, val)}</td>"
            elif col in uv_cols:
                cells += f"<td>{val * 1e6:.3f}</td>"
            elif col in flag_cols:
                cells += f"<td>{'Yes' if val else '—'}</td>"
            else:
                cells += f"<td>{val:.4f}</td>"
        rows_html.append(f"<tr>{cells}</tr>")

    return f"<table><thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>EVA Convert Report &mdash; {stem}</title>
<style>
  body    {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px 60px;
             background: #f4f6f9; color: #212529; line-height: 1.6; }}
  h1      {{ color: #1a252f; margin-bottom: 4px; }}
  h2      {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7;
             padding-bottom: 5px; margin-top: 36px; }}
  .meta   {{ color: #5d6d7e; font-size: 14px; margin-bottom: 6px; }}
  .note   {{ background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #1a5276; }}
  .warn   {{ background: #fef9e7; border-left: 4px solid #f39c12;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #7d6608; }}
  .bad    {{ background: #fdedec; border-left: 4px solid #e74c3c;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #922b21; }}
  table   {{ border-collapse: collapse; width: 100%; margin-bottom: 18px;
             font-size: 13px; background: #fff; }}
  th, td  {{ border: 1px solid #dee2e6; padding: 6px 10px; }}
  th      {{ background: #2c3e50; color: #ecf0f1; text-align: left; font-weight: 600; }}
  td      {{ text-align: right; }}
  tr:nth-child(even) {{ background: #f2f3f4; }}
  img     {{ max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px;
             margin: 10px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                   gap: 12px; margin: 16px 0; }}
  .stat-card  {{ background: #fff; border-radius: 6px; padding: 14px 18px;
                 border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }}
  .stat-card .label {{ font-size: 11px; color: #7f8c8d; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 22px; font-weight: 700; color: #2c3e50; }}
  footer  {{ margin-top: 50px; color: #aab7b8; font-size: 12px;
             border-top: 1px solid #dee2e6; padding-top: 10px; }}
</style>
</head>
<body>

<h1>Channel Quality Report &mdash; Pre-processing</h1>
<p class="meta">EVA &mdash; raw signal analysis before any filtering</p>
<p class="meta">Source file: <strong>{source_name}</strong></p>

<h2>Recording Overview</h2>
<div class="summary-grid">
  <div class="stat-card"><div class="label">Sampling Rate</div><div class="value">{sfreq:.0f} Hz</div></div>
  <div class="stat-card"><div class="label">EEG Channels</div><div class="value">{n_ch}</div></div>
  <div class="stat-card"><div class="label">Duration</div><div class="value" style="font-size:16px">{duration} s</div></div>
  <div class="stat-card"><div class="label">Good</div><div class="value" style="color:#2ecc71">{n_good}</div></div>
  <div class="stat-card"><div class="label">Needs Review</div><div class="value" style="color:#f39c12">{n_warn}</div></div>
  <div class="stat-card"><div class="label">Bad</div><div class="value" style="color:#e74c3c">{n_bad}</div></div>
</div>

<h2>Bad Channel Summary</h2>
{bad_summary}

<h2>Raw Signal &mdash; Frequency Content</h2>
<div class="note">
  Recorded before any filtering. Each thin line is one electrode; the thick
  black line is the median. Channels that deviate visually from the ensemble
  are likely artefact-dominated.
</div>
<img src="raw_psd.png" alt="Raw PSD"/>

<h2>Per-Channel Quality</h2>
<div class="note">
  Three criteria on raw data: flat/dead electrode (std &lt; {flat_std} nV),
  high amplitude (peak &gt; {high_amp} µV), spectral outlier (log-spectra
  deviation &gt; {lsd_thr}). Two or more flags &rarr; Bad.
</div>
{quality_table}

<footer>EVA &mdash; EEG data Validation and preprocessing Assistant</footer>
</body>
</html>
"""


def _generate_report(rep_dir: Path, source_name: str, raw: mne.io.BaseRaw, quality_df) -> None:
    rep_dir.mkdir(parents=True, exist_ok=True)
    stem  = rep_dir.name
    sfreq = raw.info["sfreq"]
    n_ch  = len(raw.ch_names)
    dur_s = raw.times[-1]

    _fig_raw_psd(raw.get_data(), sfreq, rep_dir / "raw_psd.png")
    quality_df.to_csv(rep_dir / "channel_quality.csv", encoding="utf-8")

    bad_chs  = quality_df[quality_df["status"] == "bad"].index.tolist()
    warn_chs = quality_df[quality_df["status"] == "warning"].index.tolist()
    good_n   = int((quality_df["status"] == "good").sum())

    if bad_chs:
        bad_summary = (
            f'<div class="bad"><strong>Channels flagged as Bad ({len(bad_chs)}):</strong> '
            f'{", ".join(bad_chs)}<br/>'
            f'Review these channels before preprocessing. Use <code>channel_picks</code> '
            f'in <code>preprocess()</code> to exclude them if needed.</div>'
        )
    else:
        bad_summary = '<div class="note">No bad channels detected.</div>'

    if warn_chs:
        bad_summary += (
            f'<div class="warn"><strong>Channels needing review ({len(warn_chs)}):</strong> '
            f'{", ".join(warn_chs)}</div>'
        )

    html = _HTML_TEMPLATE.format(
        stem=stem,
        source_name=source_name,
        sfreq=sfreq,
        n_ch=n_ch,
        duration=f"{dur_s:.1f}",
        n_good=good_n,
        n_warn=len(warn_chs),
        n_bad=len(bad_chs),
        bad_summary=bad_summary,
        quality_table=_quality_table_html(quality_df),
        flat_std=f"{100e-9 * 1e9:.0f}",
        high_amp=f"{150e-6 * 1e6:.0f}",
        lsd_thr="2.0",
    )
    (rep_dir / "report.html").write_text(html, encoding="utf-8")
    logger.info("Convert report saved -> %s/report.html", rep_dir)
