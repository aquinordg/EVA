"""
Exploratory data analysis report generator.

Produces one self-contained HTML report per recording, plus companion
CSV tables, covering:

Pre-processing analysis
    Global per-channel statistics (mean, std, min, max, peak amplitude).
    Power spectral density overview.

Post-processing analysis
    Same statistics on the filtered signal.
    Per-channel quality metrics (SNR, PaLOSi, MAE, MSE, Hjorth, spectral
    entropy) with colour-coded status (good / warning / bad).

Dataset-level analysis
    Epoch count and proportion per class.
    Class imbalance analysis: majority/minority ratio, balance entropy.
    Applied pipeline parameters.

Design note
-----------
All figures are rendered with the "Agg" (non-interactive) Matplotlib
backend so that the report can be generated in headless / CI environments
without a display.  All amplitude values in the report are expressed in
microvolts (µV) for readability; the underlying pipeline and CSV files
use SI volts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import compute_psd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

# Human-readable labels for quality metric columns
_QUALITY_LABELS: Dict[str, str] = {
    "status":               "Status",
    "snr_db":               "SNR (dB)",
    "mae_V":                "MAE (µV)",
    "mse_V2":               "MSE (µV²)",
    "log_spectra_dev":      "Log-Spectra Deviation",
    "spectral_entropy":     "Spectral Entropy",
    "hjorth_activity":      "Hjorth Activity (µV²)",
    "hjorth_mobility":      "Hjorth Mobility",
    "hjorth_complexity":    "Hjorth Complexity",
    "std_V":                "Std. Deviation (µV)",
    "peak_V":               "Peak Amplitude (µV)",
    "flag_flat":            "Flat / Dead Channel",
    "flag_high_amplitude":  "High Amplitude",
    "flag_low_snr":         "Low SNR",
    "flag_spectral_outlier":"Spectral Outlier",
}

# Columns where the raw value (V) must be converted to µV for display
_UV_AMPLITUDE_COLS = {"mae_V", "std_V", "peak_V"}
# Columns where V² must be converted to µV²
_UV2_AMPLITUDE_COLS = {"mse_V2", "hjorth_activity"}

# Human-readable labels for pipeline configuration keys
_PARAM_LABELS: Dict[str, str] = {
    "channel_picks":       "Selected channels",
    "epoch_tmin":          "Epoch start (s, relative to event onset)",
    "epoch_tmax":          "Epoch end (s, relative to event onset)",
    "l_freq":              "High-pass filter cutoff (Hz)",
    "h_freq":              "Low-pass filter cutoff (Hz)",
    "filter_order":        "Butterworth filter order",
    "notch_freq":          "Notch filter frequency (Hz)",
    "artifact_threshold":  "Soft-clipping threshold (µV)",
    "use_avg_ref":         "Common Average Reference (CAR)",
    "use_soft_clip":       "Soft amplitude clipping enabled",
}

# Short glossary entries shown at the bottom of the report
_GLOSSARY: List[tuple] = [
    ("SNR (Signal-to-Noise Ratio, dB)",
     "Measures how much of the original signal was preserved after filtering. "
     "Higher values indicate less distortion. Values near 0 dB are expected when "
     "the raw signal contains a large DC component that the high-pass filter removes."),
    ("PaLOSi — Recording-Level Spectral Homogeneity",
     "A single value for the whole recording, in the range [0, 1]. Measures the "
     "structural homogeneity of cross-spectral matrices across frequencies. "
     "The ideal range for well-preprocessed EEG is [0.3, 0.6] (Hu et al. 2025). "
     "Values below 0.3 suggest insufficient denoising (heterogeneous spectra due to "
     "retained noise). Values above 0.6 suggest over-preprocessing (channels become "
     "spectrally too similar because brain signals were removed). "
     "Based on Hu et al. (2025) NeuroImage https://doi.org/10.1016/j.neuroimage.2025.121247"),
    ("Log-Spectra Deviation — Per-Channel Spectral Outlier Score",
     "Measures how far each individual channel's frequency spectrum deviates from "
     "the median spectrum of all channels. A high score means that channel looks "
     "spectrally different from its neighbours, which is a sign of artefact "
     "contamination or electrode malfunction. Scores below 2 are generally acceptable."),
    ("MAE (Mean Absolute Error, µV)",
     "Average absolute difference between the raw (DC-removed) and the filtered "
     "signal, in microvolts. Reflects the total amount of signal modification."),
    ("MSE (Mean Squared Error, µV²)",
     "Same as MAE but squares the differences, penalising large deviations more."),
    ("Spectral Entropy",
     "Measures how spread out the signal's energy is across frequencies (0 = all "
     "energy in one frequency, 1 = energy spread evenly across all frequencies). "
     "Very low values may indicate narrow-band artefacts such as residual line noise."),
    ("Hjorth Activity (µV²)",
     "The variance of the signal — a simple measure of overall signal power."),
    ("Hjorth Mobility",
     "Approximate mean frequency of the signal, derived from how fast the signal "
     "changes over time."),
    ("Hjorth Complexity",
     "Measures how irregular or complex the signal shape is. Pure sine waves "
     "have low complexity; broadband noise has high complexity."),
    ("Std. Deviation (µV)",
     "Standard deviation of the filtered signal. Very low values (< 0.1 µV) "
     "indicate a flat or disconnected electrode."),
    ("Peak Amplitude (µV)",
     "Maximum absolute voltage observed in the filtered signal. High values "
     "may indicate muscle artefacts or eye blinks that were not fully removed."),
    ("Imbalance Ratio",
     "Ratio between the most frequent and the least frequent class. A value of 1 "
     "means perfectly balanced; higher values indicate imbalance."),
    ("Balance Entropy",
     "Normalised entropy of the class distribution (0 = only one class, "
     "1 = all classes equally represented)."),
]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _fmt_uv(val: float) -> str:
    """Adaptive decimal formatting for µV values."""
    abs_val = abs(val)
    if abs_val >= 1.0:
        return f"{val:.2f}"
    elif abs_val >= 0.001:
        return f"{val:.4f}"
    else:
        return f"{val:.2e}"


def _combined_stats_html(
    raw_arr: np.ndarray,
    proc_arr: np.ndarray,
    ch_names: List[str],
) -> str:
    """
    Single table: each cell shows raw_value | processed_value in µV.
    All statistics are pre-computed once to avoid redundant passes over data.
    """
    raw_uv  = raw_arr  * 1e6
    proc_uv = proc_arr * 1e6

    # Pre-compute every statistic once — shape (n_channels,) each
    col_defs = [
        ("Mean (µV)",     raw_uv.mean(axis=-1),                    proc_uv.mean(axis=-1)),
        ("Std. Dev. (µV)", raw_uv.std(axis=-1),                    proc_uv.std(axis=-1)),
        ("Min (µV)",      raw_uv.min(axis=-1),                     proc_uv.min(axis=-1)),
        ("Max (µV)",      raw_uv.max(axis=-1),                     proc_uv.max(axis=-1)),
        ("Peak (µV)",     np.max(np.abs(raw_uv), axis=-1),         np.max(np.abs(proc_uv), axis=-1)),
        ("RMS (µV)",      np.sqrt(np.mean(raw_uv ** 2, axis=-1)),  np.sqrt(np.mean(proc_uv ** 2, axis=-1))),
    ]

    sep = ' <span style="color:#bbb;font-weight:400">|</span> '

    header = "<tr><th>Channel</th>" + "".join(
        f"<th>{name}</th>" for name, _, _ in col_defs
    ) + "</tr>"

    rows_html = []
    for i, ch in enumerate(ch_names):
        cells = f"<td style='text-align:left'><strong>{ch}</strong></td>"
        for _, raw_vals, proc_vals in col_defs:
            r = _fmt_uv(float(raw_vals[i]))
            p = _fmt_uv(float(proc_vals[i]))
            cells += (
                f"<td>"
                f"<span style='color:#555'>{r}</span>"
                f"{sep}"
                f"<span style='color:#1a252f;font-weight:500'>{p}</span>"
                f"</td>"
            )
        rows_html.append(f"<tr>{cells}</tr>")

    return (
        f"<table><thead>{header}</thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _channel_stats_v(data: np.ndarray, ch_names: List[str]) -> pd.DataFrame:
    """Same statistics in SI volts — used for the raw CSV exports only."""
    return pd.DataFrame(
        {
            "mean_V":  data.mean(axis=-1),
            "std_V":   data.std(axis=-1),
            "min_V":   data.min(axis=-1),
            "max_V":   data.max(axis=-1),
            "peak_V":  np.max(np.abs(data), axis=-1),
            "rms_V":   np.sqrt(np.mean(data ** 2, axis=-1)),
        },
        index=ch_names,
    )


def _class_balance(event_id: Dict[str, int], epochs) -> pd.DataFrame:
    """
    Per-class epoch counts and imbalance measures.

    Returns a DataFrame with columns:
        Epoch Count, Proportion (%), Imbalance Ratio

    and DataFrame-level attributes:
        balance_entropy        (0 = single class, 1 = perfectly balanced)
        majority_minority_ratio
    """
    counts = {
        label: int(np.sum(epochs.events[:, 2] == code))
        for label, code in event_id.items()
    }
    df = pd.DataFrame.from_dict(
        counts, orient="index", columns=["Epoch Count"]
    ).sort_values("Epoch Count", ascending=False)

    total = df["Epoch Count"].sum()
    df["Proportion (%)"] = (df["Epoch Count"] / total * 100).round(1)
    majority = df["Epoch Count"].max()
    df["Imbalance Ratio"] = (majority / df["Epoch Count"]).round(2)

    p = (df["Epoch Count"] / total).values
    n = len(p)
    entropy = float(
        -np.sum(p * np.log(p + 1e-30)) / np.log(n) if n > 1 else 1.0
    )
    df.attrs["balance_entropy"] = round(entropy, 4)
    df.attrs["majority_minority_ratio"] = round(float(majority / df["Epoch Count"].min()), 4)
    return df


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _fig_psd_comparison(
    raw_arr: np.ndarray,
    proc_arr: np.ndarray,
    sfreq: float,
    out_path: Path,
) -> None:
    """Side-by-side PSD for raw and processed signals, in µV²/Hz."""
    freqs, psd_raw  = compute_psd(raw_arr,  sfreq)
    _,     psd_proc = compute_psd(proc_arr, sfreq)

    # Convert V²/Hz -> µV²/Hz
    psd_raw_uv  = psd_raw  * 1e12
    psd_proc_uv = psd_proc * 1e12

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    labels = ["Before preprocessing", "After preprocessing"]
    for psd, ax, title in zip([psd_raw_uv, psd_proc_uv], axes, labels):
        for row in psd:
            ax.semilogy(freqs, row, alpha=0.35, linewidth=0.7, color="steelblue")
        ax.semilogy(
            freqs, np.median(psd, axis=0),
            color="black", linewidth=1.8, label="Median across channels",
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Power spectral density (µV²/Hz)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, min(sfreq / 2.0, 120.0))
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Signal Power by Frequency — each thin line is one channel, "
        "the thick black line is the median",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_snr_palosi(quality_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar charts for SNR and PaLOSi, colour-coded by status."""
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(quality_df) * 0.35)))

    color_map   = {"good": "#2ecc71", "warning": "#f39c12", "bad": "#e74c3c"}
    status_label = {"good": "Good", "warning": "Needs review", "bad": "Poor quality"}
    colors = [color_map.get(s, "grey") for s in quality_df["status"]]

    # --- SNR ---
    axes[0].barh(quality_df.index, quality_df["snr_db"], color=colors)
    axes[0].set_xlabel("SNR (dB)\nHigher = less signal distortion", fontsize=9)
    axes[0].set_title("Signal-to-Noise Ratio per Channel", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.4)

    # --- Log-Spectra Deviation ---
    axes[1].barh(quality_df.index, quality_df["log_spectra_dev"], color=colors)
    axes[1].set_xlabel("Log-Spectra Deviation\nLower = more similar to the channel ensemble", fontsize=9)
    axes[1].set_title("Log-Spectra Deviation per Channel", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_map[s], label=status_label[s]) for s in color_map]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, title="Channel status")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_quality_heatmap(quality_df: pd.DataFrame, out_path: Path) -> None:
    """
    Heatmap of normalised quality metrics with human-readable column names
    and µV-converted amplitude values.
    """
    # Build display DataFrame with converted units and readable names
    display_cols = {
        "snr_db":           ("SNR (dB)",              1),
        "log_spectra_dev":  ("Log-Spectra Dev.",       1),
        "spectral_entropy": ("Spectral Entropy",       1),
        "hjorth_mobility":  ("Hjorth Mobility",        1),
        "hjorth_complexity":("Hjorth Complexity",      1),
        "std_V":            ("Std. Dev. (µV)",         1e6),
        "peak_V":           ("Peak Amplitude (µV)",    1e6),
    }
    frames = {}
    for col, (label, scale) in display_cols.items():
        if col in quality_df.columns:
            frames[label] = quality_df[col] * scale

    subset = pd.DataFrame(frames)
    subset_norm = (subset - subset.min()) / (subset.max() - subset.min() + 1e-30)

    fig, ax = plt.subplots(
        figsize=(max(9, len(frames) * 1.3), max(4, len(subset) * 0.38))
    )
    sns.heatmap(
        subset_norm,
        ax=ax,
        cmap="RdYlGn",
        annot=subset.round(3),
        fmt="",
        linewidths=0.3,
        cbar_kws={"label": "Normalised rank within column (0 = lowest, 1 = highest)"},
    )
    ax.set_title(
        "Channel Quality Overview\n"
        "Colour shows relative rank; annotation shows the actual value",
        fontsize=10,
    )
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fig_class_balance(balance_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of epoch counts per class with imbalance annotation."""
    fig, ax = plt.subplots(figsize=(max(6, len(balance_df) * 1.0), 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(balance_df)))
    balance_df["Epoch Count"].plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Condition / Class label", fontsize=10)
    ax.set_ylabel("Number of epochs", fontsize=10)

    mmr = balance_df.attrs.get("majority_minority_ratio", "?")
    ent = balance_df.attrs.get("balance_entropy", "?")
    ax.set_title(
        f"Number of Epochs per Condition\n"
        f"Most-common / least-common class ratio: {mmr}   "
        f"|   Balance score: {ent} (1.0 = perfectly balanced)",
        fontsize=9,
    )
    ax.tick_params(axis="x", rotation=45)
    for bar, cnt in zip(ax.patches, balance_df["Epoch Count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(cnt),
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _params_table_html(params: Dict[str, Any]) -> str:
    """Render pipeline parameters as a readable two-column HTML table."""
    rows = []
    for key, label in _PARAM_LABELS.items():
        if key not in params:
            continue
        val = params[key]
        if key == "artifact_threshold" and isinstance(val, (int, float)):
            val = f"{val * 1e6:.1f} µV"
        elif val is None:
            val = "All EEG channels"
        elif isinstance(val, bool):
            val = "Yes" if val else "No"
        rows.append(f"<tr><td style='text-align:left'><strong>{label}</strong></td><td>{val}</td></tr>")

    return f"<table>{''.join(rows)}</table>"


def _quality_table_html(quality_df: pd.DataFrame) -> str:
    """
    Render the channel quality DataFrame with human-readable column headers,
    µV-converted amplitudes, colour-coded status cells, and Yes/No flags.
    """
    STATUS_BADGE = {
        "good":    '<span style="background:#2ecc71;color:#fff;padding:2px 8px;border-radius:4px">Good</span>',
        "warning": '<span style="background:#f39c12;color:#fff;padding:2px 8px;border-radius:4px">Needs review</span>',
        "bad":     '<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px">Poor quality</span>',
    }

    # Columns to show in the HTML table (subset — full set is in CSV)
    display_order = [
        "status", "snr_db", "log_spectra_dev", "spectral_entropy",
        "std_V", "peak_V", "hjorth_mobility", "hjorth_complexity",
        "flag_flat", "flag_high_amplitude", "flag_low_snr", "flag_spectral_outlier",
    ]
    cols = [c for c in display_order if c in quality_df.columns]

    header_cells = "<th>Channel</th>" + "".join(
        f"<th>{_QUALITY_LABELS.get(c, c)}</th>" for c in cols
    )
    header = f"<tr>{header_cells}</tr>"

    rows_html = []
    for ch_name, row in quality_df.iterrows():
        status = row.get("status", "good")
        cells = f"<td><strong>{ch_name}</strong></td>"
        for col in cols:
            val = row[col]
            if col == "status":
                cells += f"<td>{STATUS_BADGE.get(val, val)}</td>"
            elif col in _UV_AMPLITUDE_COLS:
                cells += f"<td>{val * 1e6:.3f}</td>"
            elif col in _UV2_AMPLITUDE_COLS:
                cells += f"<td>{val * 1e12:.4f}</td>"
            elif col.startswith("flag_"):
                cells += f"<td>{'Yes' if val else '-'}</td>"
            elif isinstance(val, float):
                cells += f"<td>{val:.4f}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html.append(f"<tr>{cells}</tr>")

    return f"<table><thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


def _palosi_status(palosi_value: float) -> tuple[str, str, str]:
    """
    Return (css_color, card_style, banner_html) for a PaLOSi value.

    Ideal range [0.3, 0.6] per Hu et al. (2025).
    """
    if 0.3 <= palosi_value <= 0.6:
        color = "#2ecc71"
        banner = (
            f'<div class="note"><strong>PaLOSi = {palosi_value:.3f}</strong> '
            f'is within the ideal range [0.3, 0.6] for well-preprocessed EEG. '
            f'The recording shows healthy spectral diversity across channels.</div>'
        )
    elif palosi_value < 0.3:
        color = "#f39c12"
        banner = (
            f'<div class="warn-note"><strong>PaLOSi = {palosi_value:.3f}</strong> '
            f'is <strong>below 0.3</strong>. This suggests insufficient denoising: '
            f'channel spectra are heterogeneous, likely because broadband noise is '
            f'still present. Consider adjusting the bandpass cutoffs or notch filter.</div>'
        )
    else:
        color = "#e74c3c"
        banner = (
            f'<div class="warn-note"><strong>PaLOSi = {palosi_value:.3f}</strong> '
            f'is <strong>above 0.6</strong>. This suggests over-preprocessing: '
            f'channels have become too spectrally similar, which may indicate that '
            f'brain signals were removed along with artefacts. '
            f'Consider relaxing the filter settings or artifact threshold.</div>'
        )
    return color, f"color:{color}", banner


def _glossary_html() -> str:
    rows = "".join(
        f"<tr><td style='text-align:left;white-space:nowrap'><strong>{term}</strong></td>"
        f"<td style='text-align:left'>{desc}</td></tr>"
        for term, desc in _GLOSSARY
    )
    return f"<table>{rows}</table>"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>EVA Preprocessing Report &mdash; {stem}</title>
<style>
  body    {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px 60px;
             background: #f4f6f9; color: #212529; line-height: 1.6; }}
  h1      {{ color: #1a252f; margin-bottom: 4px; }}
  h2      {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7;
             padding-bottom: 5px; margin-top: 36px; }}
  h3      {{ color: #34495e; margin-top: 20px; }}
  .meta   {{ color: #5d6d7e; font-size: 14px; margin-bottom: 6px; }}
  .note   {{ background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #1a5276; }}
  .warn-note {{ background: #fef9e7; border-left: 4px solid #f39c12;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #7d6608; }}
  table   {{ border-collapse: collapse; width: 100%; margin-bottom: 18px;
             font-size: 13px; background: #fff; }}
  th, td  {{ border: 1px solid #dee2e6; padding: 6px 10px; }}
  th      {{ background: #2c3e50; color: #ecf0f1; text-align: left; font-weight: 600; }}
  td      {{ text-align: right; }}
  tr:nth-child(even) {{ background: #f2f3f4; }}
  img     {{ max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px;
             margin: 10px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
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

<h1>EEG Preprocessing Report</h1>
<p class="meta">Generated by EVA &mdash; EEG data Validation and preprocessing Assistant</p>

<h2>Recording Overview</h2>
<div class="summary-grid">
  <div class="stat-card"><div class="label">Recording</div><div class="value" style="font-size:14px">{stem}</div></div>
  <div class="stat-card"><div class="label">Sampling Rate</div><div class="value">{sfreq} Hz</div></div>
  <div class="stat-card"><div class="label">EEG Channels</div><div class="value">{n_ch}</div></div>
  <div class="stat-card"><div class="label">Epochs Extracted</div><div class="value">{n_epochs}</div></div>
  <div class="stat-card"><div class="label">Distinct Conditions</div><div class="value">{n_classes}</div></div>
  <div class="stat-card"><div class="label">Epoch Window</div><div class="value" style="font-size:16px">{tmin} &ndash; {tmax} s</div></div>
  <div class="stat-card"><div class="label">PaLOSi (Recording)</div><div class="value" style="font-size:18px;{palosi_card_style}">{palosi_recording:.3f}</div></div>
</div>
{palosi_banner}

<h2>Applied Preprocessing Steps</h2>
<div class="note">
  The table below shows the exact parameters used to process this recording.
  Each step is applied in the order listed: DC removal, bandpass filter,
  notch filter, optional average re-referencing, optional amplitude clipping.
</div>
{params_table}

<h2>Signal Statistics &mdash; Before and After Preprocessing</h2>
<div class="note">
  Each cell shows the value <strong>before preprocessing</strong>
  <span style="color:#bbb">|</span>
  <strong>after preprocessing</strong>, both in microvolts (&micro;V).
  Values with magnitude &ge; 1 &micro;V are shown with 2 decimal places;
  smaller values use 4 decimal places.
</div>
{combined_stats_html}

<h2>Frequency Content &mdash; Before and After</h2>
<div class="note">
  Each thin line represents one electrode.  The thick black line is the median
  across all electrodes.  The vertical axis is on a logarithmic scale.
  A well-preprocessed signal typically shows a smooth 1/f slope (more power
  at low frequencies, less at high frequencies) with no sharp peaks except
  possibly at very low frequencies.
</div>
<img src="psd_comparison.png" alt="Power spectral density before and after preprocessing"/>

<h2>Channel Quality Assessment</h2>
<div class="note">
  Every electrode is evaluated against four quality criteria.  If none are
  triggered the channel is classified as <strong>Good</strong>; one trigger
  gives <strong>Needs review</strong>; two or more give
  <strong>Poor quality</strong>.  The bar charts and table below show the
  two most informative per-channel metrics.  The recording-level
  <strong>PaLOSi</strong> score (shown in the overview cards above) measures
  the overall spectral homogeneity of the recording as a whole.
  See the Glossary at the bottom for definitions.
</div>
<img src="snr_palosi.png" alt="SNR and Log-Spectra Deviation per channel"/>

<h3>Detailed Metrics per Channel</h3>
<div class="warn-note">
  Amplitude values (Std. Deviation, Peak Amplitude) are in microvolts (&micro;V).
  Flag columns show &ldquo;Yes&rdquo; only when the threshold is exceeded; a dash means no issue detected.
</div>
{quality_html}
<img src="quality_heatmap.png" alt="Channel quality heatmap"/>

<h2>Condition Distribution</h2>
<div class="note">
  The table and chart below show how many epochs were collected for each
  experimental condition.  A balanced dataset (equal counts per condition)
  is generally preferable for machine-learning classifiers.
  The <em>Imbalance Ratio</em> compares the most-frequent to each condition:
  a value of 1 means perfect balance.
</div>
{balance_html}
<img src="class_balance.png" alt="Number of epochs per condition"/>

<h2>Metrics Glossary</h2>
<div class="note">
  Quick reference for every metric used in this report.
</div>
{glossary_html}

<footer>EVA &mdash; EEG data Validation and preprocessing Assistant</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generate an HTML exploratory analysis report for each processed recording.

    Parameters
    ----------
    report_dir : Path
        Root directory for all report outputs.
    params : dict
        Pipeline parameters used during preprocessing.
    """

    def __init__(self, report_dir: Path, params: Dict[str, Any]) -> None:
        self.report_dir = Path(report_dir)
        self.params = params

    def generate(self, artefacts: List[Dict]) -> None:
        """Generate one report per recording artefact."""
        for art in artefacts:
            try:
                self._generate_one(art)
            except Exception as exc:
                logger.error(
                    "Report generation failed for '%s': %s",
                    art.get("stem", "?"),
                    exc,
                )

    def _generate_one(self, art: Dict) -> None:
        stem             = art["stem"]
        sfreq            = art["sfreq"]
        ch_names         = art["ch_names"]
        raw_arr          = art["raw_arr"]
        proc_arr         = art["processed_arr"]
        quality_df       = art["quality_df"]
        event_id         = art["event_id"]
        epochs_proc      = art["epochs_proc"]
        palosi_recording = art.get("palosi_recording", float("nan"))

        rec_dir = self.report_dir / stem
        rec_dir.mkdir(parents=True, exist_ok=True)

        # --- Figures ---
        _fig_psd_comparison(raw_arr, proc_arr, sfreq, rec_dir / "psd_comparison.png")
        _fig_snr_palosi(quality_df, rec_dir / "snr_palosi.png")
        _fig_quality_heatmap(quality_df, rec_dir / "quality_heatmap.png")

        balance_df = _class_balance(event_id, epochs_proc)
        _fig_class_balance(balance_df, rec_dir / "class_balance.png")

        # --- CSV exports (SI volts, for downstream use) ---
        _channel_stats_v(raw_arr,  ch_names).to_csv(rec_dir / "raw_stats.csv",       encoding="utf-8")
        _channel_stats_v(proc_arr, ch_names).to_csv(rec_dir / "proc_stats.csv",      encoding="utf-8")
        quality_df.to_csv(rec_dir / "channel_quality.csv",                            encoding="utf-8")
        balance_df.to_csv(rec_dir / "class_balance.csv",                              encoding="utf-8")

        tmin = art.get("tmin", "?")
        tmax = art.get("tmax", "?")

        palosi_color, palosi_card_style, palosi_banner = _palosi_status(palosi_recording)

        html = _HTML_TEMPLATE.format(
            stem=stem,
            sfreq=sfreq,
            n_ch=len(ch_names),
            n_epochs=art["n_epochs"],
            n_classes=len(event_id),
            tmin=tmin,
            tmax=tmax,
            palosi_recording=palosi_recording,
            palosi_card_style=palosi_card_style,
            palosi_banner=palosi_banner,
            params_table=_params_table_html(self.params),
            combined_stats_html=_combined_stats_html(raw_arr, proc_arr, ch_names),
            quality_html=_quality_table_html(quality_df),
            balance_html=balance_df.to_html(
                classes="table", float_format="{:.2f}".format
            ),
            glossary_html=_glossary_html(),
        )

        (rec_dir / "report.html").write_text(html, encoding="utf-8")
        logger.info("Report saved -> %s/report.html", rec_dir)
