"""
Compare old vs new channel diagnostics on pilot VECA-EEG data.

OLD: flag_low_snr  (snr_db < 10 → any channel barely filtered = "bad")
NEW: flag_adc_clipping (fraction of samples stuck at exact min/max > 0.1%)
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\dougl\Documents\GitHub\EVA")

import numpy as np
import pandas as pd
import mne

from eva import align_veca, preprocess
from eva.metrics import (
    snr_db, adc_clipping_fraction, log_spectra_deviation,
    evaluate_all_channels, QualityConfig, palosi
)

DATA_DIR = r"C:\Users\dougl\Meu Drive\CODES\pd\VECA-EEG\data"

SESSIONS = [
    ("8GDPRM", "VECA_8GDPRM_20260626_122200.vhdr", "VECA_8GDPRM_20260626_122200.csv"),
    ("9XN7A2", "VECA_9XN7A2_20260626_124949.vhdr", "VECA_9XN7A2_20260626_124949.csv"),
    ("7T632W", "VECA_7T632W_20260626_124228.vhdr", "VECA_7T632W_20260626_124228.csv"),
]


def old_evaluate(ch_name, raw_ch, processed_ch, sfreq, log_spectra_dev=0.0):
    """Simulate OLD QualityConfig with flag_low_snr instead of flag_adc_clipping."""
    snr_val  = float(snr_db(raw_ch[None], processed_ch[None])[0])
    std_val  = float(np.std(processed_ch))
    peak_val = float(np.max(np.abs(processed_ch)))

    flag_flat             = std_val < 100e-9
    flag_high_amplitude   = peak_val > 150e-6
    flag_spectral_outlier = log_spectra_dev > 2.0
    flag_low_snr          = snr_val < 10.0

    n_flags = sum([flag_flat, flag_high_amplitude, flag_spectral_outlier, flag_low_snr])
    status = "good" if n_flags == 0 else ("warning" if n_flags == 1 else "bad")
    return {"status": status, "snr_db": round(snr_val, 1), "flag_low_snr": flag_low_snr,
            "flag_flat": flag_flat, "flag_high_amplitude": flag_high_amplitude,
            "flag_spectral_outlier": flag_spectral_outlier}


print("=" * 70)
print("CHANNEL DIAGNOSTIC COMPARISON — OLD vs NEW")
print("OLD flag: flag_low_snr  (snr < 10 dB)")
print("NEW flag: flag_adc_clipping  (stuck samples > 0.1%)")
print("=" * 70)

for pid, vhdr_file, csv_file in SESSIONS:
    vhdr_path = os.path.join(DATA_DIR, vhdr_file)
    csv_path  = os.path.join(DATA_DIR, csv_file)

    print(f"\n{'-'*70}")
    print(f"Participant: {pid}")
    print(f"{'-'*70}")

    # Load and preprocess
    raw_mne, trials = align_veca(vhdr_path, csv_path)
    raw_mne.load_data()
    ch_names = raw_mne.ch_names

    # Raw data (before any processing) for ADC clipping
    raw_data = raw_mne.get_data()

    # DC-detrend (remove mean) — same as what preprocess() does internally before filtering
    raw_detrended = raw_data - raw_data.mean(axis=-1, keepdims=True)

    # Preprocess with default params
    raw_proc = raw_mne.copy()
    raw_proc.filter(l_freq=1.0, h_freq=40.0, method="fir", fir_design="firwin",
                    filter_length="auto", l_trans_bandwidth="auto",
                    h_trans_bandwidth="auto", verbose=False)
    raw_proc.notch_filter(freqs=60.0, verbose=False)
    raw_proc.set_eeg_reference("average", verbose=False)

    proc_data = raw_proc.get_data()
    sfreq = raw_mne.info["sfreq"]

    # Compute shared metrics
    lsd_scores  = log_spectra_deviation(proc_data, sfreq)
    clip_fracs  = adc_clipping_fraction(raw_detrended)
    snr_scores  = snr_db(raw_detrended, proc_data)

    # PaLOSi
    pal = palosi(proc_data, sfreq)
    print(f"PaLOSi (default preprocessing): {pal:.3f}  "
          f"({'OK' if 0.3 <= pal <= 0.6 else 'ABOVE IDEAL RANGE'})")

    # Build comparison table
    rows = []
    for i, ch in enumerate(ch_names):
        old = old_evaluate(ch, raw_detrended[i], proc_data[i], sfreq,
                           log_spectra_dev=float(lsd_scores[i]))
        new_cfg = QualityConfig()
        new = new_cfg.evaluate(ch, raw_detrended[i], proc_data[i], sfreq,
                               log_spectra_dev=float(lsd_scores[i]),
                               adc_clip_frac=float(clip_fracs[i]))
        new_flags = []
        if new["flag_flat"]:            new_flags.append("flat")
        if new["flag_high_amplitude"]:  new_flags.append("high_amp")
        if new["flag_spectral_outlier"]:new_flags.append("lsd_out")
        if new["flag_adc_clipping"]:    new_flags.append("adc_clip")
        new_flag_str = ",".join(new_flags) if new_flags else "none"

        changed = " *" if old["status"] != new["status"] else ""
        rows.append({
            "ch":          ch,
            "snr_dB":      f"{old['snr_db']:+.1f}",
            "clip_%":      f"{clip_fracs[i]*100:.4f}",
            "lsd":         f"{lsd_scores[i]:.2f}",
            "peak_uV":     f"{np.max(np.abs(proc_data[i]))*1e6:.1f}",
            "old_status":  old["status"],
            "new_status":  new["status"],
            "new_flags":   new_flag_str,
            "change":      changed,
        })

    df = pd.DataFrame(rows)
    header = (f"{'Ch':<6} {'SNR(dB)':>8} {'Clip%':>8} {'LSD':>5} {'Peak(uV)':>9} "
              f"{'OLD':>8}  {'NEW':>8}  {'new flags':<20}")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        chg = r['change']
        print(f"{r['ch']:<6} {r['snr_dB']:>8} {r['clip_%']:>8} {r['lsd']:>5} {r['peak_uV']:>9} "
              f"{r['old_status']:>8}  {r['new_status']:>8}  {r['new_flags']:<20}{chg}")

    old_counts = df["old_status"].value_counts().to_dict()
    new_counts = df["new_status"].value_counts().to_dict()
    print(f"\n  OLD summary: {old_counts}")
    print(f"  NEW summary: {new_counts}")

print("\n" + "=" * 70)
print("DONE")
