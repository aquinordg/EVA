"""
Validation of EVA against three public EEG datasets.

Test 1 — MNE SSVEP (Nakanishi et al.)
    Peak power at stimulus frequencies (12 Hz, 15 Hz) preserved >= 75%
    after EVA default filtering.

Test 2 — PhysioNet EEGMMI
    PaLOSi in [0.3, 0.6] for >= 50% of recordings (5 subjects x 2 runs).

Test 3 — MOABB BCI Competition IV 2a
    PaLOSi in [0.3, 0.6] for >= 50% of subjects (5 subjects).
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\dougl\Documents\GitHub\EVA")

import numpy as np
import mne
mne.set_log_level("ERROR")

from eva.filters import DCDetrend, ButterworthFilter, NotchFilter, AverageReference, SoftClipper
from eva.metrics import palosi
from scipy.signal import welch

PASS = "PASS"
FAIL = "FAIL"


def apply_eva_default(data, sfreq, notch=60.0):
    chain = [
        DCDetrend(),
        ButterworthFilter(l_freq=1.0, h_freq=40.0, order=4),
        NotchFilter(freq=notch),
        AverageReference(),
        SoftClipper(threshold=100e-6),
    ]
    raw_dc = DCDetrend().apply(data.copy(), sfreq)
    proc = data.copy()
    for step in chain:
        proc = step.apply(proc, sfreq)
    return raw_dc, proc


def peak_power_ratio(raw_dc, proc, sfreq, freq_hz, bw=0.5):
    ratios = []
    for i in range(raw_dc.shape[0]):
        f, psd_r = welch(raw_dc[i], fs=sfreq, nperseg=min(2048, raw_dc.shape[1]))
        _, psd_p = welch(proc[i],   fs=sfreq, nperseg=min(2048, proc.shape[1]))
        mask = (f >= freq_hz - bw) & (f <= freq_hz + bw)
        r = float(psd_p[mask].max() / (psd_r[mask].max() + 1e-30))
        ratios.append(r)
    return float(np.mean(ratios))


# ===========================================================================
# Test 1 — MNE SSVEP (Nakanishi et al.)
# ===========================================================================

print("=" * 65)
print("TEST 1 — MNE SSVEP (Nakanishi et al.)")
print("=" * 65)

try:
    ssvep_root = mne.datasets.ssvep.data_path()
    subjects = ["sub-01", "sub-02"]
    ratios_all = []

    print(f"  {'Subj':<8} {'@12Hz':>8} {'@15Hz':>8} {'Mean':>8}")
    print(f"  {'-'*36}")

    for sub in subjects:
        vhdr = (ssvep_root / sub / "ses-01" / "eeg"
                / f"{sub}_ses-01_task-ssvep_eeg.vhdr")
        raw_s = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        raw_s.pick("eeg")
        sfreq_s = raw_s.info["sfreq"]
        data_s = raw_s.get_data()

        raw_dc, proc = apply_eva_default(data_s, sfreq_s, notch=60.0)
        r12 = peak_power_ratio(raw_dc, proc, sfreq_s, 12.0)
        r15 = peak_power_ratio(raw_dc, proc, sfreq_s, 15.0)
        mean_r = (r12 + r15) / 2
        ratios_all.append(mean_r)
        print(f"  {sub:<8} {r12:>8.3f} {r15:>8.3f} {mean_r:>8.3f}")

    overall = float(np.mean(ratios_all))
    status = PASS if overall >= 0.75 else FAIL
    print(f"\n  Mean ratio (both subjects): {overall:.3f}  (threshold >= 0.75)")
    print(f"  Result: {status}")

except Exception as e:
    print(f"  ERROR: {e}")


# ===========================================================================
# Test 2 — PhysioNet EEGMMI (resting state)
# ===========================================================================

print()
print("=" * 65)
print("TEST 2 — PhysioNet EEGMMI (resting state)")
print("=" * 65)

try:
    from mne.datasets import eegbci

    SUBJECTS = [1, 2, 3, 4, 5]
    RUNS_REST = [1, 2]   # run 1 = eyes open, run 2 = eyes closed

    palosi_scores = []
    rows = []

    print(f"  {'Subj':>5} {'Run':>4} {'PaLOSi':>8} {'In [0.3,0.6]':>14}")
    print(f"  {'-'*35}")

    for subj in SUBJECTS:
        for run in RUNS_REST:
            fnames = eegbci.load_data(subj, runs=[run],
                                      update_path=True, verbose=False)
            raw_e = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
            eegbci.standardize(raw_e)
            raw_e.pick("eeg")
            sfreq_e = raw_e.info["sfreq"]
            data_e = raw_e.get_data()

            _, proc_e = apply_eva_default(data_e, sfreq_e, notch=60.0)
            pal = palosi(proc_e, sfreq_e)
            in_range = 0.3 <= pal <= 0.6
            palosi_scores.append(pal)
            rows.append((subj, run, pal, in_range))
            mark = "OK" if in_range else "--"
            print(f"  {subj:>5} {run:>4} {pal:>8.3f} {mark:>14}")

    in_range_n = sum(r[3] for r in rows)
    pct = in_range_n / len(rows) * 100
    status = PASS if pct >= 50 else FAIL
    print(f"\n  In range: {in_range_n}/{len(rows)} ({pct:.0f}%)  (threshold >= 50%)")
    print(f"  Mean PaLOSi: {np.mean(palosi_scores):.3f}")
    print(f"  Result: {status}")

except Exception as e:
    print(f"  ERROR: {e}")


# ===========================================================================
# Test 3 — MOABB BCI Competition IV 2a
# ===========================================================================

print()
print("=" * 65)
print("TEST 3 — MOABB BCI Competition IV 2a (motor imagery)")
print("=" * 65)

try:
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001()
    SUBJECTS = [1, 2, 3, 4, 5]

    palosi_scores_m = []
    rows_m = []

    print(f"  {'Subj':>5} {'PaLOSi':>8} {'In [0.3,0.6]':>14}")
    print(f"  {'-'*30}")

    for subj in SUBJECTS:
        raw_dict = dataset.get_data(subjects=[subj])
        # Collect all training runs for this subject
        all_data = []
        sfreq_m = None
        for sess_name, runs in raw_dict[subj].items():
            if "train" not in sess_name:
                continue
            for run_name, raw_m in runs.items():
                raw_m.pick("eeg")
                sfreq_m = raw_m.info["sfreq"]
                all_data.append(raw_m.get_data())

        if not all_data or sfreq_m is None:
            continue

        # Concatenate all runs and apply EVA chain
        data_m = np.concatenate(all_data, axis=-1)
        _, proc_m = apply_eva_default(data_m, sfreq_m, notch=50.0)  # Europe: 50 Hz

        pal = palosi(proc_m, sfreq_m)
        in_range = 0.3 <= pal <= 0.6
        palosi_scores_m.append(pal)
        rows_m.append((subj, pal, in_range))
        mark = "OK" if in_range else "--"
        print(f"  {subj:>5} {pal:>8.3f} {mark:>14}")

    in_range_n = sum(r[2] for r in rows_m)
    pct = in_range_n / len(rows_m) * 100
    status = PASS if pct >= 50 else FAIL
    print(f"\n  In range: {in_range_n}/{len(rows_m)} ({pct:.0f}%)  (threshold >= 50%)")
    print(f"  Mean PaLOSi: {np.mean(palosi_scores_m):.3f}")
    print(f"  Result: {status}")

except ImportError:
    print("  moabb not installed.")
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)
