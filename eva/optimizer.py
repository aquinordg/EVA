"""Grid-search optimisation for the EEG preprocessing pipeline.

Evaluates every combination of filter parameters, scores each with a
composite SNR + PaLOSi metric, and returns the best-scoring parameter set.

Scoring
-------
    score = α × mean_SNR_dB  −  (1 − α) × |PaLOSi − 0.45|

Higher is better.  ``α`` controls the trade-off between reconstruction
fidelity (SNR) and spectral quality (PaLOSi).

The PaLOSi term penalises deviation from the centre of the ideal range
[0.3, 0.6] identified by Hu et al. (2025).  Values below 0.3 indicate
insufficient denoising; values above 0.6 indicate over-preprocessing.
Minimising |PaLOSi − 0.45| drives the pipeline towards the clean-EEG
sweet spot rather than blindly pushing PaLOSi toward zero.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .filters import AverageReference, ButterworthFilter, DCDetrend, NotchFilter, SoftClipper
from .metrics import palosi, snr_db

logger = logging.getLogger(__name__)

# Centre of the PaLOSi ideal range [0.3, 0.6] for well-preprocessed EEG
# (Hu et al. 2025, NeuroImage).  The optimizer penalises deviation from this.
_PALOSI_TARGET: float = 0.45

# Default candidate values — each key maps to a list of values to try.
# None means "skip this filter entirely".
_DEFAULT_GRID: Dict[str, List] = {
    "l_freq":              [None, 0.5, 1.0, 2.0],   # None = no high-pass
    "h_freq":              [30.0, 40.0, 50.0],
    "filter_order":        [4, 6],
    "notch_freq":          [None, 50.0, 60.0],       # None = no notch
    "artifact_threshold":  [75e-6, 100e-6, 150e-6],
    "use_avg_ref":         [True, False],
    "use_soft_clip":       [True, False],
}


def _build_chain(params: Dict[str, Any]) -> list:
    steps = [DCDetrend()]
    steps.append(ButterworthFilter(
        l_freq=params["l_freq"],
        h_freq=params["h_freq"],
        order=params["filter_order"],
    ))
    if params["notch_freq"] is not None:
        steps.append(NotchFilter(freq=params["notch_freq"]))
    if params["use_avg_ref"]:
        steps.append(AverageReference())
    if params["use_soft_clip"]:
        steps.append(SoftClipper(threshold=params["artifact_threshold"]))
    return steps


def _composite_score(
    raw_detrended: np.ndarray,
    processed: np.ndarray,
    sfreq: float,
    alpha: float,
) -> float:
    mean_snr = float(np.mean(snr_db(raw_detrended, processed)))
    pal = float(palosi(processed, sfreq))
    palosi_penalty = abs(pal - _PALOSI_TARGET)
    return alpha * mean_snr - (1.0 - alpha) * palosi_penalty


def find_best_params(
    data: np.ndarray,
    sfreq: float,
    grid: Optional[Dict[str, List]] = None,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Find the best preprocessing parameters via grid search.

    Tries every combination of filter settings, scores each with
    SNR + PaLOSi, and returns the winning parameter set.

    Parameters
    ----------
    data  : (n_channels, n_samples) raw EEG in volts
    sfreq : sampling frequency (Hz)
    grid  : custom grid of candidate values; uses built-in defaults when None
    alpha : SNR weight in [0, 1]; 1 = SNR only, 0 = PaLOSi only

    Returns
    -------
    dict with the best parameter combination
    """
    g = grid or _DEFAULT_GRID
    keys = list(g.keys())
    combos = list(itertools.product(*g.values()))
    logger.info("Grid search: %d configurations to evaluate", len(combos))

    raw_detrended = DCDetrend().apply(data, sfreq)
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for combo in tqdm(combos, desc="Optimising pipeline", unit="cfg"):
        params = dict(zip(keys, combo))
        try:
            chain = _build_chain(params)
            processed = data.copy()
            for step in chain:
                processed = step.apply(processed, sfreq)
            score = _composite_score(raw_detrended, processed, sfreq, alpha)
        except Exception as exc:
            logger.debug("Config %s raised %s — skipped.", params, exc)
            continue

        if score > best_score:
            best_score = score
            best_params = params.copy()

    if not best_params:
        raise RuntimeError(
            "All grid configurations failed. "
            "Check that sfreq is correct and data is non-empty."
        )

    logger.info("Best config  score=%.4f  params=%s", best_score, best_params)
    return best_params
