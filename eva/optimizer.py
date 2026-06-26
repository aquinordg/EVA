"""Grid-search optimisation for the EEG preprocessing pipeline.

Evaluates every combination of filter parameters, scores each with PaLOSi,
and returns the best-scoring parameter set.

Scoring
-------
    score = -(|PaLOSi − 0.45|)

Higher is better (less negative).  The score penalises any deviation from the
centre of the ideal PaLOSi range [0.3, 0.6] identified by Hu et al. (2025).
Values below 0.3 indicate insufficient denoising; values above 0.6 indicate
over-preprocessing.  Targeting 0.45 (the midpoint) drives the pipeline toward
the clean-EEG sweet spot.

SNR is not used in scoring: for clean EEG, SNR ≈ 0 dB is the expected and
correct outcome (most signal energy already lies within the passband), making
it uninformative as an optimisation criterion.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .filters import AverageReference, ButterworthFilter, DCDetrend, NotchFilter, SoftClipper
from .metrics import palosi

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


def _composite_score(processed: np.ndarray, sfreq: float) -> float:
    pal = float(palosi(processed, sfreq))
    return -(abs(pal - _PALOSI_TARGET))


def find_best_params(
    data: np.ndarray,
    sfreq: float,
    grid: Optional[Dict[str, List]] = None,
) -> Dict[str, Any]:
    """
    Find the best preprocessing parameters via grid search.

    Tries every combination of filter settings, scores each by how close
    the resulting PaLOSi is to 0.45 (midpoint of the ideal [0.3, 0.6] range),
    and returns the winning parameter set.

    Parameters
    ----------
    data  : (n_channels, n_samples) raw EEG in volts
    sfreq : sampling frequency (Hz)
    grid  : custom grid of candidate values; uses built-in defaults when None

    Returns
    -------
    dict with the best parameter combination
    """
    g = grid or _DEFAULT_GRID
    keys = list(g.keys())
    combos = list(itertools.product(*g.values()))
    logger.info("Grid search: %d configurations to evaluate", len(combos))

    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for combo in tqdm(combos, desc="Optimising pipeline", unit="cfg"):
        params = dict(zip(keys, combo))
        try:
            chain = _build_chain(params)
            processed = data.copy()
            for step in chain:
                processed = step.apply(processed, sfreq)
            score = _composite_score(processed, sfreq)
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
