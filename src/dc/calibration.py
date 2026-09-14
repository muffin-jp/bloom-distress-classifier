"""Are the scores probabilities?

This matters here for a concrete reason, not a tidy one. The cascade routes on
probability bands — skip the LLM below one threshold, go straight to support
above another — and a model's scores only mean "probability" if they are
calibrated. A model can rank perfectly and still say 0.9 about cases that are
distress half the time. Class weighting in particular moves scores away from
probabilities: ``class_weight="balanced"`` inflates the positive class by design.

Two numbers and a table:

* **Brier score** — mean squared error of the probability. Lower is better; it
  rewards both ranking and calibration.
* **Expected calibration error** — the weighted gap between what the model says
  and what happens, bin by bin.
* **Reliability table** — the bins themselves, which is what a reliability
  diagram draws. Reported as numbers so the committed report carries the
  evidence rather than a picture of it.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dc.metrics import FloatArrayLike, IntArrayLike

__all__ = ["Bin", "brier_score", "expected_calibration_error", "reliability_bins"]


@dataclass(frozen=True)
class Bin:
    """One reliability bin: what the model said, and what actually happened."""

    lower: float
    upper: float
    n: int
    mean_predicted: float | None
    observed_rate: float | None

    @property
    def gap(self) -> float | None:
        if self.mean_predicted is None or self.observed_rate is None:
            return None
        return self.mean_predicted - self.observed_rate


def _validate(y_true: IntArrayLike, y_prob: FloatArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError(f"y_true {truth.shape} and y_prob {prob.shape} differ in length")
    if truth.size == 0:
        raise ValueError("nothing to calibrate")
    if not np.all(np.isfinite(prob)) or np.any(prob < 0.0) or np.any(prob > 1.0):
        raise ValueError("probabilities must be finite and within [0, 1]")
    return truth, prob


def brier_score(y_true: IntArrayLike, y_prob: FloatArrayLike) -> float:
    truth, prob = _validate(y_true, y_prob)
    return float(np.mean((prob - truth) ** 2))


def reliability_bins(y_true: IntArrayLike, y_prob: FloatArrayLike, n_bins: int = 10) -> list[Bin]:
    """Equal-width bins over [0, 1]; a score of exactly 1.0 lands in the last bin."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    truth, prob = _validate(y_true, y_prob)
    index = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    bins: list[Bin] = []
    for b in range(n_bins):
        mask = index == b
        n = int(np.sum(mask))
        bins.append(
            Bin(
                lower=b / n_bins,
                upper=(b + 1) / n_bins,
                n=n,
                mean_predicted=float(np.mean(prob[mask])) if n else None,
                observed_rate=float(np.mean(truth[mask])) if n else None,
            )
        )
    return bins


def expected_calibration_error(
    y_true: IntArrayLike, y_prob: FloatArrayLike, n_bins: int = 10
) -> float:
    """Sum over bins of (share of rows) x |mean predicted - observed rate|."""
    truth, _ = _validate(y_true, y_prob)
    total = truth.size
    return float(
        sum(
            (b.n / total) * abs(b.gap)
            for b in reliability_bins(y_true, y_prob, n_bins)
            if b.gap is not None
        )
    )
