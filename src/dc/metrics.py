"""Scoring, and the confidence intervals that keep it honest.

Two choices here are deliberate and worth defending.

**PR-AUC, not ROC-AUC.** The positive class is rare and expensive. ROC-AUC is
flattered by a large negative class — a model can look strong on it while
missing most of what matters. Average precision tracks the thing being bought.

**Bootstrap intervals on everything.** The evaluation sets here are small enough
that a single relabelled row moves a headline number by a point or more. A bare
point estimate on 191 rows invites a comparison the data cannot support, so
every reported metric carries an interval and comparisons are made between
intervals, not between means.
"""

# NumPy's overloaded ufunc/percentile stubs and scikit-learn's partial typing
# resolve to partially-unknown types, so member and argument types leak as
# Unknown at this numeric boundary. Our own logic stays typed; this narrowly
# relaxes the three "unknown" rules for this module only — the same treatment
# bloom-langgraph gives app/rag/retriever.py.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score

from dc.schema import Row

#: Accepted everywhere a label or score vector is taken: callers pass plain
#: lists, and the bootstrap passes numpy arrays back into the same functions.
IntArrayLike = Sequence[int] | np.ndarray
FloatArrayLike = Sequence[float] | np.ndarray

__all__ = [
    "CategoryScore",
    "FloatArrayLike",
    "IntArrayLike",
    "Scores",
    "bootstrap_ci",
    "per_category",
    "per_category_columns",
    "pr_auc",
    "score",
]


@dataclass(frozen=True)
class Scores:
    """Headline metrics at one operating threshold, plus the threshold-free one."""

    n: int
    positives: int
    threshold: float
    recall: float
    precision: float
    f1: float
    pr_auc: float
    tp: int
    fp: int
    tn: int
    fn: int
    recall_ci: tuple[float, float] | None = None
    pr_auc_ci: tuple[float, float] | None = None

    @property
    def missed(self) -> int:
        """Distress cases routed to generated text. The failure that matters."""
        return self.fn


@dataclass(frozen=True)
class CategoryScore:
    """Where the errors actually live. The headline number hides this."""

    category: str
    n: int
    positives: int
    recall: float | None
    false_positive_rate: float | None
    errors: list[str] = field(default_factory=lambda: [])


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def score(
    y_true: IntArrayLike,
    y_score: FloatArrayLike,
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Scores:
    """Metrics at ``threshold``, with bootstrap intervals on recall and PR-AUC."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError(f"y_true {truth.shape} and y_score {scores.shape} differ in length")
    if truth.size == 0:
        raise ValueError("nothing to score")

    predicted = (scores >= threshold).astype(int)
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))

    recall = _rate(tp, tp + fn)
    precision = _rate(tp, tp + fp)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return Scores(
        n=int(truth.size),
        positives=int(np.sum(truth)),
        threshold=threshold,
        recall=recall,
        precision=precision,
        f1=f1,
        pr_auc=_pr_auc(truth, scores),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        recall_ci=bootstrap_ci(
            truth,
            scores,
            lambda t, s: _rate(
                int(np.sum((t == 1) & (s >= threshold))),
                int(np.sum(t == 1)),
            ),
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        pr_auc_ci=bootstrap_ci(truth, scores, _pr_auc, n_bootstrap=n_bootstrap, seed=seed),
    )


def _pr_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, or the positive rate when only one class is present.

    A resample with no positives has no meaningful precision-recall curve;
    returning the base rate keeps the bootstrap from being dominated by
    degenerate draws.
    """
    if len(np.unique(truth)) < 2:
        return float(np.mean(truth))
    return float(average_precision_score(truth, scores))


def pr_auc(y_true: IntArrayLike, y_score: FloatArrayLike) -> float:
    """Average precision: the threshold-free ranking metric every comparison uses."""
    return _pr_auc(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float))


def bootstrap_ci(
    y_true: IntArrayLike,
    y_score: FloatArrayLike,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval, or ``None`` when disabled.

    Resamples rows with replacement. Deterministic given ``seed`` — a committed
    report must reproduce.
    """
    if n_bootstrap <= 0:
        return None
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, truth.size, truth.size)
        sampled_truth = truth[index]
        if int(np.sum(sampled_truth)) == 0:
            # No positives drawn: recall is undefined rather than zero. Skipping
            # is honest; counting it as 0.0 would drag the interval down with an
            # artefact of the resampling.
            continue
        draws.append(metric(sampled_truth, scores[index]))
    if not draws:
        return None
    lower = float(np.percentile(draws, 100 * alpha / 2))
    upper = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return (lower, upper)


def per_category(
    rows: Sequence[Row],
    y_score: FloatArrayLike,
    *,
    threshold: float = 0.5,
    max_errors: int = 5,
) -> list[CategoryScore]:
    """Recall and false-positive rate per category, plus example failures.

    ``game-frustration`` is the row to read: it is where a lexical model
    collapses, and where a useful one has to earn its keep.
    """
    return per_category_columns(
        [row.id for row in rows],
        [row.label for row in rows],
        [row.category.value for row in rows],
        y_score,
        threshold=threshold,
        max_errors=max_errors,
    )


def per_category_columns(
    ids: Sequence[str],
    labels: Sequence[int],
    categories: Sequence[str],
    y_score: FloatArrayLike,
    *,
    threshold: float = 0.5,
    max_errors: int = 5,
) -> list[CategoryScore]:
    """:func:`per_category` over plain columns — for predictions saved to disk."""
    scores = np.asarray(y_score, dtype=float)
    if not len(ids) == len(labels) == len(categories) == scores.size:
        raise ValueError("rows and scores differ in length")

    by_category: dict[str, list[tuple[str, int, float]]] = {}
    for row_id, label, category, value in zip(ids, labels, categories, scores, strict=True):
        by_category.setdefault(category, []).append((row_id, label, float(value)))

    out: list[CategoryScore] = []
    for category, members in sorted(by_category.items()):
        positives = [(row_id, value) for row_id, label, value in members if label == 1]
        negatives = [(row_id, value) for row_id, label, value in members if label == 0]
        missed = [row_id for row_id, value in positives if value < threshold]
        flagged = [row_id for row_id, value in negatives if value >= threshold]
        out.append(
            CategoryScore(
                category=category,
                n=len(members),
                positives=len(positives),
                recall=(
                    None if not positives else _rate(len(positives) - len(missed), len(positives))
                ),
                false_positive_rate=(
                    None if not negatives else _rate(len(flagged), len(negatives))
                ),
                errors=(missed + flagged)[:max_errors],
            )
        )
    return out
