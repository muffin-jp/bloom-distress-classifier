"""Hyperparameter selection by grouped cross-validation, and the shortcut probe.

The rule is declared here, in code, before any result is seen
-----------------------------------------------------------
Choosing a rule after looking at the table is tuning twice on the same data, so
:data:`RULE` is fixed up front and :func:`select` implements exactly it:

1. **Feeling features are ineligible.** In this dataset the chip a player picked
   is close to a label leak: four of the seven feelings carry no distress at all,
   an artefact of how the rows were authored. :func:`feeling_swap_probe` measures
   the consequence, and it is the reason for the rule rather than a justification
   found afterwards — the leak was visible in the data before training began.
2. Among eligible configs, find the best mean fold PR-AUC.
3. Keep every config within **one standard error** of it. Differences smaller than
   the fold-to-fold noise are not evidence, and treating them as evidence is how
   a model gets chosen for its luck.
4. Among those, take the **lowest Brier score**. The cascade routes on probability
   bands, so among models that rank equally well, prefer the one whose scores are
   most nearly probabilities.
5. Break exact ties toward the simpler model: smaller ``C`` (stronger
   regularisation), then no class reweighting.

Every config is scored on **the same folds**, built once. That makes the
comparison paired: a config does not win because it happened to draw easier
folds.
"""

# scikit-learn and numpy leak Unknown under pyright strict at this boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from dc.calibration import brier_score
from dc.features import feeling_one_hot
from dc.metrics import pr_auc
from dc.schema import Feeling

__all__ = [
    "RULE",
    "ClassWeight",
    "Config",
    "ConfigResult",
    "Fold",
    "Selection",
    "SwapOutcome",
    "evaluate_config",
    "feeling_swap_probe",
    "grid",
    "make_estimator",
    "make_folds",
    "oof_segment_max",
    "probe_best_feeling_config",
    "select",
]

ClassWeight = Literal["none", "balanced"]
Fold = tuple[np.ndarray, np.ndarray]

RULE = (
    "Feeling features ineligible (label leak, see probe); best mean fold PR-AUC; "
    "keep configs within 1 SE; lowest Brier among them; ties to smaller C, then "
    "no class reweighting."
)


@dataclass(frozen=True)
class Config:
    C: float
    class_weight: ClassWeight
    include_feeling: bool

    @property
    def label(self) -> str:
        feeling = "+feeling" if self.include_feeling else ""
        return f"C={self.C:g} cw={self.class_weight}{feeling}"


def grid(
    Cs: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    class_weights: Sequence[ClassWeight] = ("none", "balanced"),
    feelings: Sequence[bool] = (False, True),
) -> list[Config]:
    return [
        Config(C=c, class_weight=w, include_feeling=f)
        for f in feelings
        for w in class_weights
        for c in Cs
    ]


def make_folds(
    y: np.ndarray, groups: np.ndarray, *, n_splits: int = 5, seed: int = 0
) -> list[Fold]:
    """Folds built once and shared by every config, grouped by ``origin_id``."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train, dtype=int), np.asarray(val, dtype=int))
        for train, val in splitter.split(np.zeros(len(y)), y, groups)
    ]


def make_estimator(config: Config, *, seed: int = 0) -> LogisticRegression:
    return LogisticRegression(
        C=config.C,
        class_weight=None if config.class_weight == "none" else "balanced",
        max_iter=5000,
        random_state=seed,
    )


def _design(config: Config, x_text: np.ndarray, x_feeling: np.ndarray) -> np.ndarray:
    return np.hstack([x_text, x_feeling]) if config.include_feeling else x_text


@dataclass(frozen=True, eq=False)
class ConfigResult:
    config: Config
    fold_pr_auc: tuple[float, ...]
    brier: float
    oof: np.ndarray = field(repr=False)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_pr_auc))

    @property
    def se(self) -> float:
        """Standard error of the mean across folds."""
        if len(self.fold_pr_auc) < 2:
            return 0.0
        return float(np.std(self.fold_pr_auc, ddof=1) / np.sqrt(len(self.fold_pr_auc)))


def evaluate_config(
    config: Config,
    x_text: np.ndarray,
    x_feeling: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    seed: int = 0,
) -> ConfigResult:
    """Fold PR-AUCs and out-of-fold probabilities for one config."""
    x = _design(config, x_text, x_feeling)
    oof = np.full(len(y), np.nan, dtype=float)
    scores: list[float] = []
    for train_index, val_index in folds:
        estimator = make_estimator(config, seed=seed)
        estimator.fit(x[train_index], y[train_index])
        predicted = estimator.predict_proba(x[val_index])[:, 1]
        oof[val_index] = predicted
        scores.append(pr_auc(y[val_index], predicted))
    if np.any(np.isnan(oof)):
        raise ValueError("folds do not cover every row; out-of-fold scores are incomplete")
    return ConfigResult(config, tuple(scores), brier_score(y, oof), oof)


def oof_segment_max(
    config: Config,
    x_text: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    segment_features: np.ndarray,
    segment_owner: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Each row's worst-segment score, from a model that never trained on that row.

    The skip band compares the highest-scoring segment of a note against ``low``,
    so ``low`` has to be fitted on that statistic — and fitting it on in-sample
    scores would set it from a model that had already seen those notes, which is
    exactly the optimism the threshold exists to guard against. So the fold models
    are refit here and each one scores only the segments of its held-out rows.

    ``segment_owner[i]`` is the row index segment ``i`` came from.
    """
    if config.include_feeling:
        raise ValueError(
            "segment scoring is undefined for a config that reads the feeling chip: a "
            "sentence inside a note has no chip of its own"
        )
    if segment_features.shape[0] != segment_owner.shape[0]:
        raise ValueError("segment features and owners differ in length")
    if segment_features.shape[1] != x_text.shape[1]:
        raise ValueError("segment features and row features have different widths")
    missing = set(range(len(y))) - set(segment_owner.tolist())
    if missing:
        raise ValueError(f"{len(missing)} row(s) produced no segments, so they cannot be scored")

    out = np.full(len(y), -np.inf, dtype=float)
    for train_index, val_index in folds:
        estimator = make_estimator(config, seed=seed)
        estimator.fit(x_text[train_index], y[train_index])
        rows = np.flatnonzero(np.isin(segment_owner, val_index))
        predicted = estimator.predict_proba(segment_features[rows])[:, 1]
        np.maximum.at(out, segment_owner[rows], predicted)
    if np.any(np.isneginf(out)):
        raise ValueError("folds do not cover every row; out-of-fold segment scores are incomplete")
    return out


@dataclass(frozen=True)
class Selection:
    chosen: ConfigResult
    best: ConfigResult
    survivors: tuple[ConfigResult, ...]
    excluded: tuple[ConfigResult, ...]
    rule: str = RULE


def select(results: Sequence[ConfigResult], *, allow_feeling: bool = False) -> Selection:
    """Apply :data:`RULE`. ``allow_feeling`` exists for tests, not for training."""
    eligible = [r for r in results if allow_feeling or not r.config.include_feeling]
    excluded = tuple(r for r in results if r not in eligible)
    if not eligible:
        raise ValueError("no eligible configs to select from")

    best = max(eligible, key=lambda r: (r.mean, -r.config.C))
    floor = best.mean - best.se
    survivors = tuple(r for r in eligible if r.mean >= floor)
    chosen = min(
        survivors,
        key=lambda r: (r.brier, r.config.C, r.config.class_weight != "none"),
    )
    return Selection(chosen=chosen, best=best, survivors=survivors, excluded=excluded)


@dataclass(frozen=True)
class SwapOutcome:
    """What happens to real distress cases if the player had picked another chip."""

    swap_to: str
    recall_before: float
    recall_after: float
    mean_score_before: float
    mean_score_after: float


def feeling_swap_probe(
    config: Config,
    x_text: np.ndarray,
    feelings: Sequence[Feeling],
    y: np.ndarray,
    folds: Sequence[Fold],
    swap_to: Sequence[Feeling],
    *,
    threshold: float = 0.5,
    seed: int = 0,
) -> list[SwapOutcome]:
    """Counterfactual: re-score each held-out distress case with a different chip.

    The text is untouched; only the feeling changes. A model that has learned the
    distinction being taught should not care. A model that has learned "this chip
    never means distress" will drop the case — which in production is a player
    who picked ``frustrated`` and then wrote a real crisis note, routed to
    generated encouragement. This probe turns that hazard from an argument into a
    number.
    """
    if not config.include_feeling:
        raise ValueError("the probe needs a config that reads the feeling")
    x_feeling = feeling_one_hot(feelings)
    x = np.hstack([x_text, x_feeling])
    offset = x_text.shape[1]

    before = np.full(len(y), np.nan, dtype=float)
    after = {target: np.full(len(y), np.nan, dtype=float) for target in swap_to}
    for train_index, val_index in folds:
        estimator = make_estimator(config, seed=seed)
        estimator.fit(x[train_index], y[train_index])
        before[val_index] = estimator.predict_proba(x[val_index])[:, 1]
        positives = val_index[y[val_index] == 1]
        if positives.size == 0:
            continue
        for target in swap_to:
            swapped = x[positives].copy()
            swapped[:, offset:] = feeling_one_hot([target] * positives.size)
            after[target][positives] = estimator.predict_proba(swapped)[:, 1]

    positive = y == 1
    outcomes: list[SwapOutcome] = []
    for target in swap_to:
        outcomes.append(
            SwapOutcome(
                swap_to=target.value,
                recall_before=float(np.mean(before[positive] >= threshold)),
                recall_after=float(np.mean(after[target][positive] >= threshold)),
                mean_score_before=float(np.mean(before[positive])),
                mean_score_after=float(np.mean(after[target][positive])),
            )
        )
    return outcomes


def probe_best_feeling_config(
    results: Sequence[ConfigResult],
    x_text: np.ndarray,
    feelings: Sequence[Feeling],
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    seed: int = 0,
) -> tuple[Config | None, list[SwapOutcome]]:
    """Probe the strongest feeling config against the chips that carry no distress.

    The swap targets are derived from the training data, not hard-coded: whichever
    feelings have zero positives are exactly the ones a leaked model has learned
    to treat as "never distress".
    """
    candidates = [r for r in results if r.config.include_feeling]
    if not candidates:
        return None, []
    best = max(candidates, key=lambda r: r.mean)
    positive_feelings = {f for f, label in zip(feelings, y, strict=True) if label == 1}
    targets = [f for f in Feeling if f in set(feelings) and f not in positive_feelings]
    if not targets:
        return best.config, []
    return best.config, feeling_swap_probe(
        best.config, x_text, feelings, y, folds, targets, seed=seed
    )
