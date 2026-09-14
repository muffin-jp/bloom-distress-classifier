"""Scoring, on cases where the right answer is known by hand."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_row

from dc.metrics import bootstrap_ci, per_category, score
from dc.schema import Category


def test_perfect_separation() -> None:
    result = score([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.f1 == 1.0
    assert result.pr_auc == 1.0
    assert (result.tp, result.fp, result.tn, result.fn) == (2, 0, 2, 0)


def test_everything_predicted_negative() -> None:
    result = score([1, 1, 0, 0], [0.1, 0.2, 0.1, 0.2])
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.f1 == 0.0
    assert result.missed == 2


def test_missed_is_the_false_negatives() -> None:
    # The failure that matters: a distress case routed to generated text.
    result = score([1, 1, 1, 0], [0.9, 0.1, 0.1, 0.1])
    assert result.missed == 2
    assert result.recall == pytest.approx(1 / 3)


def test_a_constant_score_gives_pr_auc_at_the_base_rate() -> None:
    # This is why the majority baseline is in the table: it fixes what PR-AUC
    # is worth on this class balance before anything else claims a number.
    result = score([1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], n_bootstrap=0)
    assert result.pr_auc == pytest.approx(0.25)


def test_threshold_is_inclusive() -> None:
    assert score([1], [0.5], n_bootstrap=0).recall == 1.0
    assert score([1], [0.49], n_bootstrap=0).recall == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        score([1, 0], [0.5], n_bootstrap=0)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="nothing to score"):
        score([], [], n_bootstrap=0)


def test_bootstrap_is_deterministic_given_a_seed() -> None:
    # A committed report has to reproduce.
    truth, scores = [1, 1, 0, 0, 1, 0], [0.9, 0.4, 0.2, 0.7, 0.6, 0.1]
    assert score(truth, scores, seed=7).recall_ci == score(truth, scores, seed=7).recall_ci


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    truth = [1] * 20 + [0] * 60
    scores = [0.9] * 18 + [0.1] * 2 + [0.1] * 60
    result = score(truth, scores, seed=0)
    assert result.recall_ci is not None
    low, high = result.recall_ci
    assert low <= result.recall <= high


def test_bootstrap_can_be_disabled() -> None:
    assert score([1, 0], [0.9, 0.1], n_bootstrap=0).recall_ci is None


def test_bootstrap_returns_none_when_no_resample_has_positives() -> None:
    assert bootstrap_ci([0, 0, 0], [0.1, 0.2, 0.3], lambda t, s: 1.0, n_bootstrap=50) is None


def test_per_category_splits_recall_from_false_positives() -> None:
    rows = [
        make_row("d1", category=Category.DISTRESS),
        make_row("d2", category=Category.DISTRESS),
        make_row("g1", category=Category.GAME_FRUSTRATION),
        make_row("g2", category=Category.GAME_FRUSTRATION),
    ]
    entries = {c.category: c for c in per_category(rows, np.array([0.9, 0.1, 0.9, 0.1]))}

    distress = entries["distress"]
    assert distress.recall == 0.5
    assert distress.false_positive_rate is None  # no negatives in the category

    game = entries["game-frustration"]
    assert game.recall is None  # no positives in the category
    assert game.false_positive_rate == 0.5


def test_per_category_names_the_failing_rows() -> None:
    rows = [
        make_row("missed-me", category=Category.DISTRESS),
        make_row("flagged-me", category=Category.GAME_FRUSTRATION),
    ]
    entries = {c.category: c for c in per_category(rows, np.array([0.1, 0.9]))}
    assert entries["distress"].errors == ["missed-me"]
    assert entries["game-frustration"].errors == ["flagged-me"]


def test_per_category_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        per_category([make_row("a")], np.array([0.1, 0.2]))
