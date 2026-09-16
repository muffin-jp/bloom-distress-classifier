"""Routing, the safety invariant, and how each threshold is fitted."""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import math

import numpy as np
import pytest

from dc.cascade import (
    CostModel,
    Route,
    Thresholds,
    constraint_floor,
    cost_curve,
    decisive_positives,
    envelope,
    fit_cascade,
    fit_high,
    fit_low,
    route,
    simulate,
    skip_band_positives,
    teacher_probabilities,
)

BANDS = Thresholds(low=0.1, high=0.8)


def labels(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Overlapping scores, and an LLM that is usually right.

    Positives land in [0.35, 1.0] and negatives in [0.0, 0.65], so the classes
    overlap in the middle. With perfectly separated scores one operating point
    dominates at every ratio, and there is no staircase to test.
    """
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 3 == 0).astype(int)
    scores = np.clip(0.35 * y + rng.uniform(0.0, 0.65, n), 0.0, 1.0)
    teacher = np.where(y == 1, rng.choice([1.0, 2 / 3, 1 / 3], n), rng.choice([0.0, 1 / 3], n))
    return y, scores, teacher, [f"r{i}" for i in range(n)]


# --- routing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.05, Route.SKIP_LLM),
        (0.1, Route.ESCALATE),  # exactly on low: escalate
        (0.5, Route.ESCALATE),
        (0.8, Route.ESCALATE),  # exactly on high: escalate
        (0.95, Route.SUPPORT),
    ],
)
def test_route_by_band(score: float, expected: Route) -> None:
    assert route(score, BANDS) is expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.01, 1.01, None, "0.5", True])
def test_anything_that_is_not_a_probability_escalates(bad: object) -> None:
    # Every way the local model can fail lands on today's behaviour — never on a
    # skipped LLM call.
    assert route(bad, BANDS) is Route.ESCALATE


@pytest.mark.parametrize(("low", "high"), [(0.8, 0.1), (-0.1, 0.5), (0.1, 1.5), (math.nan, 0.5)])
def test_thresholds_must_be_ordered_probabilities(low: float, high: float) -> None:
    with pytest.raises(ValueError):
        Thresholds(low, high)


def test_costs_must_be_positive() -> None:
    with pytest.raises(ValueError):
        CostModel(false_negative=0.0)
    assert CostModel(20.0, 1.0).ratio == 20.0


# --- simulation ---------------------------------------------------------------


def test_teacher_probability_is_the_share_of_votes() -> None:
    assert teacher_probabilities([(1, 1, 0), (0, 0, 0)]).tolist() == pytest.approx([2 / 3, 0.0])


def test_rows_without_votes_cannot_be_simulated() -> None:
    with pytest.raises(ValueError, match="no teacher votes"):
        teacher_probabilities([(1,), ()])


def test_escalating_everything_is_the_llm_alone() -> None:
    y = np.array([1, 1, 0, 0])
    q = np.array([1.0, 0.5, 0.25, 0.0])
    outcome = simulate(y, np.array([0.9, 0.1, 0.5, 0.0]), q, Thresholds(0.0, 1.0))
    assert outcome.escalate_share == 1.0
    assert outcome.expected_recall == pytest.approx(0.75)
    assert outcome.expected_false_positives == pytest.approx(0.25)


def test_simulation_arithmetic_by_hand() -> None:
    # r0 positive supports (caught), r1 positive escalates with q=0.5 (half missed),
    # r2 negative supports (false alarm), r3 negative skips (nothing).
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.5, 0.95, 0.05])
    q = np.array([1.0, 0.5, 0.0, 0.0])
    outcome = simulate(y, scores, q, BANDS, costs=CostModel(20.0, 1.0))
    assert outcome.expected_missed == pytest.approx(0.5)
    assert outcome.expected_false_positives == pytest.approx(1.0)
    assert outcome.expected_cost == pytest.approx(20.0 * 0.5 + 1.0)
    assert (outcome.skip_share, outcome.escalate_share, outcome.support_share) == (0.25, 0.25, 0.5)


def test_a_missing_score_escalates_in_simulation_as_in_routing() -> None:
    outcome = simulate(np.array([1]), np.array([np.nan]), np.array([1.0]), BANDS)
    assert outcome.escalate_share == 1.0


# --- low: the safety constraint ------------------------------------------------


def test_low_is_the_margin_times_the_lowest_positive_score() -> None:
    y = np.array([1, 1, 0])
    fit = fit_low(
        y, np.array([0.4, 0.2, 0.01]), ids=["a", "b", "c"], texts=["", "", ""], margin=0.5
    )
    assert fit.lowest_positive_score == 0.2
    assert fit.threshold == pytest.approx(0.1)
    assert [row_id for row_id, _, _ in fit.nearest_positives] == ["b", "a"]


def test_the_miss_bound_is_the_exact_zero_event_upper_limit() -> None:
    y, scores, _, ids = labels()
    fit = fit_low(y, scores, ids=ids, texts=ids)
    assert fit.miss_upper_bound == pytest.approx(1 - 0.05 ** (1 / int(y.sum())))


@pytest.mark.parametrize("margin", [0.0, 1.5])
def test_margin_must_be_in_the_unit_interval(margin: float) -> None:
    y, scores, _, ids = labels()
    with pytest.raises(ValueError, match="margin"):
        fit_low(y, scores, ids=ids, texts=ids, margin=margin)


@pytest.mark.parametrize("seed", range(10))
def test_no_validation_distress_case_ever_skips_the_llm(seed: int) -> None:
    """The one property the construction exists to guarantee, checked row by row."""
    y, scores, teacher, ids = labels(seed=seed)
    votes = [(int(q * 3 + 0.5) * [1] + (3 - int(q * 3 + 0.5)) * [0]) for q in teacher]
    fit = fit_cascade(y, scores, votes, ids=ids, texts=ids)
    assert skip_band_positives(y, scores, fit.thresholds) == []
    assert all(
        route(float(s), fit.thresholds) is not Route.SKIP_LLM
        for s, label in zip(scores, y, strict=True)
        if label == 1
    )


def test_on_fitted_rows_the_cascade_cannot_lose_recall_to_the_llm() -> None:
    # True by construction — no positive skips, and support only adds catches —
    # which is why the no-regression check belongs on the test set instead.
    for seed in range(10):
        y, scores, teacher, ids = labels(seed=seed)
        votes = [(int(q * 3 + 0.5) * [1] + (3 - int(q * 3 + 0.5)) * [0]) for q in teacher]
        fit = fit_cascade(y, scores, votes, ids=ids, texts=ids)
        assert fit.cascade.expected_recall >= fit.llm_alone.expected_recall - 1e-9


# --- high: the cost model --------------------------------------------------------


def test_a_useless_model_never_earns_a_support_band() -> None:
    # A perfect LLM and random scores: routing anything straight to support only
    # adds false alarms.
    rng = np.random.default_rng(1)
    y = (np.arange(60) % 3 == 0).astype(int)
    scores = rng.uniform(0.0, 1.0, 60)
    assert fit_high(y, scores, y.astype(float), low=0.0) == 1.0


def test_a_perfect_model_with_a_weak_llm_sends_positives_to_support() -> None:
    y = np.array([1] * 10 + [0] * 20)
    scores = np.array([0.9] * 10 + [0.1] * 20)
    teacher = np.array([0.5] * 10 + [0.0] * 20)
    high = fit_high(y, scores, teacher, low=0.0)
    assert 0.1 < high < 0.9
    assert simulate(y, scores, teacher, Thresholds(0.0, high)).expected_recall == 1.0


def test_candidate_cutoffs_never_sit_on_a_validation_score() -> None:
    y, scores, teacher, _ = labels()
    curve = cost_curve(y, scores, teacher, low=0.05)
    on_a_row = set(curve.candidates[:-1].tolist()) & set(scores.tolist())
    assert on_a_row == set()


def test_ties_go_to_the_higher_cutoff() -> None:
    # Both negatives: any support band costs false alarms, and no band costs
    # nothing — so every cutoff that routes nobody to support ties, and 1.0 wins.
    y = np.array([1, 0, 0])
    scores = np.array([0.5, 0.2, 0.3])
    teacher = np.array([1.0, 0.0, 0.0])
    assert fit_high(y, scores, teacher, low=0.0) == 1.0


# --- the staircase ----------------------------------------------------------------


def test_envelope_steps_down_as_the_ratio_rises() -> None:
    y, scores, teacher, _ = labels(n=90)
    segments = envelope(cost_curve(y, scores, teacher, low=0.0))
    assert len(segments) >= 2
    for before, after in zip(segments, segments[1:], strict=False):
        assert after.high < before.high
        assert after.expected_missed <= before.expected_missed
        assert after.expected_false_positives >= before.expected_false_positives
        assert before.ratio_to == pytest.approx(after.ratio_from)
    assert segments[0].ratio_from == 0.0
    assert segments[-1].ratio_to is None


def test_switch_points_are_exact() -> None:
    y, scores, teacher, _ = labels(n=90)
    curve = cost_curve(y, scores, teacher, low=0.0)
    segments = envelope(curve)
    for before, after in zip(segments, segments[1:], strict=False):
        switch = after.ratio_from
        assert (
            fit_high(y, scores, teacher, low=0.0, costs=CostModel(switch * 0.999, 1.0))
            == before.high
        )
        assert (
            fit_high(y, scores, teacher, low=0.0, costs=CostModel(switch * 1.001, 1.0))
            == after.high
        )


def test_the_chosen_segment_is_the_one_the_ratio_selects() -> None:
    y, scores, teacher, ids = labels(n=90)
    votes = [(int(q * 3 + 0.5) * [1] + (3 - int(q * 3 + 0.5)) * [0]) for q in teacher]
    fit = fit_cascade(y, scores, votes, ids=ids, texts=ids)
    chosen = fit.envelope[fit.chosen_segment]
    assert chosen.contains(fit.costs.ratio)
    assert chosen.high == pytest.approx(fit.thresholds.high)


def test_decisive_positives_are_the_rows_the_trade_rests_on() -> None:
    y = np.array([1, 1, 1, 0, 1])
    scores = np.array([0.30, 0.40, 0.45, 0.35, 0.90])
    teacher = np.array([0.0, 1.0, 1 / 3, 0.0, 0.0])
    rows = decisive_positives(
        y, scores, teacher, ids=["a", "b", "c", "d", "e"], texts=["", "", "", "", ""],
        high=0.2, alternative_high=0.5,
    )  # fmt: skip
    # b is caught by the LLM either way; d is not distress; e supports either way.
    assert [row_id for row_id, _, _, _ in rows] == ["a", "c"]


# --- a product rule outranking the cost model ----------------------------------------


def test_the_floor_is_the_highest_forbidden_score() -> None:
    scores = np.array([0.1, 0.9, 0.4, 0.95])
    assert constraint_floor(scores, np.array([1, 0, 1, 0])) == pytest.approx(0.4)
    assert constraint_floor(scores, np.zeros(4)) == 0.0


def test_the_floor_rejects_a_mismatched_mask() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        constraint_floor(np.array([0.1, 0.2]), np.array([1]))


def test_no_forbidden_row_can_be_routed_to_support() -> None:
    """The property the constraint exists to guarantee, over many draws."""
    for seed in range(10):
        y, scores, teacher, _ = labels(n=90, seed=seed)
        forbidden = y == 0
        high = fit_high(y, scores, teacher, low=0.0, forbidden=forbidden)
        assert not np.any(scores[forbidden] > high)


def test_the_constraint_only_ever_raises_the_cutoff() -> None:
    y, scores, teacher, _ = labels(n=90)
    free = fit_high(y, scores, teacher, low=0.0)
    constrained = fit_high(y, scores, teacher, low=0.0, forbidden=(y == 0))
    assert constrained >= free


def test_an_unconstrained_fit_is_unchanged() -> None:
    # Passing no mask must leave the original behaviour exactly as it was.
    y, scores, teacher, _ = labels(n=90)
    assert fit_high(y, scores, teacher, low=0.0) == fit_high(
        y, scores, teacher, low=0.0, forbidden=np.zeros(len(y))
    )


def test_the_envelope_offers_only_permitted_operating_points() -> None:
    y, scores, teacher, _ = labels(n=90)
    floor = constraint_floor(scores, y == 0)
    segments = envelope(cost_curve(y, scores, teacher, low=0.0, floor=floor))
    assert all(segment.high >= floor for segment in segments)


def test_a_curve_that_permits_nothing_says_so() -> None:
    y, scores, teacher, _ = labels()
    curve = cost_curve(y, scores, teacher, low=0.0, floor=1.5)
    with pytest.raises(ValueError, match="permits no cutoff"):
        curve.best_index(CostModel(20.0, 1.0))


def test_the_fit_records_which_row_set_the_floor() -> None:
    y, scores, teacher, ids = labels(n=90)
    votes = [(int(q * 3 + 0.5) * [1] + (3 - int(q * 3 + 0.5)) * [0]) for q in teacher]
    fit = fit_cascade(
        y, scores, votes, ids=ids, texts=ids, forbidden=(y == 0), forbidden_label="no negatives"
    )
    assert fit.constraint is not None
    worst = int(np.argmax(np.where(y == 0, scores, -1)))
    assert fit.constraint.set_by == ids[worst]
    assert fit.constraint.floor == pytest.approx(scores[worst])
    assert fit.thresholds.high >= fit.constraint.floor


def test_an_unconstrained_fit_records_no_constraint() -> None:
    y, scores, teacher, ids = labels()
    votes = [(int(q * 3 + 0.5) * [1] + (3 - int(q * 3 + 0.5)) * [0]) for q in teacher]
    assert fit_cascade(y, scores, votes, ids=ids, texts=ids).constraint is None
