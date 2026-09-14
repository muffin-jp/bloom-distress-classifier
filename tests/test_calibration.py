"""Calibration on cases where the answer is known by hand."""

from __future__ import annotations

import pytest

from dc.calibration import brier_score, expected_calibration_error, reliability_bins


def test_brier_is_zero_for_certain_and_correct() -> None:
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0


def test_brier_is_one_for_certain_and_wrong() -> None:
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0


def test_brier_of_a_coin_flip() -> None:
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_a_perfectly_calibrated_bin_has_no_error() -> None:
    # Says 0.25 four times, and one of the four is distress: exactly right.
    assert expected_calibration_error([1, 0, 0, 0], [0.25] * 4) == pytest.approx(0.0)


def test_an_overconfident_model_has_error() -> None:
    # Says 0.9 about cases that are distress half the time.
    assert expected_calibration_error([1, 0], [0.9, 0.9]) == pytest.approx(0.4)


def test_bins_record_what_was_said_and_what_happened() -> None:
    bins = reliability_bins([1, 0, 1, 1], [0.05, 0.05, 0.95, 0.95], n_bins=10)
    assert bins[0].n == 2
    assert bins[0].observed_rate == 0.5
    assert bins[-1].n == 2
    assert bins[-1].mean_predicted == pytest.approx(0.95)
    assert all(b.n == 0 for b in bins[1:-1])


def test_a_score_of_exactly_one_lands_in_the_last_bin() -> None:
    bins = reliability_bins([1], [1.0], n_bins=10)
    assert bins[-1].n == 1


def test_empty_bins_report_none_rather_than_zero() -> None:
    # A bin with no rows has no observed rate; zero would be a claim.
    bins = reliability_bins([1], [0.95], n_bins=10)
    assert bins[0].observed_rate is None
    assert bins[0].gap is None


@pytest.mark.parametrize("bad", [[1.5], [-0.1], [float("nan")]])
def test_out_of_range_probabilities_are_rejected(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="within"):
        brier_score([1], bad)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        brier_score([1, 0], [0.5])
