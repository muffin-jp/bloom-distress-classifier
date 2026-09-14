"""The selection rule, the shared folds, and the shortcut probe."""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import numpy as np
import pytest

from dc.schema import Feeling
from dc.selection import (
    Config,
    ConfigResult,
    evaluate_config,
    feeling_swap_probe,
    grid,
    make_folds,
    probe_best_feeling_config,
    select,
)


def result(
    C: float,
    folds: tuple[float, ...],
    brier: float,
    *,
    class_weight: str = "none",
    feeling: bool = False,
) -> ConfigResult:
    return ConfigResult(
        Config(C=C, class_weight=class_weight, include_feeling=feeling),  # type: ignore[arg-type]
        folds,
        brier,
        np.zeros(1),
    )


def leaky_data(seed: int = 0) -> tuple[np.ndarray, list[Feeling], np.ndarray, np.ndarray]:
    """Text weakly informative; feeling perfectly separates. The shortcut, on purpose."""
    rng = np.random.default_rng(seed)
    n = 200
    y = (np.arange(n) % 4 == 0).astype(int)
    x_text = rng.normal(size=(n, 8)).astype(np.float32)
    x_text[:, 0] += y * 1.5
    feelings = [Feeling.CUSTOM if label else Feeling.FRUSTRATED for label in y]
    groups = np.array([f"g{i}" for i in range(n)])
    return x_text, feelings, y, groups


def test_grid_covers_every_combination() -> None:
    assert len(grid()) == 5 * 2 * 2
    assert len({(c.C, c.class_weight, c.include_feeling) for c in grid()}) == 20


def test_folds_never_split_an_origin_group() -> None:
    y = np.array([i % 3 == 0 for i in range(60)], dtype=int)
    groups = np.array([f"g{i // 3}" for i in range(60)])  # three rows per family
    for train, val in make_folds(y, groups):
        assert not set(groups[train]) & set(groups[val])


def test_folds_are_deterministic() -> None:
    y = np.array([i % 3 == 0 for i in range(60)], dtype=int)
    groups = np.array([f"g{i}" for i in range(60)])
    first = make_folds(y, groups, seed=1)
    second = make_folds(y, groups, seed=1)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, second, strict=True))


def test_select_never_chooses_a_feeling_config() -> None:
    results = [
        result(1.0, (0.99, 0.99), 0.01, feeling=True),  # best by far, and ineligible
        result(1.0, (0.80, 0.82), 0.10),
    ]
    selection = select(results)
    assert not selection.chosen.config.include_feeling
    assert selection.excluded == (results[0],)


def test_select_prefers_lowest_brier_within_one_standard_error() -> None:
    # 0.90 is nominally best, but 0.89 is inside its SE and better calibrated.
    best = result(1.0, (0.86, 0.94), 0.20)
    calibrated = result(10.0, (0.85, 0.93), 0.05)
    worse = result(0.1, (0.50, 0.52), 0.01)  # great Brier, but outside the SE band
    selection = select([best, calibrated, worse])
    assert selection.best is best
    assert selection.chosen is calibrated
    assert worse not in selection.survivors


def test_select_breaks_brier_ties_toward_stronger_regularisation() -> None:
    selection = select([result(10.0, (0.9, 0.9), 0.1), result(0.1, (0.9, 0.9), 0.1)])
    assert selection.chosen.config.C == 0.1


def test_select_breaks_remaining_ties_toward_no_reweighting() -> None:
    selection = select(
        [
            result(1.0, (0.9, 0.9), 0.1, class_weight="balanced"),
            result(1.0, (0.9, 0.9), 0.1, class_weight="none"),
        ]
    )
    assert selection.chosen.config.class_weight == "none"


def test_select_refuses_when_nothing_is_eligible() -> None:
    with pytest.raises(ValueError, match="no eligible"):
        select([result(1.0, (0.9, 0.9), 0.1, feeling=True)])


def test_evaluate_config_produces_out_of_fold_scores_for_every_row() -> None:
    x_text, feelings, y, groups = leaky_data()
    from dc.features import feeling_one_hot

    folds = make_folds(y, groups)
    outcome = evaluate_config(
        Config(1.0, "none", False), x_text, feeling_one_hot(feelings), y, folds
    )
    assert len(outcome.fold_pr_auc) == 5
    assert not np.any(np.isnan(outcome.oof))
    assert 0.0 <= outcome.brier <= 1.0


def test_the_feeling_shortcut_beats_text_in_cross_validation() -> None:
    """Why cross-validation cannot be what rejects the shortcut."""
    from dc.features import feeling_one_hot

    x_text, feelings, y, groups = leaky_data()
    folds = make_folds(y, groups)
    x_feeling = feeling_one_hot(feelings)
    text_only = evaluate_config(Config(1.0, "none", False), x_text, x_feeling, y, folds)
    with_chip = evaluate_config(Config(1.0, "none", True), x_text, x_feeling, y, folds)
    assert with_chip.mean > text_only.mean


def test_the_probe_catches_a_model_that_learned_the_shortcut() -> None:
    x_text, feelings, y, groups = leaky_data()
    folds = make_folds(y, groups)
    [outcome] = feeling_swap_probe(
        Config(1.0, "none", True), x_text, feelings, y, folds, [Feeling.FRUSTRATED]
    )
    # Same text, different chip: the leaked model stops seeing the distress.
    assert outcome.recall_after < outcome.recall_before
    assert outcome.mean_score_after < outcome.mean_score_before


def test_the_probe_refuses_a_config_that_cannot_see_the_feeling() -> None:
    x_text, feelings, y, groups = leaky_data()
    with pytest.raises(ValueError, match="reads the feeling"):
        feeling_swap_probe(
            Config(1.0, "none", False), x_text, feelings, y, make_folds(y, groups), []
        )


def test_probe_targets_are_derived_from_the_data() -> None:
    from dc.features import feeling_one_hot

    x_text, feelings, y, groups = leaky_data()
    folds = make_folds(y, groups)
    results = [
        evaluate_config(Config(1.0, "none", True), x_text, feeling_one_hot(feelings), y, folds)
    ]
    config, outcomes = probe_best_feeling_config(results, x_text, feelings, y, folds)
    assert config is not None and config.include_feeling
    # FRUSTRATED is the only chip here with no positives, so it is the only target.
    assert [o.swap_to for o in outcomes] == ["frustrated"]


def test_probe_is_skipped_when_no_feeling_config_was_scored() -> None:
    x_text, feelings, y, groups = leaky_data()
    assert probe_best_feeling_config([], x_text, feelings, y, make_folds(y, groups)) == (None, [])
