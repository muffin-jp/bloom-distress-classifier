"""The two rules that make every downstream number trustworthy.

If either of these regresses, the metrics keep looking fine — that is precisely
why they are tested rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_row

from dc.schema import Category, Provenance, Row
from dc.splits import (
    Split,
    build_assignment,
    check_leakage,
    group_rows,
    load_assignment,
    split_rows,
    write_assignment,
)


def corpus() -> list[Row]:
    """A dataset with golden rows, paraphrase groups, and every category."""
    rows: list[Row] = []
    for category in Category:
        for index in range(6):
            base = f"{category.value}-{index}"
            rows.append(make_row(base, category=category))
            # Two paraphrases per source, sharing its origin_id.
            for suffix in ("p1", "p2"):
                rows.append(
                    make_row(
                        f"{base}-{suffix}",
                        origin_id=base,
                        category=category,
                        provenance=Provenance.LLM_AUGMENTED,
                    )
                )
    rows.extend(
        make_row(f"golden-{index}", category=Category.DISTRESS, provenance=Provenance.SEED)
        for index in range(10)
    )
    return rows


def test_no_origin_group_straddles_the_split() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    seen: dict[str, set[Split]] = {}
    for row in rows:
        seen.setdefault(row.origin_id, set()).add(assignment[row.id])
    straddling = [origin_id for origin_id, splits in seen.items() if len(splits) > 1]
    assert straddling == [], f"paraphrase leakage via {straddling}"


def test_every_golden_row_is_in_test() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    assert all(assignment[row.id] == "test" for row in rows if row.is_golden)


def test_golden_rows_are_never_in_train() -> None:
    rows = corpus()
    train = split_rows(rows, build_assignment(rows), "train")
    assert not any(row.is_golden for row in train)


def test_assignment_covers_every_row() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    assert set(assignment) == {row.id for row in rows}


def test_assignment_is_deterministic() -> None:
    rows = corpus()
    assert build_assignment(rows) == build_assignment(rows)


def test_assignment_ignores_row_order() -> None:
    rows = corpus()
    assert build_assignment(rows) == build_assignment(list(reversed(rows)))


def test_adding_rows_does_not_reshuffle_existing_groups() -> None:
    """The property that makes splits.json safe to commit before milestone 2.

    A seeded shuffle would reassign everything each time the dataset grows,
    silently changing what "test" means partway through the project.
    """
    before = corpus()
    assignment_before = build_assignment(before)

    after = before + [
        make_row(f"extra-{index}", category=Category.GAME_FRUSTRATION) for index in range(30)
    ]
    assignment_after = build_assignment(after)

    moved = [
        row.id
        for row in before
        if assignment_before[row.id] != assignment_after[row.id]
        # Growth moves the cut point, so a group may newly join test; the
        # guarantee is that nothing already in test is pulled back into train.
        and assignment_before[row.id] == "test"
    ]
    assert moved == [], f"{len(moved)} test row(s) were pulled back into train: {moved[:5]}"


def test_every_category_reaches_the_test_split() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    test = split_rows(rows, assignment, "test")
    assert {row.category for row in test} == set(Category)


def test_different_salts_draw_different_splits() -> None:
    rows = corpus()
    assert build_assignment(rows) != build_assignment(rows, salt="other")


def test_group_rows_rejects_a_category_change_within_a_group() -> None:
    rows = [
        make_row("a", category=Category.GAME_FRUSTRATION),
        make_row("a-p1", origin_id="a", category=Category.DISTRESS),
    ]
    with pytest.raises(ValueError, match="spans categories"):
        group_rows(rows)


def test_check_leakage_passes_on_a_sound_assignment() -> None:
    rows = corpus()
    assert check_leakage(rows, build_assignment(rows)) == []


def test_check_leakage_catches_a_straddling_group() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    victim = next(row for row in rows if row.origin_id != row.id)
    assignment[victim.id] = "train" if assignment[victim.id] == "test" else "test"
    problems = check_leakage(rows, assignment)
    assert any("straddle" in problem for problem in problems)


def test_check_leakage_catches_a_golden_row_in_train() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    golden = next(row for row in rows if row.is_golden)
    assignment[golden.id] = "train"
    problems = check_leakage(rows, assignment)
    assert any("golden" in problem for problem in problems)


def test_check_leakage_catches_a_missing_assignment() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    del assignment[rows[0].id]
    assert any("no split assignment" in problem for problem in check_leakage(rows, assignment))


def test_invalid_test_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        build_assignment(corpus(), test_fraction=0.0)


def test_assignment_round_trips_through_disk(tmp_path: Path) -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    path = tmp_path / "splits.json"
    write_assignment(assignment, path)
    assert load_assignment(path) == assignment


def test_load_assignment_rejects_an_unknown_split(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text('{"assignment": {"a": "validation"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown split"):
        load_assignment(path)
