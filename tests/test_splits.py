"""The two rules that make every downstream number trustworthy.

If either of these regresses, the metrics keep looking fine — that is precisely
why they are tested rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_row

from dc.schema import Category, Provenance, Row
from dc.splits import (
    Split,
    build_assignment,
    check_leakage,
    dominant_category,
    drifted_groups,
    group_rows,
    load_all_rows,
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


def test_a_drifted_family_is_kept_whole() -> None:
    """Review can split a family's categories. The family still stays together.

    A reviewer deciding a generated variant drifted from its seed is review
    working. Splitting the family to keep stratification exact would trade a
    silent, unfalsifiable metric inflation for a rounding error in class
    balance — the wrong way round.
    """
    rows = [
        make_row("a", category=Category.GAME_FRUSTRATION),
        make_row("a-p1", origin_id="a", category=Category.DISTRESS),
    ]
    groups = group_rows(rows)
    assert set(groups) == {"a"}
    assert len(groups["a"]) == 2


def test_a_drifted_family_never_straddles_the_split() -> None:
    rows = corpus()
    # Flip one variant the way a reviewer would, mid-family.
    victim = next(r for r in rows if r.origin_id != r.id and not r.is_golden)
    rows = [
        make_row(r.id, origin_id=r.origin_id, category=Category.DISTRESS, provenance=r.provenance)
        if r.id == victim.id
        else r
        for r in rows
    ]
    assignment = build_assignment(rows)
    family = [r for r in rows if r.origin_id == victim.origin_id]
    assert len({assignment[r.id] for r in family}) == 1
    assert check_leakage(rows, assignment) == []


def test_drifted_groups_reports_the_divergence() -> None:
    rows = [
        make_row("a", category=Category.GAME_FRUSTRATION),
        make_row("a-p1", origin_id="a", category=Category.DISTRESS),
        make_row("b", category=Category.NONSENSE),
    ]
    assert drifted_groups(rows) == {"a": ["distress", "game-frustration"]}


def test_drifted_groups_is_empty_when_nothing_drifted() -> None:
    assert drifted_groups(corpus()) == {}


def test_dominant_category_is_the_majority() -> None:
    rows = [
        make_row("a", category=Category.GAME_FRUSTRATION),
        make_row("a-p1", origin_id="a", category=Category.GAME_FRUSTRATION),
        make_row("a-p2", origin_id="a", category=Category.DISTRESS),
    ]
    assert dominant_category(rows) is Category.GAME_FRUSTRATION


def test_dominant_category_breaks_ties_deterministically() -> None:
    # The split is a committed artifact: the same rows must always produce the
    # same assignment, so a tie cannot depend on dict or set ordering.
    rows = [
        make_row("a", category=Category.GAME_FRUSTRATION),
        make_row("a-p1", origin_id="a", category=Category.DISTRESS),
    ]
    assert dominant_category(rows) is dominant_category(list(reversed(rows)))


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


def test_check_leakage_refuses_a_dataset_with_no_golden_rows() -> None:
    """The vacuous pass, closed.

    "Every golden row is in test" is trivially true when there are none — which
    is exactly how the 44 release-gate cases fell out of the split with every
    check reporting green.
    """
    rows = [make_row(f"x-{i}", category=Category.NONSENSE) for i in range(20)]
    assignment = build_assignment(rows)
    assert any("no golden rows" in problem for problem in check_leakage(rows, assignment))


def test_load_all_rows_merges_the_seed_with_the_reviewed_set(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    labelled = tmp_path / "labelled.jsonl"
    seed.write_text(
        json.dumps(make_row("g-1", provenance=Provenance.SEED).model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    labelled.write_text(
        json.dumps(make_row("r-1").model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    rows = load_all_rows(seed, labelled)
    assert {row.id for row in rows} == {"g-1", "r-1"}
    assert sum(1 for row in rows if row.is_golden) == 1


def test_load_all_rows_rejects_a_golden_row_that_was_also_reviewed(tmp_path: Path) -> None:
    # A golden case appearing in labelled.jsonl means it went through review,
    # which would make it training data.
    payload = json.dumps(make_row("g-1", provenance=Provenance.SEED).model_dump(mode="json"))
    seed = tmp_path / "seed.jsonl"
    labelled = tmp_path / "labelled.jsonl"
    seed.write_text(payload + "\n", encoding="utf-8")
    labelled.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="appear in both"):
        load_all_rows(seed, labelled)


def test_load_all_rows_needs_at_least_one_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No dataset found"):
        load_all_rows(tmp_path / "a.jsonl", tmp_path / "b.jsonl")
