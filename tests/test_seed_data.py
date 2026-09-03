"""Guards on the committed seed file itself.

Separate from test_splits.py on purpose: those tests exercise the rules, these
assert that the data actually in the repo still satisfies them. If the companion
repo's dataset drifts, this is what notices.
"""

from __future__ import annotations

import pytest

from dc.schema import Category, load_dataset
from dc.splits import SEED_PATH, build_assignment, check_leakage

pytestmark = pytest.mark.skipif(
    not SEED_PATH.exists(), reason="data/seed.jsonl not imported yet (`make seed`)"
)

#: BUILD_SPEC.md §4.1 — 51 companion cases, minus 1 regeneration and 6 chip-only.
EXPECTED_GOLDEN = 44
EXPECTED_BY_CATEGORY = {
    Category.DISTRESS: 10,
    Category.GAME_FRUSTRATION: 10,
    Category.NORMAL_FEELING: 6,
    Category.MIXED_FEELING: 6,
    Category.INJECTION: 6,
    Category.NONSENSE: 6,
}


def test_seed_matches_the_spec_inventory() -> None:
    rows = load_dataset(SEED_PATH)
    assert len(rows) == EXPECTED_GOLDEN
    counts = {category: 0 for category in Category}
    for row in rows:
        counts[row.category] += 1
    assert counts == EXPECTED_BY_CATEGORY


def test_every_seed_row_is_golden_and_carries_free_text() -> None:
    rows = load_dataset(SEED_PATH)
    assert all(row.is_golden for row in rows)
    assert all(row.free_text.strip() for row in rows)


def test_seed_positives_are_exactly_the_distress_cases() -> None:
    rows = load_dataset(SEED_PATH)
    assert sum(row.label for row in rows) == EXPECTED_BY_CATEGORY[Category.DISTRESS]


def test_seed_alone_produces_an_all_test_assignment() -> None:
    # Before milestone 2 there is nothing to train on: every row is golden, so
    # the split is degenerate and check_leakage should say so rather than
    # pretending a train set exists.
    rows = load_dataset(SEED_PATH)
    assignment = build_assignment(rows)
    assert all(value == "test" for value in assignment.values())
    assert any("'train' split is empty" in problem for problem in check_leakage(rows, assignment))
