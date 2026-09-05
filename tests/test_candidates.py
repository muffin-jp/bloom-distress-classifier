"""The one-way door: a candidate must not become a row without a human."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from dc.candidates import Candidate, load_candidates, promote, review_order
from dc.schema import Category, Feeling, Provenance


def candidate(
    cid: str = "c-1",
    *,
    category: Category = Category.GAME_FRUSTRATION,
    votes: tuple[int, ...] = (),
    confidence: float | None = None,
) -> Candidate:
    return Candidate(
        id=cid,
        origin_id=cid,
        feeling=Feeling.FRUSTRATED,
        free_text="this stage is killing me",
        category=category,
        provenance=Provenance.HUMAN_WRITTEN,
        teacher_votes=votes,
        teacher_confidence=confidence,
    )


def test_proposed_label_follows_category() -> None:
    assert candidate(category=Category.DISTRESS).proposed_label == 1
    assert candidate(category=Category.NONSENSE).proposed_label == 0


def test_teacher_label_is_none_before_the_teacher_runs() -> None:
    assert candidate().teacher_label is None
    assert not candidate().is_contested
    assert not candidate().is_unstable


def test_teacher_label_is_the_majority_vote() -> None:
    assert candidate(votes=(1, 1, 0)).teacher_label == 1
    assert candidate(votes=(0, 0, 1)).teacher_label == 0


def test_a_tied_vote_breaks_toward_distress() -> None:
    # An undecided teacher is exactly the case a human should see, and flagging
    # it positive puts it at the top of the queue instead of passing quietly.
    assert candidate(votes=(1, 0)).teacher_label == 1


def test_contested_means_the_teacher_disagrees_with_the_author() -> None:
    assert candidate(category=Category.GAME_FRUSTRATION, votes=(1, 1, 1)).is_contested
    assert not candidate(category=Category.GAME_FRUSTRATION, votes=(0, 0, 0)).is_contested
    assert candidate(category=Category.DISTRESS, votes=(0, 0, 0)).is_contested


def test_unstable_means_the_teacher_disagreed_with_itself() -> None:
    assert candidate(votes=(1, 0, 1)).is_unstable
    assert not candidate(votes=(1, 1, 1)).is_unstable


def test_non_binary_votes_are_rejected() -> None:
    with pytest.raises(ValueError):
        candidate(votes=(2,))


def test_confidence_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        candidate(confidence=1.5)


def test_review_order_puts_contested_first_then_unstable() -> None:
    ordered = review_order(
        [
            candidate("calm", votes=(0, 0, 0), confidence=0.9),
            # Unstable but NOT contested: the teacher wavered yet still landed
            # on the author's answer.
            candidate("unstable", category=Category.DISTRESS, votes=(1, 0, 1), confidence=0.9),
            candidate("contested", votes=(1, 1, 1), confidence=0.9),
        ]
    )
    assert [c.id for c in ordered] == ["contested", "unstable", "calm"]


def test_contested_and_unstable_together_sorts_above_contested_alone() -> None:
    # The teacher contradicting both the author and itself is the strongest
    # signal in the queue: two independent reasons to doubt the proposed label.
    ordered = review_order(
        [
            candidate("contested-only", votes=(1, 1, 1), confidence=0.9),
            candidate("both", votes=(1, 0, 1), confidence=0.9),
        ]
    )
    assert [c.id for c in ordered] == ["both", "contested-only"]


def test_review_order_breaks_ties_by_confidence() -> None:
    ordered = review_order(
        [
            candidate("sure", votes=(0, 0, 0), confidence=0.95),
            candidate("unsure", votes=(0, 0, 0), confidence=0.30),
        ]
    )
    assert [c.id for c in ordered] == ["unsure", "sure"]


def test_promote_requires_a_reviewer() -> None:
    with pytest.raises(ValueError, match="reviewed_by is required"):
        promote(candidate(), category=Category.GAME_FRUSTRATION, reviewed_by="   ")


def test_promote_derives_the_label_from_the_reviewed_category() -> None:
    row = promote(candidate(), category=Category.DISTRESS, reviewed_by="tester")
    assert row.label == 1
    assert row.category is Category.DISTRESS


def test_promote_can_flip_the_authors_proposal() -> None:
    # The reviewer overriding the author is the entire point of review.
    source = candidate(category=Category.DISTRESS)
    row = promote(source, category=Category.GAME_FRUSTRATION, reviewed_by="tester")
    assert row.label == 0


def test_promote_stamps_the_review_date() -> None:
    row = promote(
        candidate(),
        category=Category.GAME_FRUSTRATION,
        reviewed_by="tester",
        reviewed_on=date(2026, 9, 3),
    )
    assert row.reviewed_on == date(2026, 9, 3)


def test_promoted_rows_keep_provenance() -> None:
    source = candidate()
    row = promote(source, category=source.category, reviewed_by="tester")
    assert row.provenance is Provenance.HUMAN_WRITTEN


def test_candidates_cannot_carry_a_label_field() -> None:
    # A candidate with a `label` would look like ground truth. Only rows have one.
    with pytest.raises(ValueError):
        Candidate.model_validate(
            {
                "id": "c-1",
                "origin_id": "c-1",
                "feeling": "frustrated",
                "free_text": "x",
                "category": "distress",
                "provenance": "seed",
                "label": 1,
            }
        )


def test_duplicate_ids_across_files_are_rejected(tmp_path: Path) -> None:
    def write(name: str, cid: str) -> Path:
        path = tmp_path / name
        payload: dict[str, Any] = {
            "id": cid,
            "origin_id": cid,
            "feeling": "frustrated",
            "free_text": "x",
            "category": "nonsense",
            "provenance": "llm-augmented",
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    with pytest.raises(ValueError, match="duplicate candidate id"):
        load_candidates(write("a.jsonl", "dup"), write("b.jsonl", "dup"))


def test_missing_candidate_files_are_skipped(tmp_path: Path) -> None:
    assert load_candidates(tmp_path / "nope.jsonl") == []
