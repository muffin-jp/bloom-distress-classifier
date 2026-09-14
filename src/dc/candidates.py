"""Unreviewed examples, and the one-way door that turns them into labelled rows.

A :class:`Candidate` is a row whose label nobody has vouched for yet. It carries
a *proposed* label and, optionally, what the teacher model thought — but it is
not training data and cannot be loaded by ``dc.schema``. The only way across is
:func:`promote`, which requires a human name and stamps a review date.

That separation is the whole point. The teacher's job is to make review *fast*
by proposing an answer; if the teacher could also confirm its own answer, its
mistakes would become ground truth and the student would learn to reproduce
them. Reviewing a proposal is quick. Assigning a label from scratch is not.
That is the only reason the teacher is in the loop at all.

Review priority
---------------
Not every candidate deserves the same attention, and the teacher tells you which
ones do. Three signals, most alarming first:

* **contested** — the teacher disagrees with the author's intent. Exactly one of
  them is wrong and a human has to say which.
* **unstable** — the teacher gave different answers across repeated runs on the
  same text. Genuinely ambiguous, and the most valuable rows in the set.
* **low confidence** — the teacher's own self-reported number. Weakest of the
  three: it is a verbalised guess, not a calibrated probability, so it is used
  to order a queue and never as a label.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dc.schema import MAX_FREE_TEXT, POSITIVE_CATEGORY, Category, Feeling, Provenance, Row

CANDIDATE_DIR = Path(__file__).resolve().parents[2] / "data" / "candidates"

__all__ = [
    "CANDIDATE_DIR",
    "Candidate",
    "load_candidates",
    "load_teacher_votes",
    "promote",
    "review_order",
]


class Candidate(BaseModel):
    """One proposed example, awaiting human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    origin_id: str = Field(min_length=1)
    feeling: Feeling
    free_text: str = Field(min_length=1, max_length=MAX_FREE_TEXT)
    #: The author's *intent*. Implies ``proposed_label`` — there is no separate
    #: label field, so a candidate cannot contradict itself the way a row could.
    category: Category
    provenance: Provenance
    #: Why this example exists — e.g. ``"mode:death-idiom frame:game"``. Read as
    #: documentation: the curated pack is a taxonomy of confusion modes, and this
    #: is what names the mode.
    note: str | None = None
    #: One entry per teacher run, 1 = distress. Empty until propose_labels runs.
    teacher_votes: tuple[int, ...] = ()
    #: The teacher's self-reported confidence in its own answer, 0..1.
    teacher_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("teacher_votes")
    @classmethod
    def _votes_are_binary(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(vote not in (0, 1) for vote in value):
            raise ValueError(f"teacher_votes must be 0 or 1, got {value}")
        return value

    @property
    def proposed_label(self) -> int:
        """Derived from ``category`` — the same rule ``dc.schema`` enforces."""
        return 1 if self.category is POSITIVE_CATEGORY else 0

    @property
    def teacher_label(self) -> int | None:
        """Majority vote across runs, or ``None`` if the teacher has not run.

        Ties break toward distress. A split vote means the teacher could not
        decide, and on a safety classifier the undecided case is the one you
        want a human to look at — flagging it as positive puts it at the top of
        the review queue rather than letting it pass quietly as a negative.
        """
        if not self.teacher_votes:
            return None
        positives = sum(self.teacher_votes)
        return 1 if positives * 2 >= len(self.teacher_votes) else 0

    @property
    def is_contested(self) -> bool:
        """The teacher disagrees with the author. One of them is wrong."""
        teacher = self.teacher_label
        return teacher is not None and teacher != self.proposed_label

    @property
    def is_unstable(self) -> bool:
        """The teacher gave different answers to the same text across runs."""
        return len(set(self.teacher_votes)) > 1

    @property
    def review_priority(self) -> tuple[int, int, float]:
        """Sort key: contested first, then unstable, then least confident."""
        return (
            0 if self.is_contested else 1,
            0 if self.is_unstable else 1,
            self.teacher_confidence if self.teacher_confidence is not None else 0.0,
        )


def load_candidates(*paths: Path) -> list[Candidate]:
    """Parse candidate JSONL files, rejecting duplicate ids across all of them."""
    candidates: list[Candidate] = []
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path.name}:{lineno}: expected a JSON object")
            try:
                candidates.append(Candidate.model_validate(raw))
            except ValidationError as exc:
                raise ValueError(f"{path.name}:{lineno}: {exc}") from exc

    duplicates = [item for item, n in Counter(c.id for c in candidates).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate candidate id(s) across inputs: {sorted(duplicates)}")
    return candidates


def load_teacher_votes(directory: Path = CANDIDATE_DIR) -> dict[str, tuple[int, ...]]:
    """Recorded teacher votes by row id, for every candidate that has them.

    Read from disk rather than re-requested: scoring the teacher, and simulating
    the escalate route of the cascade, cost nothing once the votes exist.
    """
    return {
        candidate.id: candidate.teacher_votes
        for candidate in load_candidates(*sorted(directory.glob("*.jsonl")))
        if candidate.teacher_votes
    }


def review_order(candidates: list[Candidate]) -> list[Candidate]:
    """Candidates ordered so the rows most worth a human's attention come first."""
    return sorted(candidates, key=lambda c: (c.review_priority, c.id))


def promote(
    candidate: Candidate,
    *,
    category: Category,
    reviewed_by: str,
    reviewed_on: date | None = None,
) -> Row:
    """Turn a reviewed candidate into a labelled row.

    ``category`` is what the reviewer decided, which may differ from what the
    author proposed — that is the point of review. The label is derived from it
    rather than passed separately, so a reviewer cannot produce the one thing
    ``dc.schema`` forbids: a label that contradicts its category.

    ``reviewed_by`` is required and cannot be blank. A row with no one's name on
    it is an unreviewed row, and unreviewed rows do not become training data.
    """
    if not reviewed_by.strip():
        raise ValueError("reviewed_by is required: an unreviewed row is not training data")
    return Row(
        id=candidate.id,
        origin_id=candidate.origin_id,
        feeling=candidate.feeling,
        free_text=candidate.free_text,
        label=1 if category is POSITIVE_CATEGORY else 0,
        category=category,
        provenance=candidate.provenance,
        reviewed_by=reviewed_by.strip(),
        reviewed_on=reviewed_on or date.today(),
    )
