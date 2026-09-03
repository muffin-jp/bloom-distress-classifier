"""Row schema and strict loader for the labelled dataset.

The dataset is this project's audit surface — every number in the results table
is a claim about these rows — so the loader is strict on purpose. A malformed,
mis-tagged, or out-of-distribution row fails the load loudly rather than
slipping into training and quietly moving a metric.

This mirrors the discipline of ``bloom-langgraph``'s ``app/rag/build_index.py``:
validate at the boundary, report ``file:lineno``, and refuse to guess.

Enforced invariants
-------------------
1. No unknown fields. A typo'd key is an error, not a silently ignored one.
2. ``free_text`` is non-empty after stripping and at most 200 characters —
   the API contract, and the only input distribution the classifier ever sees
   (chip-only requests skip the classifier entirely; BUILD_SPEC.md §1).
3. ``label`` is 0 or 1.
4. ``label == 1`` **iff** ``category == "distress"``. The two fields are
   deliberately redundant so that a mislabelled row is a *contradiction* the
   loader can catch, rather than a plausible-looking mistake. If a row needs a
   positive label under another category, that is a spec change (§4.1), not a
   data edit.
5. ``id`` is unique across the file.
6. ``origin_id`` resolves to some ``id`` in the file (a row that is its own
   origin points at itself). A dangling origin is how paraphrase groups silently
   split apart, which is how leakage gets in (§4.6).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "MAX_FREE_TEXT",
    "Category",
    "Feeling",
    "POSITIVE_CATEGORY",
    "Provenance",
    "Row",
    "load_dataset",
    "summarize",
]

# Mirrors the ``freeText`` ceiling in bloom-langgraph's app/schemas.py. Training
# on longer text than production can produce would measure a distribution the
# model never sees.
MAX_FREE_TEXT = 200


class Feeling(StrEnum):
    """The seven chips a player can pick. Mirrors bloom-langgraph app/config.py."""

    PROUD = "proud"
    RELIEVED = "relieved"
    FRUSTRATED = "frustrated"
    DISAPPOINTED = "disappointed"
    ANXIOUS = "anxious"
    TIRED = "tired"
    CUSTOM = "custom"


class Category(StrEnum):
    """Stratification key. Carries the error analysis, not just the split.

    ``GAME_FRUSTRATION`` is the hard class: violent, hyperbolic writing about a
    puzzle that is lexically near-identical to genuine distress (§4.3).
    """

    DISTRESS = "distress"
    GAME_FRUSTRATION = "game-frustration"
    NORMAL_FEELING = "normal-feeling"
    MIXED_FEELING = "mixed-feeling"
    INJECTION = "injection"
    NONSENSE = "nonsense"


class Provenance(StrEnum):
    """Where a row came from. Reported as a share in the model card (§4.4)."""

    SEED = "seed"
    HUMAN_WRITTEN = "human-written"
    LLM_AUGMENTED = "llm-augmented"


#: The one category that carries a positive label. See invariant 4.
POSITIVE_CATEGORY = Category.DISTRESS


class Row(BaseModel):
    """One labelled example.

    Frozen: rows are read, split, and embedded, never edited in place. Mutating a
    row after the split is assigned would desynchronise it from splits.json.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    #: Grouping key for the split. Paraphrases share their source row's id so
    #: that a paraphrase can never land on the other side of the train/test line.
    origin_id: str = Field(min_length=1)
    feeling: Feeling
    free_text: str = Field(min_length=1, max_length=MAX_FREE_TEXT)
    label: int = Field(ge=0, le=1)
    category: Category
    provenance: Provenance
    #: Every row is human-reviewed, including the synthetic ones (§4.4).
    reviewed_by: str = Field(min_length=1)
    reviewed_on: date

    @model_validator(mode="after")
    def _check_label_matches_category(self) -> Row:
        expected = 1 if self.category is POSITIVE_CATEGORY else 0
        if self.label != expected:
            raise ValueError(
                f"label {self.label} contradicts category {self.category.value!r} "
                f"(expected {expected}): see invariant 4 in dc.schema"
            )
        return self

    @model_validator(mode="after")
    def _check_free_text_not_blank(self) -> Row:
        if not self.free_text.strip():
            raise ValueError("free_text is whitespace-only")
        return self

    @property
    def is_golden(self) -> bool:
        """True for the imported release-gate cases, which are test-only (§4.1)."""
        return self.provenance is Provenance.SEED


def load_dataset(path: Path) -> list[Row]:
    """Parse and validate a JSONL dataset, or raise ``ValueError``.

    Every failure names ``file:lineno`` so a bad row is one grep away. The
    cross-row invariants (unique ids, resolvable origins) are checked after the
    per-row pass, since they need the whole file.
    """
    rows: list[Row] = []
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
            rows.append(Row.model_validate(raw))
        except ValidationError as exc:
            raise ValueError(f"{path.name}:{lineno}: {exc}") from exc

    if not rows:
        raise ValueError(f"{path.name}: dataset is empty")

    ids = {row.id for row in rows}
    duplicates = [item for item, n in Counter(row.id for row in rows).items() if n > 1]
    if duplicates:
        raise ValueError(f"{path.name}: duplicate id(s) {sorted(duplicates)}")

    dangling = sorted({row.origin_id for row in rows if row.origin_id not in ids})
    if dangling:
        raise ValueError(
            f"{path.name}: origin_id(s) {dangling} do not match any id in the file. "
            "A paraphrase must point at a row that is present, or the group splits "
            "apart and leaks across the train/test line."
        )
    return rows


def summarize(rows: list[Row]) -> dict[str, Any]:
    """Counts a reviewer wants before trusting any metric computed on these rows."""
    positive = sum(row.label for row in rows)
    return {
        "rows": len(rows),
        "positive": positive,
        "positive_rate": positive / len(rows) if rows else 0.0,
        "groups": len({row.origin_id for row in rows}),
        "by_category": dict(sorted(Counter(row.category.value for row in rows).items())),
        "by_provenance": dict(sorted(Counter(row.provenance.value for row in rows).items())),
        "by_feeling": dict(sorted(Counter(row.feeling.value for row in rows).items())),
    }
