"""Shared row factory.

Tests build rows explicitly rather than reading data/, so a failing test points
at a rule in the code and not at whatever happens to be in the dataset today.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from dc.schema import Category, Feeling, Provenance, Row


def make_row(
    row_id: str,
    *,
    origin_id: str | None = None,
    category: Category = Category.NORMAL_FEELING,
    feeling: Feeling = Feeling.PROUD,
    free_text: str = "cleared it on the third try",
    provenance: Provenance = Provenance.HUMAN_WRITTEN,
) -> Row:
    return Row(
        id=row_id,
        origin_id=origin_id or row_id,
        feeling=feeling,
        free_text=free_text,
        label=1 if category is Category.DISTRESS else 0,
        category=category,
        provenance=provenance,
        reviewed_by="tester",
        reviewed_on=date(2026, 9, 2),
    )


def row_dict(**overrides: Any) -> dict[str, Any]:
    """A valid raw row as it appears in JSONL, with fields overridden.

    Overriding ``id`` alone moves ``origin_id`` with it, so the default row is a
    self-origin row. Pass ``origin_id`` explicitly to build a paraphrase.
    """
    base: dict[str, Any] = {
        "id": "nf-001",
        "origin_id": "nf-001",
        "feeling": "proud",
        "free_text": "cleared it on the third try",
        "label": 0,
        "category": "normal-feeling",
        "provenance": "human-written",
        "reviewed_by": "tester",
        "reviewed_on": "2026-09-02",
    }
    if "id" in overrides and "origin_id" not in overrides:
        base["origin_id"] = overrides["id"]
    base.update(overrides)
    return base
