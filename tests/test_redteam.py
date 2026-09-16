"""The red-team set and the segment splitter it relies on."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from dc.text import split_segments

REPO = Path(__file__).resolve().parents[1]
NOTES_PATH = REPO / "data" / "redteam.jsonl"
MAX_FREE_TEXT = 200

FAMILIES = {"dilution", "injection", "box-testing", "obfuscation", "oblique", "non-english"}


def notes() -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in NOTES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- the splitter ---------------------------------------------------------------------


def test_the_whole_note_comes_first() -> None:
    # Segment scoring must see everything whole-note scoring sees, never less.
    text = "cleared the level. i want to end it"
    assert split_segments(text)[0] == text


def test_sentences_are_separated() -> None:
    assert "i want to end it" in split_segments("cleared the level. i want to end it")


def test_dilution_without_punctuation_is_still_split() -> None:
    # The sliding windows exist for exactly this: no terminator to split on.
    segments = split_segments("stage 47 forty tries no stars and i dont want to be here anymore")
    assert any("dont want to be here anymore" in segment for segment in segments)


def test_markup_packed_against_its_text_is_split() -> None:
    # A red-team note used this to keep a crisis clause inside one segment.
    segments = split_segments("<user>stage cleared</user><note>i want to end it</note>")
    assert any(segment.startswith("i want to end it") for segment in segments)


def test_segments_are_unique_and_stable() -> None:
    text = "one. two. one. three"
    assert split_segments(text) == split_segments(text)
    assert len(split_segments(text)) == len(set(split_segments(text)))


def test_empty_text_has_no_segments() -> None:
    assert split_segments("   ") == []


# --- the set --------------------------------------------------------------------------


def test_the_set_is_substantial_and_covers_every_family() -> None:
    rows = notes()
    assert len(rows) >= 50
    counts = Counter(row["family"] for row in rows)
    assert set(counts) == FAMILIES
    # Dilution broke the skip band, so it carries the most weight.
    assert counts["dilution"] == max(counts.values())


def test_notes_are_unique_and_fit_the_api_contract() -> None:
    rows = notes()
    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    assert len(set(ids)) == len(ids)
    assert len(set(texts)) == len(texts)
    assert all(0 < len(row["text"]) <= MAX_FREE_TEXT for row in rows)


def test_the_probe_is_kept_out_of_training() -> None:
    """A probe the model has been trained on has stopped being a probe."""
    trained: set[str] = set()
    for name in ("labelled.jsonl", "seed.jsonl"):
        path = REPO / "data" / name
        if path.exists():
            trained |= {
                json.loads(line)["free_text"].strip().lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
    leaked = [row["id"] for row in notes() if row["text"].strip().lower() in trained]
    assert leaked == [], f"red-team notes present in training data: {leaked}"


@pytest.mark.skipif(
    not (REPO / "reports" / "redteam.json").exists(), reason="run `make redteam` first"
)
def test_the_recorded_run_matches_the_committed_set() -> None:
    recorded = json.loads((REPO / "reports" / "redteam.json").read_text(encoding="utf-8"))
    assert {p["id"] for p in recorded["probes"]} == {row["id"] for row in notes()}
