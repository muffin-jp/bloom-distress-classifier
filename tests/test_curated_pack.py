"""Guards on the hand-written taxonomy.

The curated pack is not just data — it is the argument that this dataset is hard
enough to be worth measuring on. These tests protect that argument: if the
matched pairs stop being matched, or the hard class stops being hard, the
numbers downstream become easy in a way nobody would notice.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from dc.candidates import load_candidates
from dc.schema import Category, load_dataset
from dc.splits import DATA_DIR

CURATED = DATA_DIR / "candidates" / "curated.jsonl"

pytestmark = pytest.mark.skipif(not CURATED.exists(), reason="curated pack not present")

MODE_RE = re.compile(r"^mode:(?P<mode>[a-z0-9-]+) frame:(?P<frame>game|life|adversarial|noise)$")


def test_pack_loads_and_is_all_human_written() -> None:
    candidates = load_candidates(CURATED)
    assert len(candidates) >= 90
    assert all(c.provenance.value == "human-written" for c in candidates)


def test_every_candidate_declares_a_parseable_mode() -> None:
    for candidate in load_candidates(CURATED):
        assert candidate.note is not None, candidate.id
        assert MODE_RE.match(candidate.note), f"{candidate.id}: bad note {candidate.note!r}"


def test_the_pack_is_candidates_not_training_data() -> None:
    """The one-way door, enforced at the file level.

    A candidate file must not be loadable as a dataset: it carries proposed
    labels, and the only way to a real label is review.
    """
    with pytest.raises(ValueError):
        load_dataset(CURATED)


def test_teacher_votes_are_all_or_nothing() -> None:
    # propose_labels writes votes back into the pack, so votes here are expected
    # once it has run. What must not happen is a half-finished pass leaving some
    # rows scored and others not — that would silently reorder the review queue
    # by how far a crashed run got.
    counts = {len(c.teacher_votes) for c in load_candidates(CURATED)}
    assert len(counts) == 1, f"mixed vote counts across the pack: {sorted(counts)}"


def test_matched_pairs_are_actually_matched() -> None:
    """Each pair mode must have exactly one game frame and one life frame."""
    frames: dict[str, set[str]] = {}
    for candidate in load_candidates(CURATED):
        if not candidate.id.startswith("cur-pair-"):
            continue
        match = MODE_RE.match(candidate.note or "")
        assert match is not None
        frames.setdefault(match.group("mode"), set()).add(match.group("frame"))
    assert frames, "no matched pairs found"
    for mode, seen in sorted(frames.items()):
        assert seen == {"game", "life"}, f"mode {mode!r} is not a matched pair: {sorted(seen)}"


def test_matched_pairs_sit_on_opposite_sides_of_the_label() -> None:
    by_id = {c.id: c for c in load_candidates(CURATED)}
    pair_ids = sorted({cid.rsplit("-", 1)[0] for cid in by_id if cid.startswith("cur-pair-")})
    assert len(pair_ids) >= 15, "the taxonomy should cover at least 15 confusion modes"
    for stem in pair_ids:
        game, life = by_id[f"{stem}-game"], by_id[f"{stem}-life"]
        assert game.proposed_label == 0, stem
        assert life.proposed_label == 1, stem
        assert game.category is Category.GAME_FRUSTRATION
        assert life.category is Category.DISTRESS


def test_pairs_do_not_share_an_origin_id() -> None:
    # They share a *mode*, not a source sentence. Sharing an origin would force
    # both frames onto the same side of the split, which is the opposite of what
    # the pairs are for.
    for candidate in load_candidates(CURATED):
        assert candidate.origin_id == candidate.id


def test_the_hard_class_is_well_represented() -> None:
    counts = Counter(c.category for c in load_candidates(CURATED))
    assert counts[Category.GAME_FRUSTRATION] >= counts[Category.DISTRESS]
    assert counts[Category.DISTRESS] >= 20


def test_every_category_appears() -> None:
    counts = Counter(c.category for c in load_candidates(CURATED))
    missing = [category.value for category in Category if counts[category] == 0]
    assert missing == [], f"curated pack has no examples of {missing}"


def test_no_duplicate_free_text() -> None:
    texts = [c.free_text.strip().lower() for c in load_candidates(CURATED)]
    duplicates = [text for text, n in Counter(texts).items() if n > 1]
    assert duplicates == [], f"duplicated example text: {duplicates}"
