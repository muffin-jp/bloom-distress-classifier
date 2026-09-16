"""Scoring a note the way the service scores it: segments in, one decision out.

This is the seam the red-team probe, the test evaluation and ``bloom-langgraph``
all sit behind, so all three exercise the same rule. Before this module the probe
had its own copy of the batching logic, which meant the thing being validated was
not quite the thing being served.

One note becomes several texts
------------------------------
:func:`score_notes` flattens every note's segments into one list, embeds that list
in a single call, and folds the scores back. The embedder is the expensive part
and it batches well, so a note costs one call rather than one per segment.

Two statistics come out of it, and they are used for different decisions:

* ``whole`` — the note scored as one string, which is what ``high`` was fitted on
  and the only thing that may route a note to **support**;
* ``worst`` — the highest score over all segments, which is what ``low`` is fitted
  on and the only thing that may let a note **skip** the LLM.

A note with no scorable text yields neither, and escalates.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dc.cascade import Route, Thresholds, route_note
from dc.features import Embedder
from dc.model import DistressModel
from dc.text import SEGMENTATION, Segmentation

__all__ = ["NoteScore", "flatten_segments", "score_notes", "segment_scores"]


@dataclass(frozen=True)
class NoteScore:
    """What the model says about one note, whole and in pieces."""

    #: The note scored as a single string. NaN when the note has no scorable text.
    whole: float
    #: The highest score over every segment, the whole note included, so
    #: ``worst >= whole`` always holds. NaN when the note has no scorable text.
    worst: float
    #: The segment that scored ``worst`` — what to show a reviewer asking why.
    worst_segment: str
    n_segments: int

    def route(self, thresholds: Thresholds) -> Route:
        return route_note(self.whole, self.worst, thresholds)

    @property
    def scorable(self) -> bool:
        return math.isfinite(self.whole) and math.isfinite(self.worst)


_UNSCORABLE = NoteScore(whole=math.nan, worst=math.nan, worst_segment="", n_segments=0)


def flatten_segments(
    notes: Sequence[str], *, segmentation: Segmentation = SEGMENTATION
) -> tuple[list[str], np.ndarray]:
    """Every note's segments in one list, plus the note index each came from.

    One flat list is what an embedder wants; the owner array is what puts the
    scores back where they belong. Training and serving both need this, and a
    note with no scorable text simply contributes nothing.
    """
    texts: list[str] = []
    owners: list[int] = []
    for index, note in enumerate(notes):
        for segment in segmentation.split(note):
            texts.append(segment)
            owners.append(index)
    return texts, np.asarray(owners, dtype=int)


def segment_scores(texts: Sequence[str], model: DistressModel, embedder: Embedder) -> np.ndarray:
    """Score a flat list of texts in one embedding call."""
    if not texts:
        return np.zeros(0, dtype=float)
    if model.layout.include_feeling:
        raise ValueError(
            "segment scoring is undefined for a model that reads the feeling chip: a "
            "sentence inside a note has no chip of its own"
        )
    features = np.asarray(embedder.embed(list(texts)), dtype=float)
    return model.predict_proba_features(features)


def score_notes(
    notes: Sequence[str],
    model: DistressModel,
    embedder: Embedder,
    *,
    segmentation: Segmentation = SEGMENTATION,
) -> list[NoteScore]:
    """Whole-note and worst-segment scores for each note, in one batch."""
    per_note = [segmentation.split(note) for note in notes]
    flat = [segment for segments in per_note for segment in segments]
    scores = segment_scores(flat, model, embedder)

    out: list[NoteScore] = []
    cursor = 0
    for segments in per_note:
        if not segments:
            out.append(_UNSCORABLE)
            continue
        window = scores[cursor : cursor + len(segments)]
        cursor += len(segments)
        worst = int(np.argmax(window))
        out.append(
            NoteScore(
                whole=float(window[0]),  # Segmentation.split yields the whole note first
                worst=float(window[worst]),
                worst_segment=segments[worst],
                n_segments=len(segments),
            )
        )
    return out
