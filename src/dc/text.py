"""Splitting a note into the pieces a safety check should look at separately.

The model scores a whole note, and ``all-MiniLM-L6-v2`` mean-pools over it. A
crisis clause inside a longer note about a puzzle is therefore *averaged away*:
"cleared the level finally, tries and tries. honestly i've been thinking about
ending it" scores 0.008 as one note, and 0.479 as its final clause alone.

So a note is split into segments and each is scored. Two kinds, because
punctuation cannot be relied on:

* **sentences**, on ordinary terminators and after a closing brace or tag, which
  handles injection payloads; and
* **sliding word windows**, which catch the same dilution written without any
  punctuation at all.

This is used by the red-team probe, and is the candidate rule for the skip band:
skip the LLM only when *every* segment scores below ``low``.
"""

from __future__ import annotations

import re

__all__ = ["WINDOW", "STRIDE", "split_segments"]

WINDOW = 8
STRIDE = 4

#: Sentence terminators, newlines, and the end of a JSON object, bracket or tag.
#: The last alternative is zero-width on purpose: markup packed tight against its
#: text ("</user><note>i want to end it") has no whitespace to split on, and a
#: red-team note used exactly that to keep a crisis clause inside one segment.
_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|\n+|(?<=[}\]>])\s*")


def split_segments(text: str, *, window: int = WINDOW, stride: int = STRIDE) -> list[str]:
    """The note, its sentences, and sliding word windows over it.

    The whole note is always included, so segment-level scoring can only ever see
    more than whole-note scoring, never less. Order is stable and duplicates are
    dropped, so the same note always yields the same segments.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    segments = [cleaned]
    segments.extend(part.strip() for part in _BOUNDARY.split(cleaned) if part.strip())

    words = cleaned.split()
    if len(words) > window:
        for start in range(0, len(words) - window + 1, stride):
            segments.append(" ".join(words[start : start + window]))
        tail = " ".join(words[-window:])
        segments.append(tail)

    seen: dict[str, None] = {}
    for segment in segments:
        seen.setdefault(segment, None)
    return list(seen)
