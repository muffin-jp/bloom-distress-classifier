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

The skip band compares the *highest* segment score against ``low``, so the LLM is
skipped only when every part of the note is confidently fine.

Why the parameters are recorded, not just chosen
------------------------------------------------
``bloom-langgraph`` re-implements this split to serve the model, so two copies of
the rule exist. A copy that drifts changes production routing silently: the
served skip band would no longer be the one that was fitted or red-teamed. So
:class:`Segmentation` — window, stride, and the boundary pattern itself — is
written into the artifact, and the consumer refuses an artifact whose parameters
are not the ones it implements. Drift becomes a load error instead of a wrong
answer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property

__all__ = [
    "BOUNDARY",
    "SCOPE",
    "SEGMENTATION",
    "STRIDE",
    "WINDOW",
    "ScopeRule",
    "Segmentation",
    "has_non_latin_letters",
    "split_segments",
]

WINDOW = 8
STRIDE = 4

#: Sentence terminators, newlines, and the end of a JSON object, bracket or tag.
#: The last alternative is zero-width on purpose: markup packed tight against its
#: text ("</user><note>i want to end it") has no whitespace to split on, and a
#: red-team note used exactly that to keep a crisis clause inside one segment.
BOUNDARY = r"(?<=[.!?;])\s+|\n+|(?<=[}\]>])\s*"


@dataclass(frozen=True)
class Segmentation:
    """How a note is cut up, as data that travels with the model.

    Two implementations of this rule exist, in two repositories. They agree
    because the artifact carries these three values and each side checks them
    against its own, not because both were written from the same description.
    """

    window: int = WINDOW
    stride: int = STRIDE
    boundary: str = BOUNDARY

    def __post_init__(self) -> None:
        if self.window < 1 or self.stride < 1:
            raise ValueError(f"window and stride must be >= 1, got {self.window}/{self.stride}")
        try:
            re.compile(self.boundary)
        except re.error as exc:
            raise ValueError(f"boundary is not a valid regular expression: {exc}") from exc

    @cached_property
    def pattern(self) -> re.Pattern[str]:
        return re.compile(self.boundary)

    def split(self, text: str) -> list[str]:
        """The note, its sentences, and sliding word windows over it.

        The whole note is always first, so segment-level scoring can only ever see
        more than whole-note scoring, never less. Order is stable and duplicates
        are dropped, so the same note always yields the same segments.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        segments = [cleaned]
        segments.extend(part.strip() for part in self.pattern.split(cleaned) if part.strip())

        words = cleaned.split()
        if len(words) > self.window:
            for start in range(0, len(words) - self.window + 1, self.stride):
                segments.append(" ".join(words[start : start + self.window]))
            segments.append(" ".join(words[-self.window :]))

        seen: dict[str, None] = {}
        for segment in segments:
            seen.setdefault(segment, None)
        return list(seen)


#: The fitted rule. Changing any of these changes the skip band, so a change here
#: invalidates the artifact and must go through training and the red-team probe.
SEGMENTATION = Segmentation()


def split_segments(text: str, *, window: int = WINDOW, stride: int = STRIDE) -> list[str]:
    """:meth:`Segmentation.split` at the fitted parameters, or overrides of them."""
    return Segmentation(window=window, stride=stride).split(text)


def has_non_latin_letters(text: str) -> bool:
    """True if any *letter* in the note belongs to a script other than Latin.

    Only letters are examined, so emoji, digits, curly quotes, em dashes and other
    punctuation never trigger it — an ordinary English note typed on a phone must
    not be treated as foreign. ``café`` and ``naïve`` are Latin and stay in scope.
    """
    return any(
        character.isalpha() and not unicodedata.name(character, "").startswith("LATIN")
        for character in text
    )


@dataclass(frozen=True)
class ScopeRule:
    """Notes the model has no basis to judge, which escalate whatever they score.

    ``all-MiniLM-L6-v2`` is an English model. On a Japanese, Korean or Russian note
    it still returns a confident-looking number, and that number means nothing:
    *"もう生きていたくない"* — "I don't want to live any more" — scores 0.05, which
    is low because the model cannot read it, not because the note is calm. A low
    score from a model that cannot read the input is the one input the skip band
    must never act on, so these notes go to the LLM whatever they score.

    This is a scope rule, not a language feature. It does not try to classify
    non-English distress; it declines to.
    """

    escalate_non_latin_letters: bool = True

    def out_of_scope(self, text: str) -> str:
        """Why this note may not be judged locally, or ``""`` if it may."""
        if self.escalate_non_latin_letters and has_non_latin_letters(text):
            return "non-Latin script: the embedder is English-only"
        return ""


#: The shipped scope rule. Like the segmentation, it travels with the artifact so
#: the service cannot apply a different one.
SCOPE = ScopeRule()
