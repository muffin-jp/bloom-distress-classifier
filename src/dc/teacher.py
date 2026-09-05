"""The teacher: the exact classifier this project is distilling.

``PRODUCTION_SYSTEM_PROMPT`` is copied **verbatim** from ``bloom-langgraph``'s
``app/prompts/distress.py``. It is duplicated rather than imported because the
two repos deploy independently, but a silent divergence would mean the benchmark
measures a classifier that is not the one in production — so
:func:`check_prompt_drift` compares the two whenever the companion repo is
checked out beside this one, and the test suite fails if they have drifted.

Two schemas, and the difference matters:

* :data:`PRODUCTION_SCHEMA` is byte-identical to production. **Baseline 4 — the
  number the student model has to beat — must use this one.** Changing what you
  ask for changes what you measure.
* :data:`CONFIDENCE_SCHEMA` adds a self-reported confidence. It is for ordering
  the human review queue only. That number is a verbalised guess, not a
  calibrated probability: use it to decide what to read first, never as a label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from anthropic.types import Message

__all__ = [
    "CONFIDENCE_SCHEMA",
    "PRODUCTION_SCHEMA",
    "PRODUCTION_SYSTEM_PROMPT",
    "TEACHER_MODEL",
    "build_user_message",
    "text_from",
    "check_prompt_drift",
    "parse_reply",
]

#: The model in production today. Named explicitly: this project distils *this*
#: classifier, so swapping it would invalidate every label and every benchmark.
TEACHER_MODEL = "claude-haiku-4-5"

PRODUCTION_SYSTEM_PROMPT = """You are a safety classifier for Bloom, a mobile puzzle game for mental wellness. After clearing a stage, players pick a feeling and can add a short note. Your only job is to decide whether the input signals serious emotional distress.

Distress = true when the input indicates: thoughts of self-harm or suicide, hopelessness about life, abuse, feeling unsafe, or an acute personal crisis.

Distress = false for ordinary game emotions, even strongly negative ones:
- "this stage is annoying", "I hate this level", "so tired of retrying" are NOT distress.
- The selected feelings (frustrated, disappointed, anxious, tired, etc.) are NOT distress on their own.
- Everyday venting about school, work, or a bad day is NOT distress by itself.

Only when a note is genuinely ambiguous between game talk and a real signal of serious suffering or danger, err on the side of distress = true.

Respond with JSON only: {"distress": true} or {"distress": false}"""

PRODUCTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"distress": {"type": "boolean"}},
    "required": ["distress"],
    "additionalProperties": False,
}

CONFIDENCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "distress": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["distress", "confidence"],
    "additionalProperties": False,
}


def build_user_message(feeling: str, free_text: str) -> str:
    """The user turn, byte-identical to the production wire format."""
    return f'Selected feeling: {feeling}\nPlayer note:\n"""{free_text}"""'


def parse_reply(text: str) -> tuple[int, float | None]:
    """Parse a teacher reply into ``(label, confidence)``.

    Raises ``ValueError`` on anything unparseable so the caller's retry can take
    another pass. Note the deliberate difference from production, which fails
    safe to ``distress=true``: an offline labelling run must not manufacture a
    positive out of a parse error and hand it to a reviewer as the model's
    opinion. Here, an unparseable reply is an error to retry, not a label.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if match is None:
        raise ValueError("teacher reply contained no JSON object")
    loaded: Any = json.loads(match.group(0))
    if not isinstance(loaded, dict):
        raise ValueError("teacher reply was not a JSON object")
    parsed = cast("dict[str, Any]", loaded)
    raw_label: Any = parsed.get("distress")
    if not isinstance(raw_label, bool):
        raise ValueError(f"teacher returned invalid distress: {raw_label!r}")
    raw_confidence: Any = parsed.get("confidence")
    confidence: float | None = None
    if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
        confidence = min(1.0, max(0.0, float(raw_confidence)))
    return (1 if raw_label else 0, confidence)


def text_from(response: Message) -> str:
    """Concatenate a response's text blocks.

    Comparing ``block.type`` narrows the content union to the text variant, so
    this stays type-safe without reaching for ``getattr``.
    """
    return "".join(block.text for block in response.content if block.type == "text")


#: Where the companion repo's copy lives, when it is checked out beside us.
COMPANION_PROMPT_PATH = (
    Path(__file__).resolve().parents[2].parent
    / "bloom-langgraph"
    / "app"
    / "prompts"
    / "distress.py"
)


def check_prompt_drift(path: Path = COMPANION_PROMPT_PATH) -> str | None:
    """Return a description of any drift, or ``None`` if in sync / not checked out.

    Compares the prompt body character-for-character. A drifted prompt does not
    raise here — the caller decides whether that is a warning or a failure —
    because the companion repo is legitimately absent in CI.
    """
    if not path.exists():
        return None
    source = path.read_text(encoding="utf-8")
    match = re.search(r'DISTRESS_SYSTEM_PROMPT = """(.*?)"""', source, re.DOTALL)
    if match is None:
        return f"{path.name}: could not find DISTRESS_SYSTEM_PROMPT to compare against"
    theirs = match.group(1)
    if theirs != PRODUCTION_SYSTEM_PROMPT:
        return (
            f"the teacher prompt has drifted from {path}. Every label and every "
            "benchmark in this repo describes the prompt in dc/teacher.py; if "
            "production changed, re-copy it and re-run the teacher."
        )
    return None
