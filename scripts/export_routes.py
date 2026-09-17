"""Freeze the routes this artifact gives, so a second implementation can be checked.

``bloom-langgraph`` serves this model, which means it re-implements the splitting
and scope rules in its own code. Two copies of one rule is the arrangement most
likely to go quietly wrong: a regex tweak on either side changes production
routing with no error anywhere, and both sides keep producing perfectly ordinary
scores.

So the routes are exported as data. This script writes every case's whole-note
score, worst-segment score, and route, together with the parameters that produced
them. The consumer commits the file, re-derives each route with its own
implementation, and fails its own test suite if any of them differs.

The cases are the adversarial probe — the notes most likely to expose a
difference in how a note is cut up — plus hand-written edge cases that pin the
parts of the rule the probe does not reach: empty text, packed markup, window
boundaries, accented Latin, emoji, and scripts the model declines to read. **No
test-set row is exported**, so the file can be copied into another repository
without carrying the release gate's answers with it.

    uv run --extra embed python scripts/export_routes.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dc.artifact import ARTIFACT_DIR, load, read_scope, read_segmentation
from dc.cascade import Thresholds
from dc.features import load_embedder
from dc.probe import load_probe_notes
from dc.serve import score_notes

ROUTES_PATH = ARTIFACT_DIR / "routes.json"

#: Cases the probe does not reach, each pinning one part of the rule. The comment
#: on each is what a reader needs to know if it starts failing.
EDGE_CASES: tuple[tuple[str, str], ...] = (
    ("edge-empty", ""),
    ("edge-blank", "   "),
    ("edge-one-word", "fine"),
    # The zero-width boundary after a closing tag: no whitespace to split on.
    ("edge-packed-markup", "</user><note>i want to end it"),
    ("edge-json-payload", '{"mood": "fine"} actually i want to die'),
    # Window boundaries: 8 words exactly, then 9, so the sliding window turns on.
    ("edge-eight-words", "one two three four five six seven eight"),
    ("edge-nine-words", "one two three four five six seven eight nine"),
    ("edge-newlines", "cleared it\n\nbut i can't keep going"),
    ("edge-semicolons", "level done; i don't want to be here; anyway"),
    # Latin script with accents and marks stays in scope.
    ("edge-accents", "café au lait and naïve résumé, feeling fine"),
    ("edge-emoji-quotes", "don't — really… 😊 that was a good one"),
    # Non-Latin letters: the scope rule escalates whatever the score is.
    ("edge-japanese", "もう生きていたくない"),
    ("edge-korean", "나 이제 못 버티겠어"),
    ("edge-cyrillic", "больше не могу"),
    ("edge-mixed-script", "stage 12 done. 다 끝내고 싶어"),
    # Latin-script non-English: NOT covered by the scope rule, on purpose.
    ("edge-german", "ich will nicht mehr leben"),
    ("edge-repeated-words", "tries and tries and tries and tries and tries and tries"),
    ("edge-trailing-space", "  i can't do this anymore  "),
    ("edge-long-ordinary", "cleared the level on the third go after about forty tries today"),
)


def main() -> None:
    artifact = load()
    segmentation = read_segmentation(artifact.metadata)
    scope = read_scope(artifact.metadata)
    bands = artifact.metadata["thresholds"]
    thresholds = Thresholds(float(bands["low"]), float(bands["high"]))

    cases = [(note.id, note.text) for note in load_probe_notes()] + list(EDGE_CASES)
    scored = score_notes(
        [text for _, text in cases],
        artifact.model,
        load_embedder(),
        segmentation=segmentation,
        scope=scope,
    )

    payload: dict[str, Any] = {
        "note": (
            "Routes frozen from bloom-distress-classifier. A consumer re-derives each "
            "route with its own implementation and fails if any differs. Regenerate with "
            "`make export-routes` after every training run."
        ),
        "thresholds": {"low": thresholds.low, "high": thresholds.high},
        "segmentation": {
            "window": segmentation.window,
            "stride": segmentation.stride,
            "boundary": segmentation.boundary,
        },
        "scope": {"escalate_non_latin_letters": scope.escalate_non_latin_letters},
        "cases": [
            {
                "id": case_id,
                "text": text,
                "segments": segmentation.split(text),
                "whole": None if not note.scorable else note.whole,
                "worst": None if not note.scorable else note.worst,
                "worst_segment": note.worst_segment,
                "out_of_scope": note.out_of_scope,
                "route": note.route(thresholds).value,
            }
            for (case_id, text), note in zip(cases, scored, strict=True)
        ],
    }
    ROUTES_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    routes = [case["route"] for case in payload["cases"]]
    print(f"Wrote {len(routes)} frozen routes to {ROUTES_PATH.relative_to(Path.cwd())}.")
    for name in ("skip-llm", "escalate", "support"):
        print(f"  {name:10s} {routes.count(name)}")


if __name__ == "__main__":
    main()
