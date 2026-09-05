"""The teacher prompt is the definition of what is being distilled."""

from __future__ import annotations

import pytest

from dc.teacher import (
    CONFIDENCE_SCHEMA,
    PRODUCTION_SCHEMA,
    PRODUCTION_SYSTEM_PROMPT,
    build_user_message,
    check_prompt_drift,
    parse_reply,
)


def test_prompt_has_not_drifted_from_production() -> None:
    # Skips silently when the companion repo is not checked out (CI).
    drift = check_prompt_drift()
    assert drift is None, drift


def test_production_schema_is_exactly_the_production_shape() -> None:
    # Baseline 4 is only a fair ceiling if we ask production's question. Adding a
    # field would change the answer distribution and quietly move the benchmark.
    assert PRODUCTION_SCHEMA == {
        "type": "object",
        "properties": {"distress": {"type": "boolean"}},
        "required": ["distress"],
        "additionalProperties": False,
    }


def test_confidence_schema_only_adds_confidence() -> None:
    assert set(CONFIDENCE_SCHEMA["properties"]) == {"distress", "confidence"}  # type: ignore[arg-type]


def test_user_message_matches_the_production_wire_format() -> None:
    assert (
        build_user_message("tired", "hello") == 'Selected feeling: tired\nPlayer note:\n"""hello"""'
    )


def test_parse_reply_reads_label_and_confidence() -> None:
    assert parse_reply('{"distress": true, "confidence": 0.9}') == (1, 0.9)
    assert parse_reply('{"distress": false, "confidence": 0.1}') == (0, 0.1)


def test_parse_reply_tolerates_surrounding_prose() -> None:
    assert parse_reply('Sure!\n```json\n{"distress": true}\n```') == (1, None)


def test_parse_reply_clamps_confidence() -> None:
    assert parse_reply('{"distress": true, "confidence": 4}') == (1, 1.0)


def test_parse_reply_rejects_a_missing_label() -> None:
    with pytest.raises(ValueError, match="invalid distress"):
        parse_reply('{"confidence": 0.5}')


def test_parse_reply_rejects_a_non_boolean_label() -> None:
    with pytest.raises(ValueError, match="invalid distress"):
        parse_reply('{"distress": "yes"}')


def test_parse_reply_rejects_prose_with_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_reply("I think this one is distress.")


def test_parse_reply_does_not_fail_safe_to_positive() -> None:
    # Production fails safe to distress=true on an unparseable reply. An offline
    # labelling run must not: manufacturing a positive and showing it to a
    # reviewer as "the model's opinion" is how a parse bug becomes ground truth.
    with pytest.raises(ValueError):
        parse_reply("")


def test_prompt_still_names_the_hard_negatives() -> None:
    # If these disappear from production's prompt, game-frustration behaviour has
    # changed and the whole game-vs-life taxonomy needs revisiting.
    assert "this stage is annoying" in PRODUCTION_SYSTEM_PROMPT
    assert "err on the side of distress = true" in PRODUCTION_SYSTEM_PROMPT
