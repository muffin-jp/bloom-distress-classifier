"""The loader's job is to reject bad data loudly. These are the ways it must."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import row_dict

from dc.schema import MAX_FREE_TEXT, Category, Provenance, Row, load_dataset, summarize


def write_jsonl(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_loads_a_valid_row(tmp_path: Path) -> None:
    rows = load_dataset(write_jsonl(tmp_path, [row_dict()]))
    assert len(rows) == 1
    assert rows[0].id == "nf-001"
    assert rows[0].label == 0


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row_dict()) + "\n\n\n", encoding="utf-8")
    assert len(load_dataset(path)) == 1


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [row_dict(freeText="typo'd key")])
    with pytest.raises(ValueError, match="rows.jsonl:1"):
        load_dataset(path)


def test_empty_free_text_is_rejected(tmp_path: Path) -> None:
    # Chip-only requests never reach the classifier, so a blank note is out of
    # distribution, not a valid negative example.
    with pytest.raises(ValueError):
        load_dataset(write_jsonl(tmp_path, [row_dict(free_text="")]))


def test_whitespace_only_free_text_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_dataset(write_jsonl(tmp_path, [row_dict(free_text="   ")]))


def test_over_length_free_text_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [row_dict(free_text="x" * (MAX_FREE_TEXT + 1))])
    with pytest.raises(ValueError):
        load_dataset(path)


def test_label_must_agree_with_category(tmp_path: Path) -> None:
    # A distress row labelled 0 is the single most damaging silent error in the
    # dataset: it teaches the model to miss the case that matters.
    path = write_jsonl(tmp_path, [row_dict(category="distress", label=0)])
    with pytest.raises(ValueError, match="contradicts category"):
        load_dataset(path)


def test_positive_label_outside_distress_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [row_dict(category="game-frustration", label=1)])
    with pytest.raises(ValueError, match="contradicts category"):
        load_dataset(path)


def test_unknown_category_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_dataset(write_jsonl(tmp_path, [row_dict(category="regeneration")]))


def test_unknown_feeling_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_dataset(write_jsonl(tmp_path, [row_dict(feeling="delighted")]))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [row_dict(), row_dict()])
    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset(path)


def test_dangling_origin_id_is_rejected(tmp_path: Path) -> None:
    # A paraphrase pointing at a row that is not present would be split as its
    # own group — which is exactly how leakage gets in.
    path = write_jsonl(tmp_path, [row_dict(id="nf-002", origin_id="nf-001")])
    with pytest.raises(ValueError, match="do not match any id"):
        load_dataset(path)


def test_resolvable_origin_id_is_accepted(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [row_dict(id="nf-001"), row_dict(id="nf-002", origin_id="nf-001")])
    rows = load_dataset(path)
    assert {row.origin_id for row in rows} == {"nf-001"}


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)


def test_error_names_the_offending_line(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [row_dict(id="a"), row_dict(id="b", free_text="")])
    with pytest.raises(ValueError, match="rows.jsonl:2"):
        load_dataset(path)


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row_dict()) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rows.jsonl:2: invalid JSON"):
        load_dataset(path)


def test_rows_are_frozen() -> None:
    row = Row.model_validate(row_dict())
    with pytest.raises(ValueError):
        row.label = 1  # type: ignore[misc]


def test_is_golden_tracks_provenance() -> None:
    assert Row.model_validate(row_dict(provenance="seed")).is_golden
    assert not Row.model_validate(row_dict(provenance="llm-augmented")).is_golden


def test_summarize_reports_class_balance(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            row_dict(id="a"),
            row_dict(id="b", category="distress", label=1, feeling="tired"),
            row_dict(id="c", origin_id="a"),
        ],
    )
    summary = summarize(load_dataset(path))
    assert summary["rows"] == 3
    assert summary["positive"] == 1
    assert summary["groups"] == 2
    assert summary["by_category"][Category.DISTRESS.value] == 1
    assert summary["by_provenance"][Provenance.HUMAN_WRITTEN.value] == 3
