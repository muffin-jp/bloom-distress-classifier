"""Every look at the test set is recorded, and a second look needs a reason."""

from __future__ import annotations

from pathlib import Path

import pytest

from dc.ledger import (
    LedgerError,
    Look,
    Render,
    append_look,
    append_render,
    authorize,
    files_sha256,
    read_looks,
    read_renders,
    uncommitted,
)


def look(number: int, reason: str = "first look") -> Look:
    return Look(number, "2026-09-14", "abc", "a" * 64, "d" * 64, "e" * 64, reason)


def test_the_first_look_needs_no_reason() -> None:
    assert authorize([], again=None) == "first look"


def test_a_second_look_is_refused_without_a_reason() -> None:
    with pytest.raises(LedgerError, match="already been evaluated 1 time"):
        authorize([look(1)], again=None)


def test_the_refusal_points_at_rerendering_first() -> None:
    with pytest.raises(LedgerError, match="--render"):
        authorize([look(1)], again=None)


def test_a_blank_reason_is_not_a_reason() -> None:
    with pytest.raises(LedgerError):
        authorize([look(1)], again="   ")


def test_a_second_look_with_a_reason_is_allowed_and_recorded() -> None:
    assert authorize([look(1)], again="fixed a bug in the scorer") == "fixed a bug in the scorer"


def test_looks_append_in_order(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    append_look(look(1), path)
    append_look(look(2, "second"), path)
    assert [entry.reason for entry in read_looks(path)] == ["first look", "second"]


def test_a_look_cannot_skip_or_rewrite_history(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    append_look(look(1), path)
    with pytest.raises(LedgerError, match="does not follow"):
        append_look(look(1), path)
    with pytest.raises(LedgerError, match="does not follow"):
        append_look(look(3), path)


def test_no_ledger_means_no_looks(tmp_path: Path) -> None:
    assert read_looks(tmp_path / "absent.json") == []


def test_file_hash_tracks_content(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    before = files_sha256([a])
    a.write_text("x = 2\n")
    assert files_sha256([a]) != before


def test_unknown_git_state_counts_as_uncommitted(tmp_path: Path) -> None:
    assert uncommitted(["data"], repo=tmp_path) == ["<git status unavailable>"]


def render(note: str = "re-rendered") -> Render:
    return Render("2026-09-14", "p" * 64, "a" * 64, "f" * 64, note)


def test_a_render_is_recorded_without_counting_as_a_look(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    append_look(look(1), path)
    append_render(render("added confidence bounds"), path)
    assert len(read_looks(path)) == 1
    assert [r.note for r in read_renders(path)] == ["added confidence bounds"]


def test_looks_and_renders_never_overwrite_each_other(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    append_look(look(1), path)
    append_render(render(), path)
    append_look(look(2, "second"), path)
    append_render(render("again"), path)
    assert len(read_looks(path)) == 2
    assert len(read_renders(path)) == 2


def test_nothing_can_be_rerendered_before_the_test_set_is_scored(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="has not been scored"):
        append_render(render(), tmp_path / "ledger.json")
