"""Credential loading: precedence, and what a missing key does."""

from __future__ import annotations

from pathlib import Path

import pytest

from dc.env import load_env


def test_env_file_is_loaded_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")

    load_env(env)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"


def test_an_exported_variable_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A key set for one command must not be silently replaced by a stale file.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "exported")
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")

    load_env(env)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "exported"


def test_a_missing_key_warns_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Not proof there are no credentials: an `ant auth login` profile also works,
    # and the SDK finds it without any env var. So this must not be fatal.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    load_env(tmp_path / "absent.env")

    assert "ant auth login" in capsys.readouterr().out


def test_an_auth_token_alone_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")

    load_env(tmp_path / "absent.env")

    assert capsys.readouterr().out == ""
