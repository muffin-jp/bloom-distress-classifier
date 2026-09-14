"""The spine's structural guarantees, run end to end with a stub embedder."""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import make_row

from dc.artifact import is_servable, load
from dc.schema import Category, Feeling, Provenance, Row
from dc.selection import Config
from dc.splits import build_assignment
from dc.train import GateError, run

CONFIGS = [Config(1.0, "none", False), Config(1.0, "none", True)]


class SpyEmbedder:
    """Separable on the word "life", and records every text it is asked to embed."""

    dim = 2

    def __init__(self, *, informative: bool = True) -> None:
        self.seen: list[str] = []
        self.informative = informative

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        if not self.informative:
            return np.ones((len(texts), self.dim), dtype=np.float32)
        return np.array(
            # len(), not hash(): Python salts str hashes per process, which would
            # make this stub — and every test using it — differ between runs.
            [[1.0 if "life" in t else 0.0, (len(t) % 7) / 10] for t in texts],
            dtype=np.float32,
        )


def corpus(*, positive_text: str = "a hard life note") -> list[Row]:
    rows: list[Row] = [
        make_row(
            f"golden-{i}",
            category=Category.DISTRESS,
            feeling=Feeling.CUSTOM,
            free_text=f"golden {positive_text} {i}",
            provenance=Provenance.SEED,
        )
        for i in range(10)
    ]
    rows += [
        make_row(
            f"pos-{i}",
            category=Category.DISTRESS,
            feeling=Feeling.CUSTOM,
            free_text=f"{positive_text} {i}",
        )
        for i in range(40)
    ]
    rows += [
        make_row(
            f"neg-{i}",
            category=Category.GAME_FRUSTRATION,
            feeling=Feeling.FRUSTRATED,
            free_text=f"a tricky game stage {i}",
        )
        for i in range(80)
    ]
    return rows


def test_training_never_embeds_a_test_row(tmp_path: Path) -> None:
    """Tuning on test is impossible by construction, not avoided by discipline."""
    rows = corpus()
    assignment = build_assignment(rows)
    spy = SpyEmbedder()

    run(
        rows,
        assignment,
        spy,
        configs=CONFIGS,
        artifact_dir=tmp_path / "a",
        report_dir=tmp_path / "r",
    )

    test_texts = {row.free_text for row in rows if assignment[row.id] == "test"}
    train_texts = {row.free_text for row in rows if assignment[row.id] == "train"}
    assert test_texts, "fixture should hold some rows out"
    assert not test_texts & set(spy.seen)
    assert train_texts <= set(spy.seen)


def test_training_writes_an_artifact_that_is_not_yet_servable(tmp_path: Path) -> None:
    rows = corpus()
    run(
        rows,
        build_assignment(rows),
        SpyEmbedder(),
        configs=CONFIGS,
        artifact_dir=tmp_path / "a",
        report_dir=tmp_path / "r",
    )
    artifact = load(tmp_path / "a")
    assert artifact.metadata["thresholds"] is None
    assert not is_servable(artifact.metadata)
    assert not artifact.model.layout.include_feeling
    assert (tmp_path / "r" / "training.md").exists()
    assert (tmp_path / "r" / "training.json").exists()


def test_training_never_ships_the_feeling_shortcut(tmp_path: Path) -> None:
    # Feeling separates this fixture perfectly, so it scores best in CV.
    rows = corpus()
    summary = run(
        rows,
        build_assignment(rows),
        SpyEmbedder(),
        configs=CONFIGS,
        artifact_dir=tmp_path / "a",
        report_dir=tmp_path / "r",
    )
    assert not summary.selection.chosen.config.include_feeling
    assert summary.probe_config is not None and summary.probe_config.include_feeling


def test_training_stops_when_a_keyword_rule_would_do(tmp_path: Path) -> None:
    # Crisis words on every positive and an embedder that sees nothing: the rule
    # wins, and the honest output is the word list rather than a model.
    rows = corpus(positive_text="I want to die")
    with pytest.raises(GateError, match="does not beat the keyword"):
        run(
            rows,
            build_assignment(rows),
            SpyEmbedder(informative=False),
            configs=CONFIGS,
            artifact_dir=tmp_path / "a",
            report_dir=tmp_path / "r",
        )
    assert not (tmp_path / "a").exists()


def test_training_refuses_an_unsound_split(tmp_path: Path) -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    assignment["golden-0"] = "train"
    with pytest.raises(GateError, match="unsound"):
        run(
            rows,
            assignment,
            SpyEmbedder(),
            configs=CONFIGS,
            artifact_dir=tmp_path / "a",
            report_dir=tmp_path / "r",
        )
