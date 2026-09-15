"""Explanations: what they read, what they record, and what they imitate."""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from conftest import make_row

from dc.cascade import Thresholds
from dc.explain import DecisionRecord, Explainer
from dc.features import EMBED_MODEL, EMBED_MODEL_REVISION
from dc.model import DistressModel, FeatureLayout
from dc.schema import Category, Row

BANDS = Thresholds(0.1, 0.8)
WORDS = ("alpha", "lately", "stage", "don't", "tired", "level", "nothing", "friend")


class WordEmbedder:
    """One dimension per word, normalised — separable and fully offline."""

    dim = len(WORDS)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.array([[1.0 if w in t else 0.0 for w in WORDS] for t in texts]) + 1e-3
        return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True, eq=False)
class CountingModel(DistressModel):
    calls: ClassVar[list[int]] = []

    def predict_proba_features(self, x: np.ndarray) -> np.ndarray:
        CountingModel.calls.append(len(x))
        return super().predict_proba_features(x)


def model(weight_on: str = "alpha") -> DistressModel:
    layout = FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, len(WORDS), False)
    coef = np.array([6.0 if w == weight_on else 0.0 for w in WORDS])
    return CountingModel(coef=coef, intercept=-2.0, layout=layout)


def train_rows() -> list[Row]:
    """Labels deliberately unrelated to "alpha": the model and the labels disagree."""
    rows: list[Row] = []
    for i in range(40):
        text = f"note {i} {'alpha' if i % 2 else 'stage'} {'lately' if i % 3 == 0 else 'level'}"
        category = Category.DISTRESS if i % 4 == 0 else Category.GAME_FRUSTRATION
        rows.append(make_row(f"t{i}", category=category, free_text=text))
    return rows


def explainer(**kwargs: object) -> Explainer:
    return Explainer(
        train_rows(),
        model(),
        WordEmbedder(),
        BANDS,
        artifact_sha256="a" * 64,
        **kwargs,  # type: ignore[arg-type]
    )


def test_neighbours_come_only_from_training_rows() -> None:
    explanation = explainer().explain("a brand new note about alpha")
    train_ids = {r.id for r in train_rows()}
    assert {n.id for n in explanation.neighbours} <= train_ids
    assert len(explanation.neighbours) == 5


def test_neighbours_are_ordered_by_similarity() -> None:
    sims = [n.similarity for n in explainer().explain("alpha lately").neighbours]
    assert sims == sorted(sims, reverse=True)


def test_the_decision_record_carries_a_hash_not_the_text() -> None:
    text = "i don't have it in me tonight"
    record = explainer().explain(text).record
    assert record.text_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert text not in repr(record)
    assert set(DecisionRecord.__dataclass_fields__) >= {"score", "route", "llm_consulted"}


def test_a_recorded_score_is_explained_without_rescoring() -> None:
    # Test errors are explained from the saved predictions, never re-scored.
    built = explainer()
    CountingModel.calls.clear()
    explanation = built.explain("alpha", recorded_score=0.123, recorded_route="support")
    assert CountingModel.calls == []
    assert (explanation.record.score, explanation.record.route) == (0.123, "support")


def test_an_unrecorded_note_is_scored_by_the_model() -> None:
    built = explainer()
    CountingModel.calls.clear()
    built.explain("alpha")
    assert CountingModel.calls == [1]


def test_the_llm_is_consulted_only_on_escalation() -> None:
    built = explainer()
    assert built.explain("x", recorded_score=0.5, recorded_route="escalate").record.llm_consulted
    assert not built.explain("x", recorded_score=0.9, recorded_route="support").record.llm_consulted
    assert not built.explain(
        "x", recorded_score=0.01, recorded_route="skip-llm"
    ).record.llm_consulted


def test_the_surrogate_imitates_the_model_not_the_labels() -> None:
    """The model keys on "alpha"; the labels do not. The words must describe the model."""
    up, _ = explainer().top_terms(3)
    assert up[0][0] == "alpha"


def test_top_terms_run_in_both_directions() -> None:
    up, down = explainer().top_terms(5)
    assert [c for _, c in up] == sorted((c for _, c in up), reverse=True)
    assert [c for _, c in down] == sorted(c for _, c in down)
    assert up[0][1] > down[0][1]


def test_fidelity_is_measured_and_bounded() -> None:
    fidelity = explainer().fidelity
    assert fidelity.r2 <= 1.0
    assert 0.0 <= fidelity.route_agreement <= 1.0
    assert fidelity.folds == 5


def test_apostrophes_stay_inside_words() -> None:
    rows = [make_row(f"a{i}", free_text=f"i don't know {i}") for i in range(10)]
    built = Explainer(rows, model(), WordEmbedder(), BANDS, artifact_sha256="a" * 64)
    up, down = built.top_terms(50)
    words = {w for w, _ in [*up, *down]}
    assert "don't" in words and "don" not in words


def test_word_attributions_have_the_right_sign() -> None:
    explanation = explainer().explain("alpha alpha")
    assert explanation.toward_distress and explanation.toward_distress[0][0] == "alpha"
    assert all(c > 0 for _, c in explanation.toward_distress)
    assert all(c < 0 for _, c in explanation.away_from_distress)


def test_the_explainer_needs_training_rows() -> None:
    with pytest.raises(ValueError, match="training rows"):
        Explainer([], model(), WordEmbedder(), BANDS, artifact_sha256="a" * 64)
