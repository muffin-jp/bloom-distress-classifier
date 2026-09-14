"""Each baseline does the specific thing it is in the table to do."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from conftest import make_row

from dc.baselines import (
    EmbeddingBaseline,
    KeywordBaseline,
    MajorityBaseline,
    TeacherBaseline,
    TfidfBaseline,
)
from dc.schema import Category, Row


class StubEmbedder:
    """Two dimensions: does the text mention the game, and is it long.

    Enough to make logistic regression separable without a 90MB download, which
    keeps the whole test suite offline.
    """

    dim = 2

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(
            [[1.0 if "stage" in t or "level" in t else 0.0, len(t) / 100.0] for t in texts],
            dtype=np.float32,
        )


def corpus() -> list[Row]:
    distress = [
        make_row(f"d{i}", category=Category.DISTRESS, free_text="I can't go on with my life")
        for i in range(10)
    ]
    game = [
        make_row(f"g{i}", category=Category.GAME_FRUSTRATION, free_text="this stage is hard")
        for i in range(10)
    ]
    return distress + game


def test_majority_predicts_the_training_positive_rate() -> None:
    baseline = MajorityBaseline()
    baseline.fit([make_row("a", category=Category.DISTRESS)] + [make_row("b"), make_row("c")])
    assert baseline.predict_proba([make_row("x")]) == pytest.approx(1 / 3)


def test_majority_is_constant_across_rows() -> None:
    baseline = MajorityBaseline()
    baseline.fit(corpus())
    scores = baseline.predict_proba(corpus())
    assert len(set(scores.tolist())) == 1


def test_keyword_fires_on_a_crisis_phrase() -> None:
    baseline = KeywordBaseline()
    rows = [make_row("a", free_text="I want to die and nothing helps")]
    assert baseline.predict_proba(rows)[0] == 1.0


def test_keyword_cannot_tell_the_frames_apart() -> None:
    """The finding the baseline exists to produce.

    Identical crisis wording in a game frame and a life frame gets identical
    scores, because a lexicon sees words and the difference is context.
    """
    baseline = KeywordBaseline()
    game = make_row("g", free_text="I want to die, this stage took me 40 tries")
    life = make_row("l", free_text="I want to die. I can't do this anymore")
    scores = baseline.predict_proba([game, life])
    assert scores[0] == scores[1] == 1.0


def test_keyword_is_silent_on_ordinary_text() -> None:
    baseline = KeywordBaseline()
    assert baseline.predict_proba([make_row("a", free_text="first try, feels great")])[0] == 0.0


def test_keyword_needs_no_fitting() -> None:
    baseline = KeywordBaseline()
    baseline.fit([])  # must not raise
    assert baseline.predict_proba([make_row("a", free_text="worthless")])[0] == 1.0


def test_tfidf_separates_a_trivially_separable_set() -> None:
    baseline = TfidfBaseline()
    rows = corpus()
    baseline.fit(rows)
    scores = baseline.predict_proba(rows)
    assert scores[:10].mean() > scores[10:].mean()


def test_tfidf_exposes_word_level_coefficients() -> None:
    # The honest half of the explainability story: these are words, unlike the
    # embedding model's dimensions.
    baseline = TfidfBaseline()
    baseline.fit(corpus())
    terms = baseline.top_terms(5)
    assert len(terms) == 5
    assert all(isinstance(term, str) for term, _ in terms)


def test_embedding_baseline_learns_from_the_stub_features() -> None:
    baseline = EmbeddingBaseline(StubEmbedder())
    rows = corpus()
    baseline.fit(rows)
    scores = baseline.predict_proba(rows)
    assert scores[:10].mean() > scores[10:].mean()


def test_embedding_baseline_caches_by_text() -> None:
    class CountingEmbedder(StubEmbedder):
        calls = 0

        def embed(self, texts: Sequence[str]) -> np.ndarray:
            CountingEmbedder.calls += len(texts)
            return super().embed(texts)

    baseline = EmbeddingBaseline(CountingEmbedder())
    rows = corpus()
    baseline.fit(rows)
    baseline.predict_proba(rows)
    baseline.predict_proba(rows)
    # Two distinct texts across 20 rows and three passes.
    assert CountingEmbedder.calls == 2


def test_teacher_scores_the_fraction_of_votes() -> None:
    baseline = TeacherBaseline({"a": (1, 1, 0)})
    assert baseline.predict_proba([make_row("a")])[0] == pytest.approx(2 / 3)


def test_teacher_returns_nan_for_rows_it_never_saw() -> None:
    # The golden cases were never candidates. Scoring them 0 would silently
    # credit the teacher with correctly rejecting rows it never classified.
    baseline = TeacherBaseline({})
    assert np.isnan(baseline.predict_proba([make_row("unseen")])[0])


def test_teacher_reports_its_coverage() -> None:
    baseline = TeacherBaseline({"a": (1,)})
    assert baseline.coverage([make_row("a"), make_row("b")]) == 1
