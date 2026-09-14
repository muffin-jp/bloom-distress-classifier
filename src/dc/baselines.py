"""The four baselines, and the teacher, behind one interface.

Each exists to make the next one prove itself. Reported in this order, the table
answers the question an ML reviewer asks first — *did the model beat the obvious
thing?* — before it answers any other.

0. **Majority class.** The floor. Its accuracy is what "accuracy" is worth on a
   72/28 split, which is why accuracy is not the headline metric anywhere here.
1. **Crisis-keyword lexicon.** The dumb baseline, written in good faith rather
   than as a strawman. If it wins, the honest conclusion is that this project
   needed a word list, not a model. It is also the one that should fail hardest
   on ``game-frustration``, because "I want to die, this stage took me 40 tries"
   contains every crisis word there is.
2. **TF-IDF + logistic regression.** Does lexical signal alone suffice, once it
   is *weighted* rather than matched? Doubles as the explanation surface: its
   coefficients are words.
3. **MiniLM embeddings + logistic regression.** The candidate. The question it
   answers is whether the frozen embedding space separates game hyperbole from
   real despair — if it does not, no linear model on top can fix that, and
   fine-tuning becomes justified rather than assumed.
4. **The teacher.** ``claude-haiku-4-5``, read from votes already recorded by
   ``scripts/propose_labels.py`` — so scoring it costs nothing. The ceiling to
   match, and the thing being replaced.
"""

# scikit-learn and numpy leak Unknown under pyright strict at this boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dc.features import Embedder
from dc.schema import Row

__all__ = [
    "CRISIS_PATTERNS",
    "Baseline",
    "EmbeddingBaseline",
    "KeywordBaseline",
    "MajorityBaseline",
    "TeacherBaseline",
    "TfidfBaseline",
]


class Baseline(Protocol):
    """Fit on rows, score rows. ``predict_proba`` returns P(distress)."""

    name: str

    def fit(self, rows: Sequence[Row]) -> None: ...

    def predict_proba(self, rows: Sequence[Row]) -> np.ndarray: ...


class MajorityBaseline:
    """Predicts the training positive rate for everything.

    Deliberately returns the *rate* rather than the majority label: a constant
    score makes PR-AUC come out at the base rate, which is exactly the number
    every other baseline has to beat to have earned anything.
    """

    name = "majority"

    def __init__(self) -> None:
        self._rate = 0.0

    def fit(self, rows: Sequence[Row]) -> None:
        self._rate = float(np.mean([row.label for row in rows])) if rows else 0.0

    def predict_proba(self, rows: Sequence[Row]) -> np.ndarray:
        return np.full(len(rows), self._rate, dtype=float)


#: A good-faith crisis lexicon: what a careful engineer writes before reaching
#: for a model. Several of these appear verbatim in game-frustration rows, which
#: is the point — the overlap is the problem this project exists to measure, not
#: a flaw in the list.
CRISIS_PATTERNS: tuple[str, ...] = (
    r"\bkill(ing)? myself\b",
    r"\bwant to die\b",
    r"\bend it all\b",
    r"\bsuicid",
    r"\bhurt(ing)? myself\b",
    r"\bself[- ]harm\b",
    r"\bhopeless\b",
    r"\bno point\b",
    r"\bwhat'?s the point\b",
    r"\bcan'?t go on\b",
    r"\bcan'?t take it\b",
    r"\bno way out\b",
    r"\bworthless\b",
    r"\bbetter off without me\b",
    r"\bhate myself\b",
    r"\bgive up\b",
    r"\bnot safe\b",
    r"\bscared of\b",
    r"\bnothing matters\b",
    r"\bfeel nothing\b",
    r"\bbreaking down\b",
    r"\bi'?m done\b",
)


class KeywordBaseline:
    """Fires if any crisis pattern matches. No fitting — it is a rule."""

    name = "keyword"

    def __init__(self, patterns: Sequence[str] = CRISIS_PATTERNS) -> None:
        self._regex = re.compile("|".join(patterns), re.IGNORECASE)

    def fit(self, rows: Sequence[Row]) -> None:
        """No-op. A rule has nothing to learn — which is its whole appeal."""

    def predict_proba(self, rows: Sequence[Row]) -> np.ndarray:
        return np.array(
            [1.0 if self._regex.search(row.free_text) else 0.0 for row in rows], dtype=float
        )


class TfidfBaseline:
    """Character- and word-level TF-IDF into logistic regression.

    ``class_weight="balanced"`` because the positive class is both rarer and
    costlier; leaving it unset optimises for the majority, which is the opposite
    of what a safety classifier is for.
    """

    name = "tfidf+lr"

    def __init__(self, seed: int = 0) -> None:
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)

    def fit(self, rows: Sequence[Row]) -> None:
        matrix = self._vectorizer.fit_transform([row.free_text for row in rows])
        self._model.fit(matrix, [row.label for row in rows])

    def predict_proba(self, rows: Sequence[Row]) -> np.ndarray:
        matrix = self._vectorizer.transform([row.free_text for row in rows])
        return np.asarray(self._model.predict_proba(matrix)[:, 1], dtype=float)

    def top_terms(self, k: int = 15) -> list[tuple[str, float]]:
        """The words driving the decision, most positive first.

        This is the honest half of the explainability story: these coefficients
        are *words*, unlike the embedding model's, whose dimensions mean nothing
        to anyone.
        """
        names = list(self._vectorizer.get_feature_names_out())
        weights = self._model.coef_[0]
        order = np.argsort(weights)[::-1][:k]
        return [(names[int(i)], float(weights[int(i)])) for i in order]


class EmbeddingBaseline:
    """Frozen MiniLM sentence vectors into logistic regression. The candidate."""

    name = "minilm+lr"

    def __init__(self, embedder: Embedder, seed: int = 0) -> None:
        self._embedder = embedder
        self._model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        self._cache: dict[str, np.ndarray] = {}

    def _features(self, rows: Sequence[Row]) -> np.ndarray:
        """Embed, memoising by text.

        Cross-validation embeds the same row in four of five folds; without the
        cache the run spends most of its time re-encoding sentences it has
        already seen.
        """
        # dict.fromkeys dedupes while preserving order: a batch that repeats a
        # text must embed it once, not once per row.
        missing = list(
            dict.fromkeys(row.free_text for row in rows if row.free_text not in self._cache)
        )
        if missing:
            for text, vector in zip(missing, self._embedder.embed(missing), strict=True):
                self._cache[text] = vector
        return np.vstack([self._cache[row.free_text] for row in rows])

    def fit(self, rows: Sequence[Row]) -> None:
        self._model.fit(self._features(rows), [row.label for row in rows])

    def predict_proba(self, rows: Sequence[Row]) -> np.ndarray:
        return np.asarray(self._model.predict_proba(self._features(rows))[:, 1], dtype=float)


class TeacherBaseline:
    """The production LLM, scored from votes already on disk.

    Its score is the fraction of votes that said distress, so three votes give
    0, 1/3, 2/3 or 1 — a coarse probability, and enough for a PR curve. Rows with
    no recorded vote (the golden cases, which were never candidates) are scored
    ``nan`` and excluded by the caller rather than silently counted as negative.
    """

    name = "teacher(haiku)"

    def __init__(self, votes: dict[str, tuple[int, ...]]) -> None:
        self._votes = votes

    def fit(self, rows: Sequence[Row]) -> None:
        """No-op. The teacher is not trained; it is the thing being replaced."""

    def predict_proba(self, rows: Sequence[Row]) -> np.ndarray:
        out: list[float] = []
        for row in rows:
            votes = self._votes.get(row.id)
            out.append(float(np.mean(votes)) if votes else float("nan"))
        return np.asarray(out, dtype=float)

    def coverage(self, rows: Sequence[Row]) -> int:
        return sum(1 for row in rows if self._votes.get(row.id))
