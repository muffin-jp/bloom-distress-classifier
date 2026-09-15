"""Why the model said what it said — and how far each answer can be trusted.

The deployed model is a weight vector over 384 embedding dimensions, and no one
can read dimension 197. Pointing at those weights and calling it an explanation
would be a pretence. So three mechanisms instead, each honest about its limits:

**Nearest labelled neighbours.** The note is placed in the same embedding space
the model reads, beside the *training* rows closest to it. "Scored 0.99; four of
its five nearest training notes are distress, all from the exhaustion mode" is
case-based and checkable: a reviewer can read those five notes. It shows what the
note resembles, not the model's arithmetic.

**A lexical surrogate.** A TF-IDF ridge regression fitted to the *deployed
model's* logits over the training rows — not to the labels — so its coefficients
describe what the model responds to, in words. It is an approximation, and its
fidelity is measured and printed beside every use: cross-validated R², and how
often its routing agrees with the model's. Where fidelity is low, the words are
a hint, not a reason.

**A decision record.** What the service would log for one note: the score, the
route, whether the LLM was consulted, the artifact and embedder that produced it,
and the neighbours' ids. It carries a hash of the note rather than the note, so
logging decisions does not mean retaining what players wrote.

What this module reads
----------------------
The training split, to build the neighbour index and fit the surrogate. For
explaining the test set's errors (``--errors``), the saved predictions: scores and
routes are taken from that file as recorded at the test look, never recomputed,
and no metric is calculated on test rows.

    uv run --extra embed python -m dc.explain "i just don't have it in me tonight"
    uv run --extra embed python -m dc.explain --errors
"""

# scikit-learn and numpy leak Unknown under pyright strict at this boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from dc.cascade import Route, Thresholds, route
from dc.features import Embedder, feeling_one_hot
from dc.model import DistressModel
from dc.schema import Feeling, Row

__all__ = [
    "SURROGATE_ALPHA",
    "DecisionRecord",
    "Explainer",
    "Explanation",
    "Fidelity",
    "Neighbour",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports"

#: Fixed, not tuned: the surrogate explains the model, and tuning it for a
#: flattering fidelity number would be tuning the explanation rather than the model.
SURROGATE_ALPHA = 1.0
NEIGHBOURS = 5
FOLDS = 5
_EPS = 1e-6


@dataclass(frozen=True)
class Neighbour:
    id: str
    similarity: float
    label: int
    category: str
    text: str


@dataclass(frozen=True)
class Fidelity:
    """How closely the lexical surrogate imitates the deployed model."""

    r2: float
    route_agreement: float
    folds: int


@dataclass(frozen=True)
class DecisionRecord:
    """What the service would log. A hash of the note, never the note."""

    text_sha256: str
    feeling: str
    score: float
    route: str
    llm_consulted: bool
    low: float
    high: float
    artifact_sha256: str
    embedder: str
    neighbour_ids: tuple[str, ...]


@dataclass(frozen=True)
class Explanation:
    text: str
    record: DecisionRecord
    neighbours: tuple[Neighbour, ...]
    toward_distress: tuple[tuple[str, float], ...]
    away_from_distress: tuple[tuple[str, float], ...]

    @property
    def neighbour_distress_share(self) -> float:
        return (
            sum(n.label for n in self.neighbours) / len(self.neighbours) if self.neighbours else 0.0
        )


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


def _vectorizer() -> TfidfVectorizer:
    # The token pattern keeps apostrophes, so "don't" stays one word instead of the
    # default's "don" — the word list is read by people, and has to read as words.
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w[\w']*\b",
    )


class Explainer:
    """Neighbours and a lexical surrogate, built from the training split only."""

    def __init__(
        self,
        train: Sequence[Row],
        model: DistressModel,
        embedder: Embedder,
        thresholds: Thresholds,
        *,
        artifact_sha256: str,
        k: int = NEIGHBOURS,
    ) -> None:
        if not train:
            raise ValueError("the explainer needs training rows")
        self._train = list(train)
        self._model = model
        self._embedder = embedder
        self._thresholds = thresholds
        self._artifact_sha256 = artifact_sha256
        self._k = k

        self._vectors = np.asarray(embedder.embed([r.free_text for r in self._train]), dtype=float)
        scores = self._score_matrix(self._vectors, [r.feeling for r in self._train])
        self._train_logits = _logit(scores)

        self._tfidf = _vectorizer()
        matrix = self._tfidf.fit_transform([r.free_text for r in self._train])
        self._surrogate = Ridge(alpha=SURROGATE_ALPHA).fit(matrix, self._train_logits)
        self.fidelity = self._cross_validated_fidelity(scores)

    def _score_matrix(self, vectors: np.ndarray, feelings: Sequence[Feeling]) -> np.ndarray:
        x = vectors
        if self._model.layout.include_feeling:
            x = np.hstack([vectors, feeling_one_hot(list(feelings))])
        return self._model.predict_proba_features(x)

    def _cross_validated_fidelity(self, scores: np.ndarray) -> Fidelity:
        """Surrogate fidelity on rows it did not see, grouped so paraphrases don't flatter it."""
        texts = [r.free_text for r in self._train]
        groups = np.array([r.origin_id for r in self._train])
        predicted = np.full(len(texts), np.nan)
        folds = min(FOLDS, len(set(groups)))
        for fit_index, held_index in GroupKFold(n_splits=folds).split(texts, groups=groups):
            vectorizer = _vectorizer()
            fitted = vectorizer.fit_transform([texts[i] for i in fit_index])
            ridge = Ridge(alpha=SURROGATE_ALPHA).fit(fitted, self._train_logits[fit_index])
            predicted[held_index] = ridge.predict(
                vectorizer.transform([texts[i] for i in held_index])
            )
        surrogate_routes = [route(float(s), self._thresholds) for s in _sigmoid(predicted)]
        model_routes = [route(float(s), self._thresholds) for s in scores]
        agreement = float(
            np.mean([a is b for a, b in zip(surrogate_routes, model_routes, strict=True)])
        )
        return Fidelity(float(r2_score(self._train_logits, predicted)), agreement, folds)

    def top_terms(self, n: int = 20) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """The words the model responds to most, in each direction — via the surrogate."""
        names = self._tfidf.get_feature_names_out()
        coef = np.asarray(self._surrogate.coef_, dtype=float)
        order = np.argsort(coef)
        up = [(str(names[i]), float(coef[i])) for i in order[::-1][:n]]
        down = [(str(names[i]), float(coef[i])) for i in order[:n]]
        return up, down

    def explain(
        self,
        text: str,
        feeling: Feeling = Feeling.CUSTOM,
        *,
        recorded_score: float | None = None,
        recorded_route: str | None = None,
        n_words: int = 3,
    ) -> Explanation:
        """Explain one note.

        ``recorded_score`` and ``recorded_route`` let an already-scored note — a test
        error from the saved predictions — be explained without being re-scored.
        """
        vector = np.asarray(self._embedder.embed([text]), dtype=float)
        if recorded_score is None:
            score = float(self._score_matrix(vector, [feeling])[0])
            decided = route(score, self._thresholds).value
        else:
            score = recorded_score
            decided = recorded_route or route(score, self._thresholds).value

        similarities = self._vectors @ vector[0]
        nearest = np.argsort(similarities, kind="stable")[::-1][: self._k]
        neighbours = tuple(
            Neighbour(
                id=self._train[int(i)].id,
                similarity=float(similarities[int(i)]),
                label=self._train[int(i)].label,
                category=self._train[int(i)].category.value,
                text=self._train[int(i)].free_text,
            )
            for i in nearest
        )

        present = cast("csr_matrix", self._tfidf.transform([text])).tocoo()
        names = self._tfidf.get_feature_names_out()
        coef = np.asarray(self._surrogate.coef_, dtype=float)
        pairs = zip(present.col, present.data, strict=True)
        contributions = sorted(
            ((str(names[j]), float(v * coef[j])) for j, v in pairs), key=lambda item: item[1]
        )
        toward = tuple((w, c) for w, c in reversed(contributions) if c > 0)[:n_words]
        away = tuple((w, c) for w, c in contributions if c < 0)[:n_words]

        record = DecisionRecord(
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            feeling=feeling.value,
            score=score,
            route=decided,
            llm_consulted=decided == Route.ESCALATE.value,
            low=self._thresholds.low,
            high=self._thresholds.high,
            artifact_sha256=self._artifact_sha256,
            embedder=f"{self._model.layout.embed_model}@{self._model.layout.embed_revision}",
            neighbour_ids=tuple(n.id for n in neighbours),
        )
        return Explanation(text, record, neighbours, toward, away)


# --- rendering ------------------------------------------------------------------------


def _words(pairs: Sequence[tuple[str, float]]) -> str:
    return ", ".join(f"“{w}” {c:+.2f}" for w, c in pairs) or "—"


def render_explanation(explanation: Explanation, fidelity: Fidelity) -> str:
    r = explanation.record
    lines = [
        f"Note     {explanation.text}",
        f"Score    {r.score:.3f} → {r.route}"
        + ("  (LLM consulted)" if r.llm_consulted else "  (LLM not consulted)"),
        f"Bands    skip below {r.low:.4f} · support above {r.high:.3f}",
        "",
        f"Nearest training notes — {explanation.neighbour_distress_share:.0%} distress:",
    ]
    for n in explanation.neighbours:
        tag = "distress" if n.label else n.category
        lines.append(f"  {n.similarity:.2f}  {tag:17} {n.id:24} {n.text}")
    lines += [
        "",
        f"Words (surrogate — fidelity R² {fidelity.r2:.2f}, "
        f"routes agree {fidelity.route_agreement:.0%}):",
        f"  toward distress  {_words(explanation.toward_distress)}",
        f"  away from it     {_words(explanation.away_from_distress)}",
    ]
    return "\n".join(lines)


def render_errors_markdown(
    explainer: Explainer, explanations: Sequence[tuple[str, str, Explanation]]
) -> str:
    fidelity = explainer.fidelity
    up, down = explainer.top_terms(15)
    lines = [
        "# Explanations",
        "",
        "Three mechanisms, none of them the deployed model's own weights — which are 384 "
        "embedding dimensions no one can read.",
        "",
        "## How far to trust each",
        "",
        "| Mechanism | What it shows | What it cannot show |",
        "| --- | --- | --- |",
        "| Nearest training notes | What a note resembles, in the model's own embedding space "
        "| The model's arithmetic |",
        f"| Lexical surrogate | Which words the model's scores move with | Anything outside words; "
        f"it matches the model at R² **{fidelity.r2:.2f}** and agrees on the route "
        f"**{fidelity.route_agreement:.0%}** of the time ({fidelity.folds}-fold, grouped) |",
        "| Decision record | What was decided, by which artifact, with which inputs | Why |",
        "",
        "## What the model responds to",
        "",
        "Surrogate coefficients on the deployed model's logit, fitted over the training rows. "
        "An approximation of the model, labelled as one.",
        "",
        "| Toward distress | | Away from distress | |",
        "| --- | --- | --- | --- |",
    ]
    for (w_up, c_up), (w_down, c_down) in zip(up, down, strict=True):
        lines.append(f"| {w_up} | {c_up:+.2f} | {w_down} | {c_down:+.2f} |")

    lines += [
        "",
        "## The test set's errors, explained",
        "",
        "Scores and routes are as recorded at the test look, read from the saved predictions; "
        "nothing here re-scores a test row or computes a metric on one.",
        "",
    ]
    for row_id, cause, e in explanations:
        lines += [
            f"### `{row_id}` — {e.record.score:.3f} → `{e.record.route}`",
            "",
            f"> {e.text}",
            "",
            f"Cause (from the error analysis): {cause or '—'}. "
            f"Nearest training notes: **{e.neighbour_distress_share:.0%} distress**. "
            f"Words toward distress: {_words(e.toward_distress)}.",
            "",
            "| Similarity | Label | Row | Note |",
            "| --- | --- | --- | --- |",
        ]
        for n in e.neighbours:
            tag = "distress" if n.label else n.category
            lines.append(f"| {n.similarity:.2f} | {tag} | `{n.id}` | {n.text} |")
        lines.append("")
    return "\n".join(lines)


# --- command ------------------------------------------------------------------------


def from_repo() -> Explainer:
    from dc.artifact import ARTIFACT_DIR, load
    from dc.features import load_embedder
    from dc.ledger import files_sha256
    from dc.splits import load_all_rows, load_assignment, split_rows

    artifact = load()
    bands = artifact.metadata["thresholds"]
    return Explainer(
        split_rows(load_all_rows(), load_assignment(), "train"),
        artifact.model,
        load_embedder(),
        Thresholds(float(bands["low"]), float(bands["high"])),
        artifact_sha256=files_sha256([ARTIFACT_DIR / "model.npz", ARTIFACT_DIR / "model.json"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="a note to explain")
    parser.add_argument(
        "--feeling", default=Feeling.CUSTOM.value, choices=[f.value for f in Feeling]
    )
    parser.add_argument("--json", action="store_true", help="print the decision record only")
    parser.add_argument(
        "--errors", action="store_true", help="explain the test set's recorded errors"
    )
    args = parser.parse_args()
    if not args.text and not args.errors:
        parser.error("give a note to explain, or --errors")

    explainer = from_repo()

    if args.text:
        explanation = explainer.explain(args.text, Feeling(args.feeling))
        if args.json:
            print(json.dumps(asdict(explanation.record), indent=2))
        else:
            print(render_explanation(explanation, explainer.fidelity))
        return

    from dc.evaluate import PREDICTIONS_PATH, load_analysis, load_predictions

    predictions = load_predictions(PREDICTIONS_PATH)
    notes = load_analysis().errors
    errors = [
        p
        for p in predictions
        if (p.label == 0 and p.route == Route.SUPPORT.value)
        or (p.label == 1 and p.route != Route.SUPPORT.value)
    ]
    explained = [
        (
            p.id,
            notes[p.id].cause if p.id in notes else "",
            explainer.explain(
                p.text, Feeling(p.feeling), recorded_score=p.score, recorded_route=p.route
            ),
        )
        for p in sorted(
            errors, key=lambda p: (notes[p.id].cause if p.id in notes else "", -p.score)
        )
    ]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "explanations.md").write_text(
        render_errors_markdown(explainer, explained), encoding="utf-8"
    )
    up, down = explainer.top_terms(15)
    payload: dict[str, Any] = {
        "fidelity": asdict(explainer.fidelity),
        "surrogate_alpha": SURROGATE_ALPHA,
        "top_terms": {"toward_distress": up, "away_from_distress": down},
        "errors": [
            {
                "id": row_id,
                "cause": cause,
                "record": asdict(e.record),
                "neighbour_distress_share": e.neighbour_distress_share,
                "toward_distress": e.toward_distress,
            }
            for row_id, cause, e in explained
        ],
    }
    (REPORT_DIR / "explanations.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fidelity = explainer.fidelity
    print(
        f"Explained {len(explained)} recorded error(s). "
        f"Surrogate R² {fidelity.r2:.2f}, route agreement {fidelity.route_agreement:.0%}."
    )


if __name__ == "__main__":
    main()
