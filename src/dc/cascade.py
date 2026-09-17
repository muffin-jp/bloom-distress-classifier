"""The cost model, the two thresholds, and what the cascade would have done.

Three routes, and two statistics
-------------------------------
Every free-text note is scored twice — as one string, and as the highest of its
segments (see :mod:`dc.text`) — and gets one of three routes:

* ``worst < low``  → **skip-llm**: straight to the encouragement branch, no LLM call.
* ``whole > high`` → **support**: straight to the reviewed support message.
* anything else    → **escalate**: ask ``claude-haiku-4-5``, exactly as production
  does today for every note.

The asymmetry is the design, twice over. Skipping the LLM is the only route that
can *add* a missed crisis, so ``low`` is not optimised at all — it is a safety
constraint — and it is compared against the *worst* part of the note, because
mean-pooling lets a long calm note hide a short alarming one. The support route
can only add a false alarm (a player who is fine receives a warm, reviewed
message), so ``high`` is the one the cost model chooses, and it reads the note as
written rather than its most alarming fragment.

The cost model
--------------
A false negative — a player in crisis routed to generated encouragement — is
treated as **20 times** as costly as a false positive. That ratio is a product
decision rather than a statistical one, and it is written down as a number so it
can be argued with. The report shows what changes at 5, 10, 50 or 100, without
retraining anything.

How each threshold is fitted
----------------------------
**low.** Take the lowest score any validation distress case received *on the
statistic the skip band actually compares* — the worst of its segments, not the
note as one string — and keep :data:`LOW_MARGIN` of it. Fitting on whole-note
scores while routing on segment maxima would measure the safety margin against a
quantity the rule never looks at, and since a segment maximum is never below the
whole-note score, it would quietly shrink the skip band by an unknown amount. By
construction no validation distress case lands in the skip band — but that is a
claim about a finite sample, so it is reported as one: with zero misses among *n*
positives, the true share of distress cases that would skip is below
``1 - 0.05 ** (1 / n)`` with 95% confidence.

**high.** Every distinct way of splitting the validation rows above ``low`` into
"escalate" and "support" is scored by expected cost, and the cheapest wins —
subject to a constraint, when one is given. ``forbidden`` marks validation rows
that may never reach support, which raises a floor under ``high``: no cutoff is
considered that would route one of them there. The cost model then chooses among
what is left, so a product rule outranks the ratio rather than competing with it. Each
cutoff is placed midway between the two scores it separates, so it never sits on
top of a validation row that the shipped model will score slightly differently.
Ties go to the *higher* cutoff: escalating more stays closer to production.

Because every split's missed-crisis count and false-alarm count are computed once,
the same curve answers the question the ratio cannot: across *every* possible
ratio, which operating points can the cost model choose at all, and where does it
switch between them? A ratio is a point on a staircase, and a staircase has cliffs.

Both parameters — the 20:1 ratio and the 0.5 margin — were fixed before any score
distribution was examined.

The simulation uses expected values: the LLM is modelled as one call that says
distress with probability equal to the share of its recorded votes on that note.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dc.metrics import FloatArrayLike, IntArrayLike
from dc.text import ScopeRule, Segmentation

__all__ = [
    "COST_MODEL",
    "DEFAULT_MARGINS",
    "LOW_MARGIN",
    "RATIO_GRID",
    "TOLERANCE",
    "AdversarialFloor",
    "CascadeFit",
    "Constraint",
    "CostCurve",
    "CostModel",
    "LowFit",
    "MarginRow",
    "Outcome",
    "Route",
    "Segment",
    "Thresholds",
    "cost_curve",
    "decisive_positives",
    "envelope",
    "fit_cascade",
    "fit_high",
    "fit_low",
    "probability",
    "route",
    "route_note",
    "simulate",
    "skip_band_positives",
    "skip_statistic",
    "teacher_probabilities",
]


class Route(StrEnum):
    SKIP_LLM = "skip-llm"
    ESCALATE = "escalate"
    SUPPORT = "support"


@dataclass(frozen=True)
class CostModel:
    false_negative: float = 20.0
    false_positive: float = 1.0

    def __post_init__(self) -> None:
        if not (self.false_negative > 0 and self.false_positive > 0):
            raise ValueError("costs must be positive")

    @property
    def ratio(self) -> float:
        return self.false_negative / self.false_positive


COST_MODEL = CostModel()
LOW_MARGIN = 0.5
#: Ratios swept to discover the operating points; the switch points between them
#: are then solved exactly, so the grid only has to be fine enough to visit each.
RATIO_GRID = np.geomspace(0.01, 1000.0, 2000)
DEFAULT_MARGINS: tuple[float, ...] = (1.0, 0.5, 0.25)
#: How far a worst-segment score may fall below its whole-note score before the
#: two arrays are treated as swapped rather than noisy. Embeddings are float32,
#: and the same note scored inside a batch of notes and inside a batch of segments
#: sums in a different order, which moves the last bit: measured at 1.2e-7 over the
#: training rows. A genuine swap moves scores by 0.1 or more, so this separates the
#: two by six orders of magnitude.
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Thresholds:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError("thresholds must be finite")
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"need 0 <= low <= high <= 1, got low={self.low}, high={self.high}")


def probability(score: object) -> float | None:
    """The score as a probability, or ``None`` if it is not one.

    Every way the local model can fail — a missing score, a NaN, a string, a
    number outside [0, 1] — comes back as ``None`` here, and every caller turns
    that into an escalation. Failure lands on today's behaviour by construction.
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    value = float(score)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return None
    return value


def route(score: object, thresholds: Thresholds) -> Route:
    """The decision from a whole-note score alone. A score on either threshold escalates.

    This is the rule as it was first fitted, and it is what the red-team probe
    defeated: a crisis clause diluted by game talk scores below ``low`` as one
    string. :func:`route_note` is what the service uses.
    """
    value = probability(score)
    if value is None:
        return Route.ESCALATE
    if value < thresholds.low:
        return Route.SKIP_LLM
    if value > thresholds.high:
        return Route.SUPPORT
    return Route.ESCALATE


def route_note(whole: object, worst_segment: object, thresholds: Thresholds) -> Route:
    """The decision the service makes, from a note's two scores.

    **The whole note decides support; the worst segment guards the skip band.**

    The two bands answer different questions, so they read different statistics.
    Support asks "is this note, as written, a crisis?" — a whole-note judgement,
    and the one ``high`` was fitted on. Routing on the worst segment instead would
    send a note to support because one clause inside it looked bad in isolation,
    which is a new source of false alarms that nothing here has measured.

    Skipping asks the opposite: "is there *nothing* here that needs a human-grade
    reader?" That has to be true of every part of the note, because mean-pooling
    lets a long calm note hide a short alarming one. So a note skips only when its
    highest-scoring segment is still below ``low``.

    Since the whole note is itself a segment, ``worst >= whole``, and the skip band
    can only ever be narrower than the whole-note rule's — never wider.
    """
    value = probability(whole)
    if value is None:
        return Route.ESCALATE
    if value > thresholds.high:
        return Route.SUPPORT
    worst = probability(worst_segment)
    if worst is None:
        return Route.ESCALATE
    if worst < thresholds.low:
        return Route.SKIP_LLM
    return Route.ESCALATE


@dataclass(frozen=True)
class Outcome:
    """Expected results of one routing policy over the validation rows."""

    name: str
    n: int
    positives: int
    expected_recall: float
    expected_missed: float
    expected_false_positives: float
    expected_cost: float
    skip_share: float
    escalate_share: float
    support_share: float


def skip_statistic(scores: FloatArrayLike, skip_scores: FloatArrayLike | None) -> np.ndarray:
    """The score the skip band compares against ``low``.

    ``None`` means the whole-note score, which is the pre-segmentation rule and
    what the unit tests exercise. Otherwise it is the worst segment, and the
    ordering is checked: a segment maximum includes the whole note, so it can
    never be lower. That check is here because passing the two arrays the wrong
    way round would widen the skip band and produce entirely plausible numbers.
    """
    whole = np.asarray(scores, dtype=float)
    if skip_scores is None:
        return whole
    worst = np.asarray(skip_scores, dtype=float)
    if worst.shape != whole.shape:
        raise ValueError("skip scores and scores differ in length")
    comparable = np.isfinite(worst) & np.isfinite(whole)
    if np.any(worst[comparable] < whole[comparable] - TOLERANCE):
        raise ValueError(
            "a skip score is below its whole-note score, which is impossible when the "
            "whole note is one of the segments. Are the two arrays swapped?"
        )
    # Within tolerance the shortfall is float32 noise, not a different answer, so the
    # definition is restored rather than carried forward a fraction of an epsilon out.
    return np.maximum(worst, whole)


def teacher_probabilities(votes: Sequence[Sequence[int]]) -> np.ndarray:
    """P(the LLM says distress) per row: the share of its recorded votes."""
    missing = sum(1 for v in votes if not v)
    if missing:
        raise ValueError(
            f"{missing} row(s) have no teacher votes, so the escalate route cannot be "
            "simulated for them. Run `make propose-run`."
        )
    return np.array([sum(v) / len(v) for v in votes], dtype=float)


def simulate(
    y: IntArrayLike,
    scores: FloatArrayLike,
    teacher: FloatArrayLike,
    thresholds: Thresholds,
    *,
    costs: CostModel = COST_MODEL,
    name: str = "cascade",
    skip_scores: FloatArrayLike | None = None,
) -> Outcome:
    truth = np.asarray(y, dtype=float)
    p = np.asarray(scores, dtype=float)
    q = np.asarray(teacher, dtype=float)
    if not truth.shape == p.shape == q.shape:
        raise ValueError("y, scores and teacher probabilities differ in length")
    positives = float(np.sum(truth))
    if positives == 0:
        raise ValueError("no positives: recall is undefined")

    # nan compares False both ways, so a missing score escalates, as route_note()
    # does. Support reads the whole note; the skip band reads the worst segment.
    worst = skip_statistic(p, skip_scores)
    support = p > thresholds.high
    skip = (worst < thresholds.low) & ~support
    escalate = ~(skip | support)
    flagged = np.where(support, 1.0, np.where(escalate, q, 0.0))
    missed = float(np.sum(truth * (1.0 - flagged)))
    false_positives = float(np.sum((1.0 - truth) * flagged))
    return Outcome(
        name=name,
        n=int(truth.size),
        positives=int(positives),
        expected_recall=(positives - missed) / positives,
        expected_missed=missed,
        expected_false_positives=false_positives,
        expected_cost=costs.false_negative * missed + costs.false_positive * false_positives,
        skip_share=float(np.mean(skip)),
        escalate_share=float(np.mean(escalate)),
        support_share=float(np.mean(support)),
    )


@dataclass(frozen=True)
class AdversarialFloor:
    """Notes written to evade the skip band, and the ceiling they put on ``low``.

    The mirror image of :class:`Constraint`. That one forbids rows from reaching
    support and so raises a floor under ``high``; this one forbids notes from
    skipping and so lowers a ceiling onto ``low``. Both are product rules applied
    before any fitted number, not prices traded against one.

    Applying it has a cost worth stating plainly: after this, ``make redteam``
    passes **by construction**, exactly as the release-gate constraint makes its
    own check pass by construction. The probe stops being evidence about this
    artifact and becomes a guarantee about it. Only notes written *after* this fit
    can test the skip band again.
    """

    label: str
    n: int
    #: Lowest worst-segment score in the probe set, scored by the shipped model.
    lowest: float
    set_by: str
    set_by_text: str
    #: Whether the probe, rather than the labelled distress rows, decided ``low``.
    binding: bool


@dataclass(frozen=True)
class LowFit:
    threshold: float
    margin: float
    lowest_positive_score: float
    #: (id, score, text) of the distress cases closest to the skip band — the rows
    #: that decided ``low``, and the first ones a reviewer should re-read.
    nearest_positives: tuple[tuple[str, float, str], ...]
    n_positive: int
    #: The adversarial probe, if one constrained the fit.
    adversarial: AdversarialFloor | None = None

    @property
    def miss_upper_bound(self) -> float:
        """95% upper bound on the share of distress cases the skip band would take."""
        return 1.0 - 0.05 ** (1.0 / self.n_positive)


def fit_low(
    y: IntArrayLike,
    scores: FloatArrayLike,
    *,
    ids: Sequence[str],
    texts: Sequence[str],
    margin: float = LOW_MARGIN,
    n_nearest: int = 5,
    adversarial: Sequence[tuple[str, float, str]] = (),
    adversarial_label: str = "",
) -> LowFit:
    """``low`` = ``margin`` x the lowest score any note that must not skip received.

    Two kinds of note must not skip: the labelled distress rows, scored out of
    fold, and — when given — the adversarial probe, scored by the shipped model.
    The lower of the two decides, because a rule that only holds for one of them
    does not hold. ``adversarial`` is ``(id, worst-segment score, text)`` per note.
    """
    if not 0.0 < margin <= 1.0:
        raise ValueError(f"margin must be in (0, 1], got {margin}")
    truth = np.asarray(y, dtype=int)
    p = np.asarray(scores, dtype=float)
    if not len(ids) == len(texts) == truth.size == p.size:
        raise ValueError("y, scores, ids and texts differ in length")
    positive_index = np.flatnonzero((truth == 1) & np.isfinite(p))
    if positive_index.size == 0:
        raise ValueError("no scored positives to fit the skip band against")
    order = positive_index[np.argsort(p[positive_index], kind="stable")]
    lowest = float(p[order[0]])

    floor: AdversarialFloor | None = None
    base = lowest
    if adversarial:
        probe_id, probe_score, probe_text = min(adversarial, key=lambda row: row[1])
        base = min(lowest, probe_score)
        floor = AdversarialFloor(
            label=adversarial_label or "notes written to evade the skip band",
            n=len(adversarial),
            lowest=probe_score,
            set_by=probe_id,
            set_by_text=probe_text,
            binding=probe_score < lowest,
        )
    return LowFit(
        threshold=margin * base,
        margin=margin,
        lowest_positive_score=lowest,
        nearest_positives=tuple(
            (ids[int(i)], float(p[int(i)]), texts[int(i)]) for i in order[:n_nearest]
        ),
        n_positive=int(positive_index.size),
        adversarial=floor,
    )


def skip_band_positives(
    y: IntArrayLike, skip_scores: FloatArrayLike, thresholds: Thresholds
) -> list[int]:
    """Indices of distress cases that would skip the LLM. Must be empty on validation.

    ``skip_scores`` is the statistic the skip band compares — the worst segment
    once segmentation is in play, the whole note before it.
    """
    truth = np.asarray(y, dtype=int)
    p = np.asarray(skip_scores, dtype=float)
    return [int(i) for i in np.flatnonzero((truth == 1) & (p < thresholds.low))]


@dataclass(frozen=True)
class Constraint:
    """Validation rows that may never reach support, and the floor they impose."""

    label: str
    floor: float
    rows: int
    #: The row whose score set the floor — the one closest to being routed wrongly.
    set_by: str
    set_by_text: str


@dataclass(frozen=True, eq=False)
class CostCurve:
    """Expected misses and false alarms for every distinct choice of ``high``."""

    low: float
    candidates: np.ndarray
    expected_missed: np.ndarray
    expected_false_positives: np.ndarray
    escalate_share: np.ndarray
    support_share: np.ndarray
    positives: int
    #: Cutoffs below this are not permitted; see :class:`Constraint`.
    floor: float = 0.0

    def permitted(self) -> np.ndarray:
        return self.candidates >= self.floor

    def best_index(self, costs: CostModel) -> int:
        """Cheapest permitted candidate; ties go to the highest cutoff."""
        total = (
            costs.false_negative * self.expected_missed
            + costs.false_positive * self.expected_false_positives
        )
        allowed = np.flatnonzero(self.permitted())
        if allowed.size == 0:
            raise ValueError("the constraint permits no cutoff at all")
        best = total[allowed].min()
        return int(allowed[total[allowed] <= best + 1e-9].max())


def cost_curve(
    y: IntArrayLike,
    scores: FloatArrayLike,
    teacher: FloatArrayLike,
    *,
    low: float,
    floor: float = 0.0,
    skip_scores: FloatArrayLike | None = None,
) -> CostCurve:
    truth = np.asarray(y, dtype=float)
    p = np.asarray(scores, dtype=float)
    q = np.asarray(teacher, dtype=float)
    if not truth.shape == p.shape == q.shape:
        raise ValueError("y, scores and teacher probabilities differ in length")
    positives = int(np.sum(truth))
    if positives == 0:
        raise ValueError("no positives: recall is undefined")

    skip = skip_statistic(p, skip_scores) < low
    # Candidates come from the notes `high` can act on, which is why the bound is
    # `low` and not "not skipped". A note below `low` that escapes the skip band on
    # one alarming segment still escalates at every valid cutoff — `high` may never
    # sit below `low` — so it is not a place to put one.
    in_range = p[np.isfinite(p) & (p >= low) & (p <= 1.0)]
    edges = np.unique(np.concatenate([[low, 1.0], in_range]))
    # Midpoints: each candidate splits the rows exactly as the score below it would,
    # without resting on a validation row. 1.0 is "never route straight to support".
    candidates = np.concatenate([(edges[:-1] + edges[1:]) / 2.0, [1.0]])

    support = p[None, :] > candidates[:, None]
    escalate = ~support & ~skip[None, :]
    flagged = np.where(support, 1.0, np.where(escalate, q[None, :], 0.0))
    return CostCurve(
        low=low,
        candidates=candidates,
        expected_missed=np.sum(truth[None, :] * (1.0 - flagged), axis=1),
        expected_false_positives=np.sum((1.0 - truth)[None, :] * flagged, axis=1),
        escalate_share=np.mean(escalate, axis=1),
        support_share=np.mean(support, axis=1),
        positives=positives,
        floor=floor,
    )


def constraint_floor(scores: FloatArrayLike, forbidden: IntArrayLike) -> float:
    """The smallest ``high`` that keeps every forbidden row out of support.

    Support is ``p > high``, so a row is excluded once ``high`` reaches its score.
    """
    p = np.asarray(scores, dtype=float)
    mask = np.asarray(forbidden, dtype=bool)
    if mask.shape != p.shape:
        raise ValueError("forbidden mask and scores differ in length")
    blocked = p[mask & np.isfinite(p)]
    return float(blocked.max()) if blocked.size else 0.0


def fit_high(
    y: IntArrayLike,
    scores: FloatArrayLike,
    teacher: FloatArrayLike,
    *,
    low: float,
    costs: CostModel = COST_MODEL,
    forbidden: IntArrayLike | None = None,
    skip_scores: FloatArrayLike | None = None,
) -> float:
    floor = 0.0 if forbidden is None else constraint_floor(scores, forbidden)
    curve = cost_curve(y, scores, teacher, low=low, floor=floor, skip_scores=skip_scores)
    return float(curve.candidates[curve.best_index(costs)])


@dataclass(frozen=True)
class Segment:
    """One operating point, and the range of ratios that would choose it."""

    ratio_from: float
    ratio_to: float | None
    high: float
    expected_recall: float
    expected_missed: float
    expected_false_positives: float
    escalate_share: float
    support_share: float

    def contains(self, ratio: float) -> bool:
        return self.ratio_from <= ratio and (self.ratio_to is None or ratio < self.ratio_to)


def envelope(curve: CostCurve, *, grid: np.ndarray = RATIO_GRID) -> tuple[Segment, ...]:
    """Every operating point any ratio can select, with exact switch points.

    As the ratio rises, the cheapest cutoff can only move down — more support,
    fewer misses — so sweeping the ratio visits each operating point once, in
    order. The switch between two neighbours is where their costs are equal:
    ``ratio * missed_a + false_alarms_a == ratio * missed_b + false_alarms_b``.
    """
    order: list[int] = []
    for ratio in grid:
        index = curve.best_index(CostModel(float(ratio), 1.0))
        if not order or order[-1] != index:
            order.append(index)

    def crossing(before: int, after: int) -> float:
        gain = float(curve.expected_missed[before] - curve.expected_missed[after])
        extra = float(
            curve.expected_false_positives[after] - curve.expected_false_positives[before]
        )
        return extra / gain if gain > 0 else float(grid[0])

    segments: list[Segment] = []
    for n, index in enumerate(order):
        segments.append(
            Segment(
                ratio_from=0.0 if n == 0 else crossing(order[n - 1], index),
                ratio_to=None if n == len(order) - 1 else crossing(index, order[n + 1]),
                high=float(curve.candidates[index]),
                expected_recall=1.0 - float(curve.expected_missed[index]) / curve.positives,
                expected_missed=float(curve.expected_missed[index]),
                expected_false_positives=float(curve.expected_false_positives[index]),
                escalate_share=float(curve.escalate_share[index]),
                support_share=float(curve.support_share[index]),
            )
        )
    return tuple(segments)


def decisive_positives(
    y: IntArrayLike,
    scores: FloatArrayLike,
    teacher: FloatArrayLike,
    *,
    ids: Sequence[str],
    texts: Sequence[str],
    high: float,
    alternative_high: float,
) -> tuple[tuple[str, float, float, str], ...]:
    """Distress cases whose expected outcome changes between two choices of ``high``.

    Between the two cutoffs a case goes to support under one and to the LLM under
    the other. It only matters if the LLM might miss it — so these are exactly the
    rows where the teacher's votes disagreed with the reviewer, and exactly the
    rows a whole operating point can rest on. Returned as (id, score, share of
    teacher votes that said distress, text).
    """
    truth = np.asarray(y, dtype=int)
    p = np.asarray(scores, dtype=float)
    q = np.asarray(teacher, dtype=float)
    lower, upper = sorted((high, alternative_high))
    index = np.flatnonzero((truth == 1) & (p > lower) & (p <= upper) & (q < 1.0))
    index = index[np.argsort(p[index], kind="stable")]
    return tuple((ids[int(i)], float(p[int(i)]), float(q[int(i)]), texts[int(i)]) for i in index)


@dataclass(frozen=True)
class MarginRow:
    margin: float
    low: float
    skip_share: float


@dataclass(frozen=True)
class CascadeFit:
    thresholds: Thresholds
    costs: CostModel
    low: LowFit
    cascade: Outcome
    llm_alone: Outcome
    model_alone: Outcome
    #: The same thresholds with the skip band reading whole notes instead of
    #: segments — the rule the red-team probe defeated. ``None`` when no segment
    #: scores were supplied. The gap between this and ``cascade`` is what the fix
    #: costs in skipped LLM calls.
    whole_note_cascade: Outcome | None
    envelope: tuple[Segment, ...]
    #: The product rule that outranked the cost model, if one was given.
    constraint: Constraint | None
    #: How notes were cut up to produce the skip band's scores. Recorded here so it
    #: reaches the artifact from the same call that fitted the threshold, rather
    #: than being supplied again, correctly, by a second caller.
    segmentation: Segmentation | None
    #: Which notes the model declines to judge at all. Recorded with the fit for
    #: the same reason as the segmentation: the service must apply the same rule.
    scope: ScopeRule | None
    chosen_segment: int
    #: The operating point one step more conservative than the chosen one, if any.
    alternative: Segment | None
    decisive: tuple[tuple[str, float, float, str], ...]
    by_margin: tuple[MarginRow, ...]


def fit_cascade(
    y: IntArrayLike,
    scores: FloatArrayLike,
    votes: Sequence[Sequence[int]],
    *,
    ids: Sequence[str],
    texts: Sequence[str],
    costs: CostModel = COST_MODEL,
    margin: float = LOW_MARGIN,
    margins: Sequence[float] = DEFAULT_MARGINS,
    forbidden: IntArrayLike | None = None,
    forbidden_label: str = "",
    skip_scores: FloatArrayLike | None = None,
    segmentation: Segmentation | None = None,
    scope: ScopeRule | None = None,
    adversarial: Sequence[tuple[str, float, str]] = (),
    adversarial_label: str = "",
) -> CascadeFit:
    if (skip_scores is None) != (segmentation is None):
        raise ValueError(
            "segment scores and the segmentation that produced them must be given together: "
            "the artifact records the segmentation, and a consumer that splits notes "
            "differently from the fit would serve a skip band nobody measured"
        )
    q = teacher_probabilities(votes)
    # `low` is fitted on the statistic the skip band compares, not on the
    # whole-note score. See the module docstring.
    worst = skip_statistic(scores, skip_scores)
    low_fit = fit_low(
        y, worst, ids=ids, texts=texts, margin=margin,
        adversarial=adversarial, adversarial_label=adversarial_label,
    )  # fmt: skip

    constraint: Constraint | None = None
    floor = 0.0
    if forbidden is not None:
        floor = constraint_floor(scores, forbidden)
        mask = np.asarray(forbidden, dtype=bool)
        blocked = np.flatnonzero(mask)
        highest = int(blocked[np.argmax(np.asarray(scores, dtype=float)[blocked])])
        constraint = Constraint(
            label=forbidden_label or "rows that may not reach support",
            floor=floor,
            rows=int(mask.sum()),
            set_by=ids[highest],
            set_by_text=texts[highest],
        )

    curve = cost_curve(y, scores, q, low=low_fit.threshold, floor=floor, skip_scores=worst)
    thresholds = Thresholds(low_fit.threshold, float(curve.candidates[curve.best_index(costs)]))

    leaked = skip_band_positives(y, worst, thresholds)
    if leaked:
        # Impossible while margin <= 1; checked anyway, because it is the one
        # property this whole construction exists to guarantee.
        raise ValueError(f"{len(leaked)} validation distress case(s) fall in the skip band")

    segments = envelope(curve)
    chosen = next(
        (n for n, s in enumerate(segments) if math.isclose(s.high, thresholds.high)),
        next(n for n, s in enumerate(segments) if s.contains(costs.ratio)),
    )
    alternative = segments[chosen - 1] if chosen > 0 else None
    decisive = (
        decisive_positives(
            y, scores, q, ids=ids, texts=texts,
            high=thresholds.high, alternative_high=alternative.high,
        )
        if alternative is not None
        else ()
    )  # fmt: skip

    by_margin: list[MarginRow] = []
    for m in margins:
        m_low = fit_low(
            y, worst, ids=ids, texts=texts, margin=m, adversarial=adversarial
        ).threshold  # fmt: skip
        outcome = simulate(
            y, scores, q, Thresholds(m_low, max(m_low, thresholds.high)),
            costs=costs, skip_scores=worst,
        )  # fmt: skip
        by_margin.append(MarginRow(m, m_low, outcome.skip_share))

    high = thresholds.high
    return CascadeFit(
        thresholds=thresholds,
        costs=costs,
        low=low_fit,
        cascade=simulate(y, scores, q, thresholds, costs=costs, name="cascade", skip_scores=worst),
        llm_alone=simulate(
            y, scores, q, Thresholds(0.0, 1.0), costs=costs, name="LLM alone (today)"
        ),
        model_alone=simulate(y, scores, q, Thresholds(high, high), costs=costs, name="model alone"),
        whole_note_cascade=(
            None
            if skip_scores is None
            else simulate(y, scores, q, thresholds, costs=costs, name="whole-note skip band")
        ),
        envelope=segments,
        constraint=constraint,
        segmentation=segmentation,
        scope=scope,
        chosen_segment=chosen,
        alternative=alternative,
        decisive=decisive,
        by_margin=tuple(by_margin),
    )
