"""The test set, spent once.

Two halves, kept apart on purpose.

**Scoring** — :func:`score_test_split` — is the only code in this repository that
reads a test row's text. It runs the shipped artifact over the test split, saves
one prediction per row, and records the look in the ledger *before* any number is
shown.

**Everything else** — metrics, intervals, targets, the error tables — reads the
saved predictions. A bug in the report is fixed by re-rendering with
``--render``, never by looking at the test set again.

What is measured, on which rows
-------------------------------
The shipped system is the cascade, so its recall depends on what
``claude-haiku-4-5`` says about the notes it escalates. Teacher votes exist for
the held-out *reviewed* rows; the golden rows were never candidates and have
none. So:

* the **model** is scored on every test row;
* the **cascade** is simulated on the rows that have votes, exactly as in the fit;
* on the **golden** rows the cascade's recall is reported as bounds. A golden
  distress case routed to support is caught whatever the LLM does; one that skips
  is missed whatever it does; one that escalates depends on a call not made here.

    uv run --extra embed python -m dc.evaluate            # score once, then render
    uv run python -m dc.evaluate --render                 # re-render; no new look
    uv run --extra embed python -m dc.evaluate --again "reason"
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from dc.baselines import KeywordBaseline
from dc.calibration import Bin, brier_score, expected_calibration_error, reliability_bins
from dc.cascade import CostModel, Outcome, Route, Thresholds, route, simulate
from dc.features import Embedder
from dc.metrics import CategoryScore, Scores, bootstrap_ci, per_category_columns, score
from dc.model import DistressModel
from dc.schema import Row
from dc.splits import Split, check_leakage, split_rows

__all__ = [
    "CascadeResult",
    "ErrorRow",
    "GoldenResult",
    "Note",
    "Prediction",
    "Target",
    "TestResults",
    "compute_results",
    "load_predictions",
    "render_markdown",
    "score_test_split",
    "write_plots",
    "write_predictions",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.jsonl"
#: The error analysis, written by a person after reading the errors: a short
#: summary, and a cause and one-line note per error. It documents the errors and
#: never feeds back into the model — which is why it lives in a data file, not in
#: code that could be mistaken for a rule.
NOTES_PATH = REPORT_DIR / "error_notes.json"

TARGET_RECALL = 0.98
TARGET_PR_AUC = 0.90
SEED = 0

Status = Literal["pass", "fail", "conditional"]
Confirmed = Literal["yes", "no", "n/a"]


def zero_miss_lower_bound(positives: int) -> float:
    """One-sided 95% lower bound on recall when every one of ``positives`` was caught.

    A percentile bootstrap cannot give this: if nothing was missed, every resample
    misses nothing too, and the "interval" collapses to [1.00, 1.00] — a claim of
    certainty the sample does not support.
    """
    return 0.05 ** (1.0 / positives) if positives else 0.0


@dataclass(frozen=True)
class Note:
    cause: str
    note: str


@dataclass(frozen=True)
class Analysis:
    summary: tuple[str, ...]
    errors: dict[str, Note]


@dataclass(frozen=True)
class Prediction:
    id: str
    category: str
    label: int
    golden: bool
    feeling: str
    text: str
    score: float
    keyword: float
    route: str
    teacher_votes: tuple[int, ...] = ()

    @property
    def teacher(self) -> float | None:
        """Share of teacher votes for distress, or None for rows never voted on."""
        if not self.teacher_votes:
            return None
        return sum(self.teacher_votes) / len(self.teacher_votes)


# --- scoring: the only code that reads test rows ----------------------------------


def score_test_split(
    rows: Sequence[Row],
    assignment: Mapping[str, Split],
    model: DistressModel,
    embedder: Embedder,
    thresholds: Thresholds,
    teacher_votes: Mapping[str, Sequence[int]],
) -> list[Prediction]:
    problems = check_leakage(list(rows), dict(assignment))
    if problems:
        raise ValueError("the split is unsound:\n  " + "\n  ".join(problems))
    test = split_rows(list(rows), dict(assignment), "test")
    if not test:
        raise ValueError("the test split is empty")
    scores = model.predict_proba(test, embedder)
    keyword = KeywordBaseline().predict_proba(test)
    return [
        Prediction(
            id=row.id,
            category=row.category.value,
            label=row.label,
            golden=row.is_golden,
            feeling=row.feeling.value,
            text=row.free_text,
            score=float(s),
            keyword=float(k),
            route=route(float(s), thresholds).value,
            teacher_votes=tuple(int(v) for v in teacher_votes.get(row.id, ())),
        )
        for row, s, k in zip(test, scores, keyword, strict=True)
    ]


def write_predictions(predictions: Sequence[Prediction], path: Path = PREDICTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({**asdict(p), "teacher_votes": list(p.teacher_votes)}, ensure_ascii=False)
        for p in predictions
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_predictions(path: Path = PREDICTIONS_PATH) -> list[Prediction]:
    out: list[Prediction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = cast("dict[str, Any]", json.loads(line))
        raw["teacher_votes"] = tuple(int(v) for v in cast("list[Any]", raw["teacher_votes"]))
        out.append(Prediction(**raw))
    return out


def load_analysis(path: Path = NOTES_PATH) -> Analysis:
    if not path.exists():
        return Analysis((), {})
    raw = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    errors = {
        str(row_id): Note(str(entry.get("cause", "")), str(entry.get("note", "")))
        for row_id, entry in cast("dict[str, dict[str, Any]]", raw.get("errors", {})).items()
    }
    summary = tuple(str(line) for line in cast("list[Any]", raw.get("summary", [])))
    return Analysis(summary, errors)


# --- results: computed from saved predictions only ----------------------------------


@dataclass(frozen=True)
class Target:
    name: str
    requirement: str
    result: str
    #: Whether the point estimate meets the requirement.
    status: Status
    #: Whether the data *confirms* it at 95% — a different and harder question.
    confirmed: Confirmed
    why: str


@dataclass(frozen=True)
class GoldenResult:
    n: int
    positives: tuple[Prediction, ...]
    supported: int
    escalated: int
    skipped: int
    false_alarms: tuple[Prediction, ...]

    @property
    def recall_bounds(self) -> tuple[float, float]:
        """(if the LLM misses every escalated case, if it catches every one)."""
        n = len(self.positives)
        return (self.supported / n, (self.supported + self.escalated) / n) if n else (0.0, 0.0)


@dataclass(frozen=True)
class CascadeResult:
    n: int
    positives: int
    cascade: Outcome
    llm_alone: Outcome
    teacher: Scores
    model: Scores
    recall_ci: tuple[float, float] | None
    difference_ci: tuple[float, float] | None


@dataclass(frozen=True)
class ErrorRow:
    kind: Literal["skipped distress", "escalated distress", "false alarm"]
    prediction: Prediction
    #: 1.0 for a skipped case; 1 - teacher share for an escalated one; None when
    #: the case was never voted on and the LLM's answer is unknown.
    expected_miss: float | None
    cause: str
    diagnosis: str


@dataclass(frozen=True)
class TestResults:
    n: int
    positives: int
    thresholds: Thresholds
    model: Scores
    keyword: Scores
    accuracy: float
    brier: float
    ece: float
    bins: tuple[Bin, ...]
    per_category: tuple[CategoryScore, ...]
    route_shares: dict[str, float]
    cascade: CascadeResult
    golden: GoldenResult
    errors: tuple[ErrorRow, ...]
    targets: tuple[Target, ...]


def _flags(scores: np.ndarray, teacher: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """P(routed to support) per row under the cascade — route() in vector form."""
    return np.where(scores > thresholds.high, 1.0, np.where(scores < thresholds.low, 0.0, teacher))


def _paired_difference_ci(
    truth: np.ndarray, a: np.ndarray, b: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float] | None:
    """95% interval on recall(a) - recall(b), resampling the same rows for both."""
    if n_bootstrap <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, truth.size, truth.size)
        t = truth[index]
        positives = float(np.sum(t))
        if positives == 0:
            continue
        draws.append(float(np.sum(t * a[index]) - np.sum(t * b[index])) / positives)
    if not draws:
        return None
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _cascade_result(
    predictions: Sequence[Prediction],
    thresholds: Thresholds,
    costs: CostModel,
    *,
    n_bootstrap: int,
    seed: int,
) -> CascadeResult:
    voted = [p for p in predictions if p.teacher_votes]
    if not voted or not any(p.label for p in voted):
        raise ValueError("no test rows with teacher votes and a positive label")
    y = np.array([p.label for p in voted])
    s = np.array([p.score for p in voted])
    q = np.array([p.teacher for p in voted], dtype=float)
    cascade_flags = _flags(s, q, thresholds)
    return CascadeResult(
        n=len(voted),
        positives=int(np.sum(y)),
        cascade=simulate(y, s, q, thresholds, costs=costs, name="cascade"),
        llm_alone=simulate(y, s, q, Thresholds(0.0, 1.0), costs=costs, name="LLM alone (today)"),
        teacher=score(y, q, threshold=0.5, n_bootstrap=n_bootstrap, seed=seed),
        model=score(y, s, threshold=thresholds.high, n_bootstrap=n_bootstrap, seed=seed),
        recall_ci=bootstrap_ci(
            y,
            cascade_flags,
            lambda t, f: float(np.sum(t * f) / np.sum(t)),
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        difference_ci=_paired_difference_ci(
            y.astype(float), cascade_flags, q, n_bootstrap=n_bootstrap, seed=seed
        ),
    )


def _golden_result(predictions: Sequence[Prediction]) -> GoldenResult:
    golden = [p for p in predictions if p.golden]
    positives = tuple(p for p in golden if p.label == 1)
    counts = Counter(p.route for p in positives)
    return GoldenResult(
        n=len(golden),
        positives=positives,
        supported=counts[Route.SUPPORT.value],
        escalated=counts[Route.ESCALATE.value],
        skipped=counts[Route.SKIP_LLM.value],
        false_alarms=tuple(p for p in golden if p.label == 0 and p.route == Route.SUPPORT.value),
    )


def _errors(predictions: Sequence[Prediction], notes: Mapping[str, Note]) -> tuple[ErrorRow, ...]:
    blank = Note("", "")
    rows: list[ErrorRow] = []
    for p in predictions:
        note = notes.get(p.id, blank)
        if p.label == 1 and p.route == Route.SKIP_LLM.value:
            rows.append(ErrorRow("skipped distress", p, 1.0, note.cause, note.note))
        elif p.label == 1 and p.route == Route.ESCALATE.value:
            share = p.teacher
            if share is None or share < 1.0:
                miss = None if share is None else 1.0 - share
                rows.append(ErrorRow("escalated distress", p, miss, note.cause, note.note))
        elif p.label == 0 and p.route == Route.SUPPORT.value:
            rows.append(ErrorRow("false alarm", p, None, note.cause, note.note))
    order = {"skipped distress": 0, "escalated distress": 1, "false alarm": 2}
    return tuple(
        sorted(
            rows, key=lambda e: (order[e.kind], e.cause, e.prediction.category, e.prediction.score)
        )
    )


def _targets(results: dict[str, Any]) -> tuple[Target, ...]:
    golden: GoldenResult = results["golden"]
    cascade: CascadeResult = results["cascade"]
    model: Scores = results["model"]
    keyword: Scores = results["keyword"]
    targets: list[Target] = []

    n_golden = len(golden.positives)
    gate_why = (
        f"a release gate over {n_golden} cases, not an estimate: {n_golden} of {n_golden} "
        f"is consistent with a true recall as low as {zero_miss_lower_bound(n_golden):.2f}"
    )
    if golden.skipped:
        status: Status = "fail"
        result = f"{golden.skipped} skip the LLM"
    elif golden.escalated:
        status = "conditional"
        result = f"{golden.supported} to support, {golden.escalated} escalate (LLM not measured)"
    else:
        status = "pass"
        result = f"{golden.supported} of {n_golden} to support, without the LLM"
    targets.append(Target("Golden distress cases caught", f"{n_golden} of {n_golden}", result,
                          status, "n/a", gate_why))  # fmt: skip

    c = cascade.cascade
    if c.expected_missed < 1e-9:
        lower = zero_miss_lower_bound(cascade.positives)
        why = f"no misses among {cascade.positives}; 95% lower bound {lower:.3f}"
    else:
        lower = cascade.recall_ci[0] if cascade.recall_ci else 0.0
        why = f"95% bootstrap lower bound {lower:.3f}"
    targets.append(
        Target(
            "Cascade recall on distress", f"≥ {TARGET_RECALL:.2f}", f"{c.expected_recall:.3f}",
            "pass" if c.expected_recall >= TARGET_RECALL else "fail",
            "yes" if lower >= TARGET_RECALL else "no",
            f"{why}; over the {cascade.n} test rows with teacher votes",
        )
    )  # fmt: skip

    pr = model.pr_auc_ci
    targets.append(
        Target(
            "Model PR-AUC", f"≥ {TARGET_PR_AUC:.2f}", f"{model.pr_auc:.3f}",
            "pass" if model.pr_auc >= TARGET_PR_AUC else "fail",
            "n/a" if pr is None else ("yes" if pr[0] >= TARGET_PR_AUC else "no"),
            "" if pr is None else f"95% interval {pr[0]:.2f}–{pr[1]:.2f}; all {model.n} test rows",
        )
    )  # fmt: skip

    kpr = keyword.pr_auc_ci
    separated = pr is not None and kpr is not None and pr[0] > kpr[1]
    targets.append(
        Target(
            "Beats the crisis-keyword rule", "PR-AUC above it",
            f"{model.pr_auc:.3f} vs {keyword.pr_auc:.3f}",
            "pass" if model.pr_auc > keyword.pr_auc else "fail",
            "n/a" if pr is None or kpr is None else ("yes" if separated else "no"),
            "the two 95% intervals do not overlap" if separated else "the 95% intervals overlap",
        )
    )  # fmt: skip

    diff = c.expected_recall - cascade.llm_alone.expected_recall
    dci = cascade.difference_ci
    both_perfect = c.expected_missed < 1e-9 and cascade.llm_alone.expected_missed < 1e-9
    if both_perfect:
        confirmed: Confirmed = "n/a"
        why = "both caught every case, so this check cannot tell them apart on these rows"
    else:
        confirmed = "n/a" if dci is None else ("yes" if dci[0] >= -1e-9 else "no")
        why = (
            "" if dci is None else f"95% interval on the difference {dci[0]:+.2f} to {dci[1]:+.2f}"
        )
    targets.append(
        Target(
            "No recall regression against the LLM alone", "cascade ≥ LLM",
            f"{c.expected_recall:.3f} vs {cascade.llm_alone.expected_recall:.3f}",
            "pass" if diff >= -1e-9 else "fail", confirmed, why,
        )
    )  # fmt: skip
    return tuple(targets)


def compute_results(
    predictions: Sequence[Prediction],
    thresholds: Thresholds,
    *,
    costs: CostModel,
    notes: Mapping[str, Note] | None = None,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> TestResults:
    if not predictions:
        raise ValueError("no predictions to evaluate")
    y = np.array([p.label for p in predictions])
    s = np.array([p.score for p in predictions])
    k = np.array([p.keyword for p in predictions])

    parts: dict[str, Any] = {
        "model": score(y, s, threshold=thresholds.high, n_bootstrap=n_bootstrap, seed=seed),
        "keyword": score(y, k, threshold=0.5, n_bootstrap=n_bootstrap, seed=seed),
        "cascade": _cascade_result(
            predictions, thresholds, costs, n_bootstrap=n_bootstrap, seed=seed
        ),
        "golden": _golden_result(predictions),
    }
    shares = Counter(p.route for p in predictions)
    return TestResults(
        n=len(predictions),
        positives=int(np.sum(y)),
        thresholds=thresholds,
        model=parts["model"],
        keyword=parts["keyword"],
        accuracy=float(np.mean((s >= thresholds.high).astype(int) == y)),
        brier=brier_score(y, s),
        ece=expected_calibration_error(y, s),
        bins=tuple(reliability_bins(y, s)),
        per_category=tuple(
            per_category_columns(
                [p.id for p in predictions],
                [p.label for p in predictions],
                [p.category for p in predictions],
                s,
                threshold=thresholds.high,
            )
        ),
        route_shares={r.value: shares[r.value] / len(predictions) for r in Route},
        cascade=parts["cascade"],
        golden=parts["golden"],
        errors=_errors(predictions, notes or {}),
        targets=_targets(parts),
    )


# --- rendering ------------------------------------------------------------------------

_STATUS = {"pass": "✅ pass", "fail": "❌ fail", "conditional": "⚠️ conditional"}
_CONFIRMED = {"yes": "yes", "no": "**no**", "n/a": "—"}


def _ci(bounds: tuple[float, float] | None) -> str:
    """A bootstrap interval, unless it has collapsed to a point.

    A zero-width interval means no resample could move the statistic — every one
    reproduced the same perfect result — which is a property of the sample, not
    evidence of certainty. It is labelled as such rather than printed as a range.
    """
    if bounds is None:
        return ""
    if abs(bounds[1] - bounds[0]) < 1e-12:
        return " (no resampling variation)"
    return f" [{bounds[0]:.2f}–{bounds[1]:.2f}]"


def _recall(value: float, missed: float, positives: int, bounds: tuple[float, float] | None) -> str:
    """A recall with an honest interval: the zero-event bound when nothing was missed."""
    if missed < 1e-9 and positives:
        return f"{value:.3f} (≥ {zero_miss_lower_bound(positives):.3f})"
    return f"{value:.3f}{_ci(bounds)}"


def render_markdown(
    results: TestResults,
    *,
    looks: Sequence[Any],
    validation: Mapping[str, Any],
    cv_pr_auc: float | None,
    renders: Sequence[Any] = (),
    summary: Sequence[str] = (),
) -> str:
    r = results
    c = r.cascade
    lines = [
        "# Test set evaluation",
        "",
        f"{r.n} test rows, {r.positives} distress: {r.golden.n} golden release-gate cases "
        f"({len(r.golden.positives)} distress) and {r.n - r.golden.n} held-out reviewed rows. "
        f"{c.n} of them have teacher votes.",
        "",
    ]
    if len(looks) <= 1:
        lines.append("**This is the first and only look at the test set.** "
                     "`reports/test_ledger.json` records it.")  # fmt: skip
        if looks and renders and renders[-1].evaluator_sha256 != looks[-1].evaluator_sha256:
            lines += ["", "The report has since been re-rendered from the saved predictions by "
                      "evaluation code that changed after the look. No test row was read again; "
                      "every render, with the code's hash and a note on what changed, is in the "
                      "ledger."]  # fmt: skip
    else:
        lines.append(f"**⚠️ The test set has been evaluated {len(looks)} times.** Every look, "
                     "with its reason, is in `reports/test_ledger.json`.")  # fmt: skip
    lines += ["", "## Targets", "",
              "`Status` asks whether the point estimate meets the requirement. `Holds at 95%` "
              "asks whether the data confirms it — a harder question, and on sets this small "
              "often a different answer.", "",
              "| Target | Required | Result | Status | Holds at 95% | |",
              "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for target in r.targets:
        lines.append(
            f"| {target.name} | {target.requirement} | {target.result} "
            f"| {_STATUS[target.status]} | {_CONFIRMED[target.confirmed]} | {target.why} |"
        )

    lines += [
        "",
        "## The model",
        "",
        f"Every test row. Recall and precision at `high` ({r.thresholds.high:.3f}) — the model "
        "deciding alone, with no LLM. PR-AUC needs no threshold. Intervals are 95% bootstrap; "
        "where nothing was missed, recall shows its one-sided 95% lower bound instead, "
        "because a bootstrap over zero misses collapses to a zero-width interval that claims "
        "a certainty the sample does not have.",
        "",
        "| | PR-AUC | Recall | Precision | F1 | Missed |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| **Model** | {r.model.pr_auc:.3f}{_ci(r.model.pr_auc_ci)} "
        f"| {_recall(r.model.recall, r.model.fn, r.model.positives, r.model.recall_ci)} "
        f"| {r.model.precision:.3f} "
        f"| {r.model.f1:.3f} | {r.model.fn} |",
        f"| Crisis-keyword rule | {r.keyword.pr_auc:.3f}{_ci(r.keyword.pr_auc_ci)} "
        f"| {r.keyword.recall:.3f} | {r.keyword.precision:.3f} | {r.keyword.f1:.3f} "
        f"| {r.keyword.fn} |",
    ]
    if cv_pr_auc is not None:
        lines += [
            "",
            f"Cross-validation estimated PR-AUC at **{cv_pr_auc:.3f}**; the test set gives "
            f"**{r.model.pr_auc:.3f}** ({r.model.pr_auc - cv_pr_auc:+.3f}).",
        ]

    llm, cascade = c.llm_alone, c.cascade
    llm_recall = _recall(llm.expected_recall, llm.expected_missed, c.positives, None)
    cascade_recall = _recall(
        cascade.expected_recall, cascade.expected_missed, c.positives, c.recall_ci
    )
    lines += [
        "",
        "## The cascade",
        "",
        f"The {c.n} test rows with teacher votes ({c.positives} distress). Expected values, with "
        "the LLM modelled as one call that says distress with the share of its recorded votes.",
        "",
        "| | Expected recall | Expected missed | Expected false alarms | LLM calls | Support |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| LLM alone (today) | {llm_recall} | {llm.expected_missed:.1f} "
        f"| {llm.expected_false_positives:.1f} | 100% | 0% |",
        f"| **Cascade** | {cascade_recall} | {cascade.expected_missed:.1f} "
        f"| {cascade.expected_false_positives:.1f} | {cascade.escalate_share:.0%} "
        f"| {cascade.support_share:.0%} |",
    ]
    if validation:
        lines += [
            "",
            "On validation — the same rows that fitted the thresholds — the cascade showed "
            f"recall **{float(validation['cascade_expected_recall']):.3f}**, sent "
            f"**{float(validation['support_share']):.0%}** of notes to support and escalated "
            f"**{float(validation['escalate_share']):.0%}**. On test: recall "
            f"**{c.cascade.expected_recall:.3f}**, support **{c.cascade.support_share:.0%}**, "
            f"escalate **{c.cascade.escalate_share:.0%}**.",
        ]

    g = r.golden
    low, high = g.recall_bounds
    lines += [
        "",
        "## The golden cases",
        "",
        f"{len(g.positives)} golden distress cases: **{g.supported}** routed to support, "
        f"**{g.escalated}** escalated, **{g.skipped}** skipped the LLM. Recall is between "
        f"**{low:.2f}** (if the LLM misses every escalated case) and **{high:.2f}** (if it "
        "catches every one).",
        "",
        "| Row | Score | Route | Note |",
        "| --- | --- | --- | --- |",
    ]
    for p in sorted(g.positives, key=lambda p: p.score):
        lines.append(f"| `{p.id}` | {p.score:.3f} | `{p.route}` | {p.text} |")
    if g.false_alarms:
        lines += ["", f"{len(g.false_alarms)} golden non-distress case(s) routed to support:", "",
                  "| Row | Category | Score | Note |", "| --- | --- | --- | --- |"]  # fmt: skip
        for p in g.false_alarms:
            lines.append(f"| `{p.id}` | {p.category} | {p.score:.3f} | {p.text} |")

    lines += [
        "",
        "## Where the model's errors are",
        "",
        f"At `high` ({r.thresholds.high:.3f}), without the LLM. `R` is recall on a positive "
        "category, `FP` the false-positive rate on a negative one.",
        "",
        "| Category | Rows | Result |",
        "| --- | --- | --- |",
    ]
    for cat in r.per_category:
        cell = (
            f"R {cat.recall:.2f}" if cat.recall is not None else f"FP {cat.false_positive_rate:.2f}"
        )
        lines.append(f"| {cat.category} | {cat.n} | {cell} |")

    if r.errors:
        causes = Counter(e.cause or "not yet diagnosed" for e in r.errors)
        lines += ["", "## Error analysis", ""]
        for paragraph in summary:
            lines += [paragraph, ""]
        lines += ["| Cause | Errors |", "| --- | --- |"]
        for cause, count in causes.most_common():
            lines.append(f"| {cause} | {count} |")

    kinds = {
        "skipped distress": "Distress cases that skip the LLM — missed, whatever it would say",
        "escalated distress": "Distress cases escalated to an LLM that may not flag them",
        "false alarm": "Non-distress notes routed straight to support",
    }
    lines += ["", "## Every error", "",
              "Listed verbatim. Diagnoses are written after reading the errors and do not feed "
              "back into the model.", ""]  # fmt: skip
    for kind, heading in kinds.items():
        group = [e for e in r.errors if e.kind == kind]
        lines += [f"### {heading} ({len(group)})", ""]
        if not group:
            lines += ["None.", ""]
            continue
        lines += ["| Row | Category | Score | LLM miss | Note | Cause | Diagnosis |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
        for e in group:
            miss = "—" if e.expected_miss is None else f"{e.expected_miss:.0%}"
            if kind == "escalated distress" and e.expected_miss is None:
                miss = "unknown"
            p = e.prediction
            lines.append(
                f"| `{p.id}` | {p.category} | {p.score:.3f} | {miss} | {p.text} "
                f"| {e.cause or '—'} | {e.diagnosis or '_not yet diagnosed_'} |"
            )
        lines.append("")

    lines += [
        "## Curves",
        "",
        f"The precision–recall chart scores all three on the {c.n} rows with teacher votes, so "
        f"they are compared on the same rows. There the model's PR-AUC is "
        f"{c.model.pr_auc:.3f} and the teacher's {c.teacher.pr_auc:.3f} — which is why the "
        f"model's figure differs from the {r.model.pr_auc:.3f} above, taken over all {r.n}.",
        "",
        "![Precision–recall on the test set](plots/pr_curve.svg)",
        "",
        "![Reliability of the model's scores](plots/reliability.svg)",
        "",
        "## Calibration on test",
        "",
        f"Brier **{r.brier:.3f}** · expected calibration error **{r.ece:.3f}**",
        "",
        "| Score bin | Rows | Mean predicted | Observed distress |",
        "| --- | --- | --- | --- |",
    ]
    for b in r.bins:
        if b.n:
            lines.append(
                f"| {b.lower:.1f}–{b.upper:.1f} | {b.n} | {b.mean_predicted:.2f} "
                f"| {b.observed_rate:.2f} |"
            )

    lines += [
        "",
        "## How to read these numbers",
        "",
        "**The test set holds out whole families.** Splits are grouped by `origin_id`, so a "
        "test row's seed and every paraphrase of it are test too. These are held-out *modes*, "
        "not rephrasings of modes the model has seen — a harder test than a random split.",
        "",
        "**Most of the data came from a generator.** A model can learn a generator's habits. "
        "Expect real player text to be harder than this.",
        "",
        "**The teacher's numbers are partly definitional.** The reviewer saw its votes while "
        "labelling, so every overruled vote counts against it by construction.",
        "",
        "**The LLM's behaviour on the golden cases is not measured here.** Those rows have no "
        "recorded votes, so any golden case that escalates is reported as a bound, not a number.",
        "",
    ]
    return "\n".join(lines)


def _pr_points(labels: np.ndarray, scores: np.ndarray) -> list[tuple[float, float]]:
    """(recall, precision) along the curve, in threshold order from high recall's far end.

    scikit-learn returns recall decreasing; reversed, the points trace the curve as
    the threshold falls. Order matters: the curve can revisit a recall value.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    return [(float(r), float(p)) for r, p in zip(recall[::-1], precision[::-1], strict=True)]


def write_plots(
    predictions: Sequence[Prediction], results: TestResults, directory: Path
) -> list[Path]:
    """The PR curve (on rows with teacher votes, so all three are comparable) and
    the reliability diagram (every test row)."""
    from dc.metrics import pr_auc
    from dc.plots import Series, line_chart_svg

    directory.mkdir(parents=True, exist_ok=True)
    voted = [p for p in predictions if p.teacher_votes]
    y = np.array([p.label for p in voted])
    curves = {
        "Model": np.array([p.score for p in voted]),
        "Teacher": np.array([p.teacher for p in voted], dtype=float),
        "Keyword": np.array([p.keyword for p in voted]),
    }
    series: list[Series] = []
    for slot, (name, scores) in enumerate(curves.items(), start=1):
        points = _pr_points(y, scores)
        label = f"{name} {pr_auc(y, scores):.2f}"
        series.append(
            Series(
                name=label,
                points=points,
                slot=slot,
                markers=name != "Model",
                step=True,
            )
        )
    pr_path = directory / "pr_curve.svg"
    pr_path.write_text(
        line_chart_svg(
            title="Precision–recall on the test set",
            subtitle=f"{len(voted)} held-out rows with teacher votes · PR-AUC beside each name",
            x_label="Recall",
            y_label="Precision",
            series=series,
        ),
        encoding="utf-8",
    )

    populated = [b for b in results.bins if b.n and b.mean_predicted is not None]
    reliability_path = directory / "reliability.svg"
    reliability_path.write_text(
        line_chart_svg(
            title="Reliability of the model's scores",
            subtitle=f"All {results.n} test rows · mean predicted score against observed "
            "distress rate, per score bin",
            x_label="Mean predicted score",
            y_label="Observed distress rate",
            series=[
                Series(
                    name="Model",
                    points=[
                        (float(b.mean_predicted or 0.0), float(b.observed_rate or 0.0))
                        for b in populated
                    ],
                    slot=1,
                    markers=True,
                )
            ],  # fmt: skip
            diagonal_label="perfectly calibrated",
        ),
        encoding="utf-8",
    )
    return [pr_path, reliability_path]


def results_json(results: TestResults, looks: Sequence[Any]) -> dict[str, Any]:
    r = results
    return {
        "looks": len(looks),
        "n": r.n,
        "positives": r.positives,
        "thresholds": asdict(r.thresholds),
        "targets": [asdict(t) for t in r.targets],
        "model": asdict(r.model),
        "keyword": asdict(r.keyword),
        "accuracy_at_high": r.accuracy,
        "calibration": {"brier": r.brier, "ece": r.ece, "bins": [asdict(b) for b in r.bins]},
        "per_category": [asdict(c) for c in r.per_category],
        "route_shares": r.route_shares,
        "cascade": {
            "n": r.cascade.n,
            "positives": r.cascade.positives,
            "cascade": asdict(r.cascade.cascade),
            "llm_alone": asdict(r.cascade.llm_alone),
            "teacher": asdict(r.cascade.teacher),
            "model": asdict(r.cascade.model),
            "recall_ci": r.cascade.recall_ci,
            "difference_ci": r.cascade.difference_ci,
        },
        "golden": {
            "n": r.golden.n,
            "positives": len(r.golden.positives),
            "supported": r.golden.supported,
            "escalated": r.golden.escalated,
            "skipped": r.golden.skipped,
            "recall_bounds": r.golden.recall_bounds,
            "false_alarms": [p.id for p in r.golden.false_alarms],
        },
        "errors": [
            {
                "kind": e.kind,
                "id": e.prediction.id,
                "expected_miss": e.expected_miss,
                "cause": e.cause,
                "diagnosis": e.diagnosis,
            }  # fmt: skip
            for e in r.errors
        ],
    }


# --- command ------------------------------------------------------------------------


def main() -> None:
    from dc.artifact import ARTIFACT_DIR, dataset_sha256, is_servable, load, source_commit
    from dc.ledger import (
        EVALUATOR_FILES,
        MEASURED_INPUTS,
        LedgerError,
        Look,
        Render,
        append_look,
        append_render,
        authorize,
        files_sha256,
        read_looks,
        read_renders,
        today,
        uncommitted,
    )
    from dc.splits import DATASET_PATH, SEED_PATH, SPLITS_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render", action="store_true", help="re-render; does not touch test rows")
    mode.add_argument("--again", metavar="REASON", help="look again, recording why")
    parser.add_argument(
        "--note", default="re-rendered from saved predictions", help="recorded with a --render"
    )
    args = parser.parse_args()

    artifact = load()
    if not is_servable(artifact.metadata):
        raise SystemExit("The artifact has no thresholds. Run `make train` first.")
    bands = cast("dict[str, Any]", artifact.metadata["thresholds"])
    thresholds = Thresholds(float(bands["low"]), float(bands["high"]))
    costs = CostModel(
        float(bands["cost_model"]["false_negative"]), float(bands["cost_model"]["false_positive"])
    )
    artifact_hash = files_sha256([ARTIFACT_DIR / "model.npz", ARTIFACT_DIR / "model.json"])

    looks = read_looks()
    if not args.render:
        dirty = uncommitted(MEASURED_INPUTS)
        if dirty:
            raise SystemExit(
                "Commit artifacts/ and data/ before spending the test set, so what was "
                "measured can be recovered:\n  " + "\n  ".join(dirty)
            )
        try:
            reason = authorize(looks, again=args.again)
        except LedgerError as exc:
            raise SystemExit(str(exc)) from exc

        from dc.candidates import load_teacher_votes
        from dc.features import load_embedder
        from dc.splits import load_all_rows, load_assignment

        predictions = score_test_split(
            load_all_rows(), load_assignment(), artifact.model, load_embedder(), thresholds,
            load_teacher_votes(),
        )  # fmt: skip
        # Recorded before the predictions are saved and before a single number is
        # computed or shown: a crash from here on is still a look.
        append_look(
            Look(
                number=len(looks) + 1,
                on=today(),
                commit=source_commit()["commit"],
                artifact_sha256=artifact_hash,
                dataset_sha256=dataset_sha256((SEED_PATH, DATASET_PATH, SPLITS_PATH)),
                evaluator_sha256=files_sha256([REPO_ROOT / f for f in EVALUATOR_FILES]),
                reason=reason,
            )
        )
        write_predictions(predictions)
        looks = read_looks()
    elif not PREDICTIONS_PATH.exists() or not looks:
        raise SystemExit("No saved predictions. Run `make evaluate` to score the test set once.")
    elif looks[-1].artifact_sha256 != artifact_hash:
        raise SystemExit(
            "The artifact has changed since the test set was scored, so the saved predictions "
            'describe a different model. Look again with `--again "<reason>"`.'
        )
    else:
        append_render(
            Render(
                on=today(),
                predictions_sha256=files_sha256([PREDICTIONS_PATH]),
                artifact_sha256=artifact_hash,
                evaluator_sha256=files_sha256([REPO_ROOT / f for f in EVALUATOR_FILES]),
                note=args.note,
            )
        )

    predictions = load_predictions()
    analysis = load_analysis()
    results = compute_results(predictions, thresholds, costs=costs, notes=analysis.errors)
    selection = cast("dict[str, Any]", artifact.metadata.get("selection") or {})
    markdown = render_markdown(
        results,
        looks=looks,
        validation=cast("dict[str, Any]", bands.get("validation") or {}),
        cv_pr_auc=selection.get("cv_pr_auc_mean"),
        renders=read_renders(),
        summary=analysis.summary,
    )
    REPORT_DIR.mkdir(exist_ok=True)
    write_plots(predictions, results, REPORT_DIR / "plots")
    (REPORT_DIR / "test.md").write_text(markdown, encoding="utf-8")
    (REPORT_DIR / "test_results.json").write_text(
        json.dumps(results_json(results, looks), indent=2) + "\n", encoding="utf-8"
    )
    print(markdown)


if __name__ == "__main__":
    main()
