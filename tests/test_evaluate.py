"""The evaluator: what it reads, what it reports, and what it refuses."""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import make_row

from dc.cascade import CostModel, Thresholds, route
from dc.evaluate import (
    Note,
    Prediction,
    compute_results,
    load_analysis,
    load_predictions,
    render_markdown,
    results_json,
    score_test_split,
    write_predictions,
    zero_miss_lower_bound,
)
from dc.features import EMBED_MODEL, EMBED_MODEL_REVISION
from dc.model import DistressModel, FeatureLayout
from dc.schema import Category, Provenance, Row
from dc.splits import build_assignment
from dc.text import SCOPE, SEGMENTATION

REPO = Path(__file__).resolve().parents[1]
BANDS = Thresholds(low=0.1, high=0.8)
COSTS = CostModel(20.0, 1.0)


def pred(
    row_id: str,
    label: int,
    score: float,
    *,
    golden: bool = False,
    votes: tuple[int, ...] = (1, 1, 1),
    category: str | None = None,
) -> Prediction:
    return Prediction(
        id=row_id,
        category=category or ("distress" if label else "game-frustration"),
        label=label,
        golden=golden,
        feeling="custom",
        text=f"note {row_id}",
        score=score,
        keyword=0.0,
        route=route(score, BANDS).value,
        teacher_votes=() if golden else votes,
    )


def base() -> list[Prediction]:
    """Ten golden distress cases routed to support, plus reviewed rows with votes."""
    rows = [pred(f"g{i}", 1, 0.95, golden=True) for i in range(10)]
    rows += [pred(f"gn{i}", 0, 0.02, golden=True, category="nonsense") for i in range(10)]
    rows += [pred(f"p{i}", 1, 0.9) for i in range(10)]
    rows += [pred(f"n{i}", 0, 0.05, votes=(0, 0, 0)) for i in range(20)]
    return rows


def target(predictions: Sequence[Prediction], name: str) -> str:
    outcome = compute_results(predictions, BANDS, costs=COSTS, n_bootstrap=50)
    return next(t.status for t in outcome.targets if t.name.startswith(name))


# --- what the evaluator reads ----------------------------------------------------------


class SpyEmbedder:
    dim = 2

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        return np.array([[1.0 if "life" in t else 0.0, 0.0] for t in texts], dtype=np.float32)


def corpus() -> list[Row]:
    rows = [
        make_row(f"g{i}", category=Category.DISTRESS, free_text=f"golden life {i}",
                 provenance=Provenance.SEED)
        for i in range(10)
    ]  # fmt: skip
    rows += [
        make_row(f"p{i}", category=Category.DISTRESS, free_text=f"life {i}") for i in range(30)
    ]
    rows += [
        make_row(f"n{i}", category=Category.NONSENSE, free_text=f"game {i}") for i in range(60)
    ]
    return rows


def test_scoring_reads_only_test_rows() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    spy = SpyEmbedder()
    layout = FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, 2, False)
    model = DistressModel(coef=np.array([6.0, 0.0]), intercept=-3.0, layout=layout)

    predictions = score_test_split(rows, assignment, model, spy, BANDS, {}, SEGMENTATION, SCOPE)

    test_texts = {r.free_text for r in rows if assignment[r.id] == "test"}
    train_texts = {r.free_text for r in rows if assignment[r.id] == "train"}
    # Segment scoring embeds each note's parts as well as the note, so the seen
    # texts are the segments of test rows — never anything from another row.
    allowed = {segment for text in test_texts for segment in SEGMENTATION.split(text)}
    assert test_texts <= set(spy.seen) <= allowed
    assert not train_texts & set(spy.seen)
    assert {p.id for p in predictions} == {r.id for r in rows if assignment[r.id] == "test"}


def test_scoring_refuses_an_unsound_split() -> None:
    rows = corpus()
    assignment = build_assignment(rows)
    assignment["g0"] = "train"
    layout = FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, 2, False)
    model = DistressModel(coef=np.zeros(2), intercept=0.0, layout=layout)
    with pytest.raises(ValueError, match="unsound"):
        score_test_split(rows, assignment, model, SpyEmbedder(), BANDS, {}, SEGMENTATION, SCOPE)


def test_only_the_evaluator_ever_asks_for_test_rows() -> None:
    """Structural: nothing else in src/ or scripts/ calls split_rows(..., "test")."""
    offenders: list[str] = []
    for path in [*REPO.glob("src/dc/*.py"), *REPO.glob("scripts/*.py")]:
        if path.name == "evaluate.py":
            continue
        if re.search(r"split_rows\([^)]*[\"']test[\"']", path.read_text()):
            offenders.append(path.name)
    assert offenders == []


def test_training_never_imports_the_evaluator() -> None:
    tree = ast.parse((REPO / "src" / "dc" / "train.py").read_text())
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "dc.evaluate" not in imported and "dc.ledger" not in imported


def test_predictions_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    original = base()
    write_predictions(original, path)
    assert load_predictions(path) == original


# --- the targets -------------------------------------------------------------------------


def test_golden_passes_when_every_case_goes_to_support() -> None:
    assert target(base(), "Golden") == "pass"


def test_golden_is_conditional_when_a_case_escalates() -> None:
    rows = base()
    rows[0] = pred("g0", 1, 0.5, golden=True)
    assert target(rows, "Golden") == "conditional"


def test_golden_fails_when_a_case_skips_the_llm() -> None:
    rows = base()
    rows[0] = pred("g0", 1, 0.05, golden=True)
    assert target(rows, "Golden") == "fail"


def test_golden_bounds_bracket_the_unknown_llm() -> None:
    rows = base()
    rows[0] = pred("g0", 1, 0.5, golden=True)
    rows[1] = pred("g1", 1, 0.05, golden=True)
    outcome = compute_results(rows, BANDS, costs=COSTS, n_bootstrap=0)
    assert outcome.golden.recall_bounds == pytest.approx((0.8, 0.9))


def test_cascade_recall_uses_only_rows_with_votes() -> None:
    rows = base()
    # A golden case that skips hurts the golden target, not the voted-row recall.
    rows[0] = pred("g0", 1, 0.05, golden=True)
    outcome = compute_results(rows, BANDS, costs=COSTS, n_bootstrap=0)
    assert outcome.cascade.n == 30
    assert outcome.cascade.cascade.expected_recall == 1.0


def test_a_skipped_case_the_llm_would_catch_is_a_regression() -> None:
    rows = base()
    rows[20] = pred("p0", 1, 0.05, votes=(1, 1, 1))
    assert target(rows, "No recall regression") == "fail"


def test_recall_below_the_bar_fails() -> None:
    rows = base()
    for i in range(3):
        rows[20 + i] = pred(f"p{i}", 1, 0.05, votes=(1, 1, 1))
    assert target(rows, "Cascade recall") == "fail"


def test_losing_to_the_keyword_rule_fails() -> None:
    rows = [replace(p, keyword=float(p.label), score=0.5) for p in base()]
    assert target(rows, "Beats the crisis-keyword") == "fail"


# --- the error tables -------------------------------------------------------------------


def test_errors_are_classified() -> None:
    rows = base()
    rows[20] = pred("skipped", 1, 0.05)
    rows[21] = pred("unsure", 1, 0.5, votes=(1, 0, 0))
    rows[22] = pred("sure", 1, 0.5, votes=(1, 1, 1))
    rows[30] = pred("alarm", 0, 0.9, votes=(0, 0, 0))
    rows[0] = pred("g0", 1, 0.5, golden=True)
    errors = {
        e.prediction.id: e for e in compute_results(rows, BANDS, costs=COSTS, n_bootstrap=0).errors
    }

    assert errors["skipped"].kind == "skipped distress" and errors["skipped"].expected_miss == 1.0
    assert errors["unsure"].kind == "escalated distress"
    assert errors["unsure"].expected_miss == pytest.approx(2 / 3)
    assert "sure" not in errors  # the LLM catches it every time
    assert errors["alarm"].kind == "false alarm"
    assert errors["g0"].expected_miss is None  # golden: the LLM's answer is unknown


def test_diagnoses_and_causes_come_from_the_notes() -> None:
    rows = base()
    rows[20] = pred("skipped", 1, 0.05)
    notes = {"skipped": Note("held-out mode", "indirect phrasing")}
    outcome = compute_results(rows, BANDS, costs=COSTS, notes=notes, n_bootstrap=0)
    assert (outcome.errors[0].cause, outcome.errors[0].diagnosis) == (
        "held-out mode",
        "indirect phrasing",
    )


def test_the_analysis_file_carries_a_summary_and_notes(tmp_path: Path) -> None:
    path = tmp_path / "notes.json"
    path.write_text(
        json.dumps({"summary": ["one", "two"], "errors": {"a": {"cause": "c", "note": "n"}}})
    )
    analysis = load_analysis(path)
    assert analysis.summary == ("one", "two")
    assert analysis.errors["a"] == Note("c", "n")


def test_no_analysis_file_means_no_notes(tmp_path: Path) -> None:
    assert load_analysis(tmp_path / "absent.json").errors == {}


# --- honest intervals ------------------------------------------------------------------


def test_the_zero_miss_bound_is_the_exact_one_sided_limit() -> None:
    assert zero_miss_lower_bound(10) == pytest.approx(0.05 ** (1 / 10))
    assert zero_miss_lower_bound(52) == pytest.approx(0.944, abs=1e-3)


def test_a_perfect_recall_is_not_confirmed_when_the_sample_is_small() -> None:
    """Ten catches out of ten sounds certain. It is consistent with a recall of 0.74."""
    outcome = compute_results(base(), BANDS, costs=COSTS, n_bootstrap=50)
    recall = next(t for t in outcome.targets if t.name.startswith("Cascade recall"))
    assert recall.status == "pass"
    assert recall.confirmed == "no"
    assert "lower bound 0.741" in recall.why


def test_regression_check_admits_it_cannot_discriminate_at_the_ceiling() -> None:
    outcome = compute_results(base(), BANDS, costs=COSTS, n_bootstrap=50)
    check = next(t for t in outcome.targets if t.name.startswith("No recall regression"))
    assert check.status == "pass"
    assert check.confirmed == "n/a"
    assert "cannot tell them apart" in check.why


def test_golden_is_a_gate_not_an_estimate() -> None:
    outcome = compute_results(base(), BANDS, costs=COSTS, n_bootstrap=0)
    golden = next(t for t in outcome.targets if t.name.startswith("Golden"))
    assert golden.confirmed == "n/a"
    assert "as low as 0.74" in golden.why


def test_the_report_never_prints_a_collapsed_bootstrap_interval() -> None:
    outcome = compute_results(base(), BANDS, costs=COSTS, n_bootstrap=50)
    markdown = render_markdown(outcome, looks=[object()], validation={}, cv_pr_auc=None)
    assert "[1.00–1.00]" not in markdown


# --- the report ---------------------------------------------------------------------------


def test_results_serialise() -> None:
    outcome = compute_results(base(), BANDS, costs=COSTS, n_bootstrap=20)
    json.dumps(results_json(outcome, looks=[object()]))


def test_the_report_says_how_many_times_the_test_set_was_seen() -> None:
    outcome = compute_results(base(), BANDS, costs=COSTS, n_bootstrap=0)
    once = render_markdown(outcome, looks=[object()], validation={}, cv_pr_auc=None)
    twice = render_markdown(outcome, looks=[object(), object()], validation={}, cv_pr_auc=None)
    assert "first and only look" in once
    assert "evaluated 2 times" in twice


# --- the evaluator routes the way the service routes ----------------------------------


class ClauseEmbedder:
    """Dimension 0 is the share of alarming words — the property mean pooling has."""

    dim = 2

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for row, text in enumerate(texts):
            words = text.split()
            out[row] = (sum(w == "unsafe" for w in words) / max(len(words), 1), 1.0)
        return out


def test_a_diluted_test_note_escalates_rather_than_skipping() -> None:
    """The test look must measure the shipped rule, not the one it replaced.

    The note's crisis clause is diluted until the whole note scores below `low`;
    the clause survives as a segment, so the recorded route is `escalate`.
    """
    diluted = "cleared it again and again and again and again and again. unsafe"
    rows = [*corpus(), make_row("dilute", category=Category.DISTRESS, free_text=diluted)]
    assignment = build_assignment(rows)
    assignment["dilute"] = "test"

    layout = FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, 2, False)
    model = DistressModel(coef=np.array([14.0, 0.0]), intercept=-5.0, layout=layout)
    bands = Thresholds(low=0.05, high=0.95)

    predictions = score_test_split(rows, assignment, model, ClauseEmbedder(), bands, {},
                                   SEGMENTATION, SCOPE)  # fmt: skip
    note = next(p for p in predictions if p.id == "dilute")

    assert note.score < bands.low  # the whole note looks safe
    assert note.worst is not None and note.worst >= bands.low  # one segment does not
    assert note.route == "escalate"
    assert "unsafe" in note.worst_segment


def test_predictions_written_before_segment_scoring_still_render() -> None:
    """Their skip statistic *was* the whole-note score, so that is what is reproduced."""
    old = Prediction(
        id="x", category="distress", label=1, golden=False, feeling="tired",
        text="a note", score=0.4, keyword=0.0, route="escalate", teacher_votes=(1, 1, 0),
    )  # fmt: skip
    assert old.worst is None
    assert old.skip_score == old.score
