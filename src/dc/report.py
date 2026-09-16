"""The training report: what was chosen, why, and what the chip would have cost.

Rendered to ``reports/training.md`` for people and ``reports/training.json`` for
tools. The JSON omits out-of-fold arrays — the report records decisions and the
numbers behind them, not intermediate state.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dc.calibration import Bin, expected_calibration_error, reliability_bins
from dc.cascade import CascadeFit, Outcome
from dc.schema import Row
from dc.selection import Config, ConfigResult, Selection, SwapOutcome

__all__ = [
    "REPORT_DIR",
    "TrainingSummary",
    "artifact_metadata",
    "render_cascade_markdown",
    "render_markdown",
    "summarize",
    "thresholds_metadata",
    "write_report",
]

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass(frozen=True)
class TrainingSummary:
    n_train: int
    n_positive: int
    folds: int
    seed: int
    keyword_pr_auc: float
    results: tuple[ConfigResult, ...]
    selection: Selection
    probe_config: Config | None
    probe: tuple[SwapOutcome, ...]
    feeling_counts: dict[str, tuple[int, int]]
    brier: float
    ece: float
    bins: tuple[Bin, ...]
    cascade: CascadeFit


def summarize(
    train: Sequence[Row],
    y: np.ndarray,
    *,
    folds: int,
    seed: int,
    keyword_pr_auc: float,
    results: Sequence[ConfigResult],
    selection: Selection,
    probe_config: Config | None,
    probe: Sequence[SwapOutcome],
    cascade: CascadeFit,
) -> TrainingSummary:
    counts: dict[str, tuple[int, int]] = {}
    for row in train:
        n, positives = counts.get(row.feeling.value, (0, 0))
        counts[row.feeling.value] = (n + 1, positives + row.label)
    oof = selection.chosen.oof
    return TrainingSummary(
        n_train=len(train),
        n_positive=int(np.sum(y)),
        folds=folds,
        seed=seed,
        keyword_pr_auc=keyword_pr_auc,
        results=tuple(results),
        selection=selection,
        probe_config=probe_config,
        probe=tuple(probe),
        feeling_counts=dict(sorted(counts.items())),
        brier=selection.chosen.brier,
        ece=expected_calibration_error(y, oof),
        bins=tuple(reliability_bins(y, oof)),
        cascade=cascade,
    )


def _status(result: ConfigResult, selection: Selection) -> str:
    if result is selection.chosen:
        return "**chosen**"
    if result in selection.excluded:
        return "ineligible (feeling)"
    if result in selection.survivors:
        return "within 1 SE"
    return ""


def render_markdown(summary: TrainingSummary) -> str:
    s = summary
    chosen = s.selection.chosen
    lines = [
        "# Training",
        "",
        f"{s.n_train} training rows, {s.n_positive} positive "
        f"({s.n_positive / s.n_train:.0%}). {s.folds}-fold StratifiedGroupKFold "
        f"grouped by `origin_id`, seed {s.seed}. Every config is scored on the same folds.",
        "",
        "**The test split is untouched.** It is evaluated once, at the end.",
        "",
        "## The bar",
        "",
        f"A crisis-keyword rule needs no training and scores PR-AUC "
        f"**{s.keyword_pr_auc:.3f}** on these rows. The chosen model scores "
        f"**{chosen.mean:.3f}** mean fold PR-AUC, so a model was worth building.",
        "",
        "## Why the chip a player picked is not a feature",
        "",
        "In this dataset the feeling is close to a label leak — an artefact of how",
        "rows were authored, not a fact about players:",
        "",
        "| Feeling | Rows | Distress | P(distress) |",
        "| --- | --- | --- | --- |",
    ]
    for feeling, (n, positives) in s.feeling_counts.items():
        lines.append(f"| `{feeling}` | {n} | {positives} | {positives / n:.0%} |")

    with_feeling = [r for r in s.results if r.config.include_feeling]
    without_feeling = [r for r in s.results if not r.config.include_feeling]
    if with_feeling and without_feeling:
        best_with = max(with_feeling, key=lambda r: r.mean)
        best_without = max(without_feeling, key=lambda r: r.mean)
        lines += [
            "",
            f"Given the chip, the best config reaches **{best_with.mean:.3f}** PR-AUC "
            f"against **{best_without.mean:.3f}** without it "
            f"({best_with.mean - best_without.mean:+.3f}). Cross-validation rewards the "
            "shortcut, because the shortcut is present in every fold — which is exactly",
            "why cross-validation cannot be the thing that rejects it.",
        ]

    if s.probe and s.probe_config is not None:
        lines += [
            "",
            f"**The probe.** Take the held-out distress cases, leave the text untouched, "
            f"and change only the chip. Scored by `{s.probe_config.label}` at threshold 0.5:",
            "",
            "| Chip swapped to | Recall before | Recall after | Mean score before → after |",
            "| --- | --- | --- | --- |",
        ]
        for outcome in s.probe:
            lines.append(
                f"| `{outcome.swap_to}` | {outcome.recall_before:.2f} "
                f"| {outcome.recall_after:.2f} "
                f"| {outcome.mean_score_before:.2f} → {outcome.mean_score_after:.2f} |"
            )
        worst = min(s.probe, key=lambda o: o.recall_after)
        lines += [
            "",
            f"A player who picks `{worst.swap_to}` and then writes a real crisis note is "
            f"caught {worst.recall_after:.0%} of the time instead of "
            f"{worst.recall_before:.0%}. That is the shortcut, measured. The selection "
            "rule makes every feeling config ineligible.",
        ]

    lines += [
        "",
        "## Selection",
        "",
        f"Rule, fixed before any result existed: {s.selection.rule}",
        "",
        "| Config | PR-AUC (mean ± SE) | Brier | |",
        "| --- | --- | --- | --- |",
    ]
    for result in sorted(s.results, key=lambda r: (r.config.include_feeling, -r.mean)):
        lines.append(
            f"| `{result.config.label}` | {result.mean:.3f} ± {result.se:.3f} "
            f"| {result.brier:.3f} | {_status(result, s.selection)} |"
        )
    lines += [
        "",
        f"Best eligible mean was `{s.selection.best.config.label}` at "
        f"{s.selection.best.mean:.3f}; {len(s.selection.survivors)} config(s) fell within "
        f"one standard error of it, and `{chosen.config.label}` had the lowest Brier score "
        "among them.",
        "",
        "## Calibration of the chosen model (out-of-fold)",
        "",
        f"Brier **{s.brier:.3f}** · expected calibration error **{s.ece:.3f}**",
        "",
        "| Score bin | Rows | Mean predicted | Observed distress |",
        "| --- | --- | --- | --- |",
    ]
    for b in s.bins:
        if b.n:
            lines.append(
                f"| {b.lower:.1f}–{b.upper:.1f} | {b.n} | {b.mean_predicted:.2f} "
                f"| {b.observed_rate:.2f} |"
            )
    populated = [b for b in s.bins if b.n and b.gap is not None]
    if populated:
        edge_rows = s.bins[0].n + s.bins[-1].n
        worst = max(populated, key=lambda b: abs(b.gap or 0.0))
        lines += [
            "",
            f"{edge_rows / s.n_train:.0%} of rows sit in the two outermost bins, where "
            "the model is close to calibrated. The middle bins — exactly where an "
            '"uncertain, ask the LLM" band would sit — hold few rows and carry the '
            f"largest gaps (worst: {worst.lower:.1f}–{worst.upper:.1f}, predicted "
            f"{worst.mean_predicted:.2f}, observed {worst.observed_rate:.2f}, "
            f"{worst.n} rows).",
            "",
            "Two consequences. Thresholds are fitted empirically on these out-of-fold "
            "scores, so they do not depend on the scores being probabilities. But these "
            "scores should not be *described* as probabilities anywhere a person reads "
            "them without recalibrating first.",
        ]

    lines += [
        "",
        "## Artifact",
        "",
        "`artifacts/model.npz` + `artifacts/model.json`, with operating thresholds "
        f"`low` {s.cascade.thresholds.low:.4f} and `high` {s.cascade.thresholds.high:.3f} "
        "fitted from the cost model. How they were chosen, and what the cascade would "
        "have done on validation, is in `reports/cascade.md`.",
        "",
    ]
    return "\n".join(lines)


def _outcome_row(outcome: Outcome, *, bold: bool = False) -> str:
    name = f"**{outcome.name}**" if bold else outcome.name
    return (
        f"| {name} | {outcome.expected_recall:.3f} | {outcome.expected_missed:.1f} "
        f"| {outcome.expected_false_positives:.1f} | {outcome.escalate_share:.0%} "
        f"| {outcome.expected_cost:.1f} |"
    )


def render_cascade_markdown(summary: TrainingSummary) -> str:
    fit = summary.cascade
    low, high = fit.thresholds.low, fit.thresholds.high
    ratio = fit.costs.ratio
    lines = [
        "# Cascade",
        "",
        f"Thresholds fitted on the chosen model's out-of-fold scores: {summary.n_train} "
        f"training rows, {summary.n_positive} distress. **The test split is untouched.**",
        "",
        "## Three routes",
        "",
        "| Score | Route | What happens |",
        "| --- | --- | --- |",
        f"| p < {low:.4f} | `skip-llm` | Encouragement branch. No LLM call. |",
        f"| {low:.4f} ≤ p ≤ {high:.3f} | `escalate` | `claude-haiku-4-5` decides — "
        "exactly what production does today. |",
        f"| p > {high:.3f} | `support` | The reviewed support message. No LLM call. |",
        "",
        "A score exactly on a threshold escalates, and so does any score that is not a "
        "finite probability. Every way the local model can fail lands on today's "
        "behaviour, never on a skipped call.",
        "",
        "## The cost model",
        "",
        f"A missed crisis is treated as **{ratio:g}×** as costly as an unneeded support "
        "message. That is a product decision, not a statistical one, and it is written "
        "as a number so it can be argued with — the sensitivity table below shows what "
        "changes if it is set differently.",
        "",
        "## `low` is a safety constraint, not an optimisation",
        "",
        "Skipping the LLM is the only route that can add a missed crisis, so `low` is "
        f"not tuned for cost. The lowest score any validation distress case received "
        f"was **{fit.low.lowest_positive_score:.4f}**; `low` keeps "
        f"{fit.low.margin:.0%} of it, **{low:.4f}**. No validation distress case falls "
        "in the skip band.",
        "",
        f"Zero misses among {fit.low.n_positive} is still a finite sample. The honest "
        "form of the claim: the true share of distress cases that would skip the LLM is "
        f"below **{fit.low.miss_upper_bound:.1%}** with 95% confidence.",
        "",
        "The distress cases nearest the skip band — the rows that set `low`, and the "
        "first ones to re-read:",
        "",
        "| Row | Score | Note |",
        "| --- | --- | --- |",
    ]
    for row_id, score_value, text in fit.low.nearest_positives:
        lines.append(f"| `{row_id}` | {score_value:.4f} | {text} |")

    if fit.constraint is not None:
        c = fit.constraint
        lines += [
            "",
            "## A product rule outranks the cost model",
            "",
            f"**{c.label}**",
            "",
            f"That rule is a constraint, not a price, so it is applied before the ratio is: no "
            f"cutoff is considered that would route one of the {c.rows} protected validation "
            f"rows to support. The highest-scoring one sets the floor at **{c.floor:.4f}**:",
            "",
            f"> `{c.set_by}` — {c.set_by_text}",
            "",
            f"Every operating point below that floor is removed, which leaves "
            f"{len(fit.envelope)} of them. A single row can therefore decide the whole "
            "operating point; if that row's label is contested, so is the threshold.",
        ]

    lines += [
        "",
        "## `high` is chosen by expected cost",
        "",
        f"Every permitted split of the rows above `low` was scored by expected cost at "
        f"{ratio:g}:1, and **{high:.3f}** was cheapest. The cutoff sits midway between the "
        "two scores it separates, so it does not rest on a validation row. Ties went to "
        "the higher cutoff, which escalates more and so stays closer to production.",
        "",
        "## What the cascade would have done",
        "",
        "Expected values over the validation rows. The LLM is modelled as one call that "
        "says distress with probability equal to the share of its three recorded votes.",
        "",
        "| Policy | Expected recall | Expected missed | Expected false alarms "
        "| LLM calls | Expected cost |",
        "| --- | --- | --- | --- | --- | --- |",
        _outcome_row(fit.llm_alone),
        _outcome_row(fit.model_alone),
        _outcome_row(fit.cascade, bold=True),
        "",
        f"Routes under the cascade: skip **{fit.cascade.skip_share:.0%}** · escalate "
        f"**{fit.cascade.escalate_share:.0%}** · support **{fit.cascade.support_share:.0%}**.",
    ]

    if fit.cascade.expected_missed < 0.5:
        lines += [
            "",
            f"**Recall {fit.cascade.expected_recall:.3f} describes how the thresholds were "
            "built, not how they will perform.** `low` was placed under the lowest-scoring "
            "distress case in these rows, and `high` was optimised on the same rows. Expect "
            "misses on the test set.",
        ]

    lines += [
        "",
        "## If the ratio is different",
        "",
        "`low` does not depend on the ratio. `high` does — and not smoothly. Across every "
        "possible ratio, the cost model can only ever choose one of these operating points "
        "(a constraint, where one applies, has already removed the rest):",
        "",
        "| Ratios that choose it | `high` | Expected recall | Expected missed "
        "| Expected false alarms | LLM calls | Support |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n, segment in enumerate(fit.envelope):
        if segment.ratio_to is None:
            span = f"{segment.ratio_from:.1f}:1 and above"
        elif n == 0:
            span = f"below {segment.ratio_to:.1f}:1"
        else:
            span = f"{segment.ratio_from:.1f}:1 – {segment.ratio_to:.1f}:1"
        marker = f" ← {ratio:g}:1" if n == fit.chosen_segment else ""
        lines.append(
            f"| {span}{marker} | {segment.high:.3f} | {segment.expected_recall:.3f} "
            f"| {segment.expected_missed:.1f} | {segment.expected_false_positives:.1f} "
            f"| {segment.escalate_share:.0%} | {segment.support_share:.0%} |"
        )

    chosen_segment = fit.envelope[fit.chosen_segment]
    if fit.alternative is not None:
        alt = fit.alternative
        fewer_missed = alt.expected_missed - chosen_segment.expected_missed
        more_alarms = chosen_segment.expected_false_positives - alt.expected_false_positives
        lines += [
            "",
            f"**{ratio:g}:1 sits above a switch at {chosen_segment.ratio_from:.1f}:1.** One "
            f"step more conservative, `high` would be {alt.high:.3f}: "
            f"{fewer_missed:.1f} more expected missed crises, and {more_alarms:.0f} fewer "
            "false alarms. The switch sits exactly where one expected catch is worth "
            f"{chosen_segment.ratio_from:.1f} false alarms — so choosing {ratio:g}:1 is "
            f"choosing to send {more_alarms:.0f} more players the support message to catch "
            f"those {fewer_missed:.1f}.",
        ]
        if fit.decisive:
            lines += [
                "",
                f"That trade rests on {len(fit.decisive)} distress case(s) where the "
                "teacher's votes disagreed with the reviewer. They are the rows to re-read "
                f"before accepting {ratio:g}:1 — and they are exactly the disagreements the "
                "caveat below calls partly definitional.",
                "",
                "| Row | Score | Teacher votes for distress | Note |",
                "| --- | --- | --- | --- |",
            ]
            for row_id, score_value, share, text in fit.decisive:
                lines.append(f"| `{row_id}` | {score_value:.3f} | {share:.0%} | {text} |")

    lines += [
        "",
        "## If the margin is different",
        "",
        "| Margin | `low` | Notes that skip the LLM |",
        "| --- | --- | --- |",
    ]
    for row in fit.by_margin:
        marker = " ←" if row.margin == fit.low.margin else ""
        lines.append(f"| {row.margin:.2f}{marker} | {row.low:.4f} | {row.skip_share:.0%} |")

    lines += [
        "",
        "## How to read these numbers",
        "",
        "**The same validation data was used twice.** These out-of-fold scores chose "
        "the config and then fitted both thresholds, so every number above is "
        "optimistic. The test set, spent once in the next milestone, is the check.",
        "",
        "**The shipped model did not produce these scores.** Out-of-fold scores come "
        "from models trained on 80% of train; the shipped model saw all of it and tends "
        "to score more confidently. The thresholds are expected to transfer. Until the "
        "test set confirms it, that is an expectation.",
        "",
        "**The LLM's recall here is partly definitional.** The reviewer saw the "
        "teacher's votes while labelling, so every overruled vote counts against it by "
        "construction. That understates the LLM alone, and flatters the cascade's lead "
        "over it.",
        "",
        "**Latency is not in this report.** `make bench` measures it separately, because "
        "timings vary run to run and this file should not.",
        "",
    ]
    return "\n".join(lines)


def _result_json(result: ConfigResult, selection: Selection) -> dict[str, Any]:
    return {
        "config": asdict(result.config),
        "fold_pr_auc": list(result.fold_pr_auc),
        "mean_pr_auc": result.mean,
        "se_pr_auc": result.se,
        "brier": result.brier,
        "status": _status(result, selection).strip("*") or None,
    }


def write_report(summary: TrainingSummary, directory: Path = REPORT_DIR) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / "training.md"
    json_path = directory / "training.json"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    (directory / "cascade.md").write_text(render_cascade_markdown(summary), encoding="utf-8")
    payload: dict[str, Any] = {
        "n_train": summary.n_train,
        "n_positive": summary.n_positive,
        "folds": summary.folds,
        "seed": summary.seed,
        "keyword_pr_auc": summary.keyword_pr_auc,
        "rule": summary.selection.rule,
        "chosen": asdict(summary.selection.chosen.config),
        "results": [_result_json(r, summary.selection) for r in summary.results],
        "feeling_counts": {
            k: {"rows": n, "distress": p} for k, (n, p) in summary.feeling_counts.items()
        },
        "probe_config": None if summary.probe_config is None else asdict(summary.probe_config),
        "probe": [asdict(o) for o in summary.probe],
        "calibration": {
            "brier": summary.brier,
            "ece": summary.ece,
            "bins": [asdict(b) for b in summary.bins],
        },
        "cascade": thresholds_metadata(summary.cascade),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def artifact_metadata(summary: TrainingSummary, provenance: dict[str, Any]) -> dict[str, Any]:
    chosen = summary.selection.chosen
    return {
        "hyperparameters": {
            "C": chosen.config.C,
            "class_weight": chosen.config.class_weight,
            "penalty": "l2",
            "solver": "lbfgs",
        },
        "thresholds": thresholds_metadata(summary.cascade),
        "selection": {
            "rule": summary.selection.rule,
            "cv_pr_auc_mean": chosen.mean,
            "cv_pr_auc_se": chosen.se,
            "folds": summary.folds,
            "seed": summary.seed,
        },
        "calibration": {"brier": summary.brier, "ece": summary.ece},
        "provenance": {**provenance, "n_train": summary.n_train, "n_positive": summary.n_positive},
    }


def thresholds_metadata(fit: CascadeFit) -> dict[str, Any]:
    """The decision rule, carried by the artifact so the consumer cannot guess it."""
    return {
        "low": fit.thresholds.low,
        "high": fit.thresholds.high,
        "routes": {
            "below_low": "skip-llm",
            "between_inclusive": "escalate",
            "above_high": "support",
            "invalid_score": "escalate",
        },
        "cost_model": {
            "false_negative": fit.costs.false_negative,
            "false_positive": fit.costs.false_positive,
        },
        "low_margin": fit.low.margin,
        "fitted_on": "out-of-fold scores of the chosen config, grouped CV on train",
        "validation": {
            "llm_alone_expected_recall": fit.llm_alone.expected_recall,
            "cascade_expected_recall": fit.cascade.expected_recall,
            "llm_alone_expected_cost": fit.llm_alone.expected_cost,
            "cascade_expected_cost": fit.cascade.expected_cost,
            "skip_share": fit.cascade.skip_share,
            "escalate_share": fit.cascade.escalate_share,
            "support_share": fit.cascade.support_share,
            "skip_miss_upper_bound_95": fit.low.miss_upper_bound,
        },
    }
