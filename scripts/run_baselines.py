"""Baselines 0-4, cross-validated on the training split.

**The test set is not touched here.** It is spent once, at the end, and every
look costs some of its power to tell you anything. So this runs 5-fold
cross-validation *inside train*, grouped by ``origin_id`` so a paraphrase never
sits in the fold that scores its own sibling, and pools the out-of-fold
predictions into one table.

Everything is reported at threshold 0.5. Choosing an operating point is a later
milestone and belongs on validation data with a cost model behind it; picking
one here by looking at these numbers would be tuning on the same data twice.
PR-AUC is the threshold-free column, and the one to compare on.

Usage::

    uv run python scripts/run_baselines.py
    uv run python scripts/run_baselines.py --skip-embedding   # no model download
"""

# scikit-learn's splitter stubs resolve to partially-unknown types at this
# boundary; our own logic stays typed. Same narrow relaxation used in dc.metrics.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dc.baselines import (
    Baseline,
    EmbeddingBaseline,
    KeywordBaseline,
    MajorityBaseline,
    TeacherBaseline,
    TfidfBaseline,
)
from dc.candidates import load_candidates
from dc.metrics import Scores, per_category, score
from dc.schema import Row
from dc.splits import DATA_DIR, load_all_rows, load_assignment, split_rows

REPORTS = Path(__file__).resolve().parents[1] / "reports"
CANDIDATE_DIR = DATA_DIR / "candidates"
N_FOLDS = 5
SEED = 0


def teacher_votes() -> dict[str, tuple[int, ...]]:
    """Votes recorded by propose_labels, keyed by row id. No API calls."""
    return {
        candidate.id: candidate.teacher_votes
        for candidate in load_candidates(*sorted(CANDIDATE_DIR.glob("*.jsonl")))
        if candidate.teacher_votes
    }


def cross_validate(baseline: Baseline, rows: list[Row]) -> np.ndarray:
    """Out-of-fold P(distress) for every row, in input order.

    Grouped by ``origin_id``: a fold must never be scored by a model that
    trained on a paraphrase of the row being scored, or the fold flatters
    itself exactly the way a contaminated test set would.
    """
    labels = np.array([row.label for row in rows])
    groups = np.array([row.origin_id for row in rows])
    out = np.full(len(rows), np.nan, dtype=float)

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_index, val_index in splitter.split(np.zeros(len(rows)), labels, groups):
        baseline.fit([rows[i] for i in train_index])
        out[val_index] = baseline.predict_proba([rows[i] for i in val_index])
    return out


def evaluate(name: str, rows: list[Row], scores: np.ndarray) -> dict[str, Any]:
    """Score one baseline, dropping rows it could not score at all."""
    usable = ~np.isnan(scores)
    kept = [row for row, ok in zip(rows, usable, strict=True) if ok]
    kept_scores = scores[usable]
    result: Scores = score([row.label for row in kept], kept_scores, seed=SEED)
    return {
        "name": name,
        "scored_rows": len(kept),
        "skipped_rows": len(rows) - len(kept),
        "metrics": asdict(result),
        "by_category": [asdict(entry) for entry in per_category(kept, kept_scores)],
    }


def _ci(bounds: list[float] | None) -> str:
    return "" if bounds is None else f" [{bounds[0]:.2f}-{bounds[1]:.2f}]"


def render_markdown(results: list[dict[str, Any]], n_rows: int, n_pos: int) -> str:
    lines = [
        "# Baselines (5-fold CV on train)",
        "",
        f"{n_rows} training rows, {n_pos} positive ({n_pos / n_rows:.0%}). "
        f"{N_FOLDS}-fold StratifiedGroupKFold grouped by `origin_id`, seed {SEED}.",
        "",
        "**The test split is untouched.** It is evaluated once, at the end.",
        "Metrics are at threshold 0.5; the operating point is chosen later, on a",
        "cost model. Compare on PR-AUC, which needs no threshold. Intervals are",
        "95% percentile bootstrap over 1000 resamples.",
        "",
        "| Baseline | Recall | Precision | F1 | PR-AUC | Missed | n |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        m = entry["metrics"]
        lines.append(
            f"| {entry['name']} | {m['recall']:.2f}{_ci(m['recall_ci'])} "
            f"| {m['precision']:.2f} | {m['f1']:.2f} "
            f"| {m['pr_auc']:.2f}{_ci(m['pr_auc_ci'])} "
            f"| {m['fn']} | {entry['scored_rows']} |"
        )

    lines += ["", "## Recall and false positives by category", ""]
    categories = [c["category"] for c in results[0]["by_category"]]
    lines.append("| Baseline | " + " | ".join(categories) + " |")
    lines.append("| --- " * (len(categories) + 1) + "|")
    for entry in results:
        cells: list[str] = []
        for category in entry["by_category"]:
            if category["positives"]:
                cells.append(f"R {category['recall']:.2f}")
            else:
                cells.append(f"FP {category['false_positive_rate']:.2f}")
        lines.append(f"| {entry['name']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`R` is recall on a positive category, `FP` the false-positive rate on a",
        "negative one. `game-frustration` is the column that matters: it is where",
        "a lexical rule collapses, because the words are the same.",
        "",
        "## How to read these numbers",
        "",
        "Three caveats, in descending order of how much they should temper the",
        "table above.",
        "",
        "**Most of this data came from a generator.** 552 of 644 reviewed rows are",
        "LLM variants of 92 hand-written seeds. Grouped CV stops a paraphrase from",
        "scoring its own sibling, but train and validation still share a generator,",
        "and a model can learn that generator's habits rather than the distinction",
        "being taught. Trust the *ordering* of these baselines more than their",
        "magnitudes; expect the absolute numbers to fall on real player text.",
        "",
        "**The teacher's recall is partly definitional.** The labels are the",
        "reviewer's, and the reviewer saw the teacher's votes while deciding. Every",
        "row where the reviewer overruled it counts against it by construction. Read",
        "its row as *disagreement with the reviewed labels*, not as an error rate.",
        "",
        "**Threshold 0.5 is a placeholder, not a choice.** The operating point comes",
        "from a stated cost model on validation data, later. Nothing here is tuned,",
        "which is why PR-AUC is the column to compare on.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="omit baseline 3 (needs `make vendor-model` and the embed extra)",
    )
    args = parser.parse_args()

    rows = load_all_rows()
    train = split_rows(rows, load_assignment(), "train")
    n_pos = sum(row.label for row in train)
    print(f"{len(train)} training rows, {n_pos} positive ({n_pos / len(train):.0%}).")
    print(f"{N_FOLDS}-fold CV grouped by origin_id. The test split is untouched.\n")

    votes = teacher_votes()
    baselines: list[Baseline] = [MajorityBaseline(), KeywordBaseline(), TfidfBaseline(SEED)]

    if not args.skip_embedding:
        try:
            from dc.features import load_embedder

            baselines.append(EmbeddingBaseline(load_embedder(), SEED))
        except (ImportError, FileNotFoundError) as exc:
            print(f"! skipping baseline 3 (minilm+lr): {exc}\n")

    baselines.append(TeacherBaseline(votes))

    results: list[dict[str, Any]] = []
    for baseline in baselines:
        print(f"  {baseline.name} ...", flush=True)
        # The teacher is not fitted, so cross-validating it would only shuffle
        # the same recorded votes. Score it directly on the same rows.
        scores = (
            baseline.predict_proba(train)
            if isinstance(baseline, TeacherBaseline)
            else cross_validate(baseline, train)
        )
        results.append(evaluate(baseline.name, train, scores))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "baselines.json").write_text(
        json.dumps(
            {"n_train": len(train), "n_positive": n_pos, "folds": N_FOLDS, "results": results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(results, len(train), n_pos)
    (REPORTS / "baselines.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)


if __name__ == "__main__":
    main()
