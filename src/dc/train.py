"""Train the distress classifier. Read it top to bottom: this file is the argument.

Each step calls the module that does the work; the comments say *why*, because
the why is what gets asked. After step 2 no test row is in scope — test
evaluation is a separate command, so tuning on the test set is impossible by
construction rather than avoided by discipline.

    uv run --extra embed python -m dc.train
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from dc.artifact import ARTIFACT_DIR, dataset_sha256, save, source_commit
from dc.baselines import KeywordBaseline
from dc.cascade import fit_cascade
from dc.features import EMBED_MODEL, EMBED_MODEL_REVISION, Embedder, build_features, feeling_one_hot
from dc.metrics import pr_auc
from dc.model import FeatureLayout, fit_model
from dc.report import REPORT_DIR, TrainingSummary, artifact_metadata, summarize, write_report
from dc.schema import Row
from dc.selection import (
    Config,
    evaluate_config,
    grid,
    make_folds,
    probe_best_feeling_config,
    select,
)
from dc.splits import DATASET_PATH, SEED_PATH, SPLITS_PATH, Split, check_leakage, split_rows

SEED = 0
N_FOLDS = 5

#: The product rule that constrains `high`, taken from bloom-langgraph's release
#: gate: a note routed straight to support produces no reply for its judge to
#: score, so every encouragement case that reaches support counts as a failure
#: there. Declared here, not discovered from a result.
FORBIDDEN_LABEL = "no non-distress note may be routed to support (bloom-langgraph release gate)"


class GateError(RuntimeError):
    """A precondition for shipping failed. Training stops rather than guessing."""


def run(
    rows: Sequence[Row],
    assignment: dict[str, Split],
    embedder: Embedder,
    *,
    configs: Sequence[Config],
    teacher_votes: Mapping[str, Sequence[int]],
    artifact_dir: Path = ARTIFACT_DIR,
    report_dir: Path = REPORT_DIR,
    dataset_files: Sequence[Path] = (),
    seed: int = SEED,
) -> TrainingSummary:
    # 1. LOAD — rows arrive already validated by dc.schema's strict loader. A
    #    malformed or contradictory row failed before this function was called.

    # 2. SPLIT — two guarantees, checked rather than trusted: the 44 golden rows
    #    are test-only (they are the release gate), and no origin family
    #    straddles the line (paraphrase leakage). Then only train stays in scope.
    problems = check_leakage(list(rows), assignment)
    if problems:
        raise GateError("the split is unsound:\n  " + "\n  ".join(problems))
    train = split_rows(list(rows), assignment, "train")
    y = np.array([row.label for row in train])
    groups = np.array([row.origin_id for row in train])

    # 3. EMBED — frozen MiniLM at production's pinned revision. ~500 rows cannot
    #    train its 22.7M parameters without memorising them; they can fit 385.
    x_text = build_features(train, embedder, include_feeling=False)
    x_feeling = feeling_one_hot([row.feeling for row in train])

    # 4. BASELINE — the bar. A crisis-keyword rule needs no training at all, so
    #    a model that cannot beat it has no reason to exist.
    keyword = pr_auc(y, KeywordBaseline().predict_proba(train))

    # 5. SELECT — every config on the same grouped folds, chosen by a rule fixed
    #    in dc.selection before any result existed. Feeling configs are still
    #    scored: measuring the shortcut is the point of the ablation.
    folds = make_folds(y, groups, n_splits=N_FOLDS, seed=seed)
    results = [evaluate_config(c, x_text, x_feeling, y, folds, seed=seed) for c in configs]
    selection = select(results)
    if selection.chosen.mean <= keyword:
        raise GateError(
            f"chosen model PR-AUC {selection.chosen.mean:.3f} does not beat the keyword "
            f"rule's {keyword:.3f}. Ship the word list, or fix the data."
        )
    probe_config, probe = probe_best_feeling_config(
        results, x_text, [row.feeling for row in train], y, folds, seed=seed
    )

    # 6. THRESHOLD — two cutoffs on the chosen config's out-of-fold scores. Below
    #    `low` the LLM is skipped, and only below where any distress case scored.
    #    Above `high` a note goes straight to support; between, the LLM decides as
    #    it does today. `high` minimises expected cost with a missed crisis at 20x
    #    an unneeded kind message — but under a constraint that outranks the ratio.
    #    bloom-langgraph's release gate scores any encouragement case routed to
    #    support as a failure, so no non-distress note may reach support; that
    #    raises a floor under `high`. "The cascade loses no recall to the LLM"
    #    holds here by construction, so that check belongs on the test set.
    cascade = fit_cascade(
        y, selection.chosen.oof, [teacher_votes.get(row.id, ()) for row in train],
        ids=[row.id for row in train], texts=[row.free_text for row in train],
        forbidden=(y == 0), forbidden_label=FORBIDDEN_LABEL,
    )  # fmt: skip

    # 7. FIT — refit on all of train at the chosen config. The CV models existed
    #    to choose; this one exists to ship.
    chosen = selection.chosen.config
    layout = FeatureLayout(
        EMBED_MODEL, EMBED_MODEL_REVISION, int(x_text.shape[1]), chosen.include_feeling
    )
    x = np.hstack([x_text, x_feeling]) if chosen.include_feeling else x_text
    model = fit_model(chosen, x, y, layout, seed=seed)

    # 8. SAVE — npz + json, no pickle: the consumer reads data, never runs code.
    summary = summarize(
        train, y, folds=N_FOLDS, seed=seed, keyword_pr_auc=keyword, results=results,
        selection=selection, probe_config=probe_config, probe=probe, cascade=cascade,
    )  # fmt: skip
    provenance = {**source_commit(), "dataset_sha256": dataset_sha256(dataset_files)}
    save(model, artifact_metadata(summary, provenance), artifact_dir)
    write_report(summary, report_dir)
    return summary


def main() -> None:
    from dc.candidates import load_teacher_votes
    from dc.features import load_embedder
    from dc.report import render_cascade_markdown, render_markdown
    from dc.splits import load_all_rows, load_assignment

    summary = run(
        load_all_rows(),
        load_assignment(),
        load_embedder(),
        configs=grid(),
        teacher_votes=load_teacher_votes(),
        dataset_files=(SEED_PATH, DATASET_PATH, SPLITS_PATH),
    )
    print(render_markdown(summary))
    print(render_cascade_markdown(summary))


if __name__ == "__main__":
    main()
