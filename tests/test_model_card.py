"""The model card cannot quietly drift from the model it describes.

Every number below is read from the committed artifact and reports, formatted the way
the card writes it, and must appear in MODEL_CARD.md. Retrain, re-render, or look at
the test set again, and this fails until the card is updated to match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "MODEL_CARD.md"


def load(relative: str) -> Any:
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def expected_facts() -> dict[str, str]:
    model = load("artifacts/model.json")
    test = load("reports/test_results.json")
    ledger = load("reports/test_ledger.json")
    explanations = load("reports/explanations.json")
    baselines = {r["name"]: r for r in load("reports/baselines.json")["results"]}
    thresholds = model["thresholds"]
    cascade = test["cascade"]
    looks, renders = len(ledger["looks"]), len(ledger["renders"])
    categories = {c["category"]: c for c in test["per_category"]}
    rows = sum(1 for _ in (REPO / "data" / "labelled.jsonl").open()) + 44
    generated = sum(
        1 for line in (REPO / "data" / "labelled.jsonl").open()
        if json.loads(line)["provenance"] == "llm-augmented"
    )  # fmt: skip

    return {
        "artifact hash": ledger["looks"][0]["artifact_sha256"][:12],
        "low threshold": f"{thresholds['low']:.4f}",
        "high threshold": f"{thresholds['high']:.3f}",
        "cross-validated PR-AUC": f"{model['selection']['cv_pr_auc_mean']:.3f}",
        "test PR-AUC": f"{test['model']['pr_auc']:.3f}",
        "test keyword PR-AUC": f"{test['keyword']['pr_auc']:.3f}",
        "CV keyword PR-AUC": f"{baselines['keyword']['metrics']['pr_auc']:.3f}",
        "CV teacher PR-AUC": f"{baselines['teacher(haiku)']['metrics']['pr_auc']:.3f}",
        "test teacher PR-AUC": f"{cascade['teacher']['pr_auc']:.3f}",
        "test model PR-AUC on voted rows": f"{cascade['model']['pr_auc']:.3f}",
        "test Brier": f"{test['calibration']['brier']:.3f}",
        "test ECE": f"{test['calibration']['ece']:.3f}",
        "cascade recall lower bound": f"{0.05 ** (1 / cascade['positives']):.3f}",
        "cascade false alarms": f"{cascade['cascade']['expected_false_positives']:.1f}",
        "LLM-alone false alarms": f"{cascade['llm_alone']['expected_false_positives']:.1f}",
        "skip-band bound": f"{thresholds['validation']['skip_miss_upper_bound_95']:.1%}",
        "surrogate R²": f"R² {explanations['fidelity']['r2']:.2f}",
        "route agreement": f"{explanations['fidelity']['route_agreement']:.0%}",
        "nonsense false-positive rate": f"{categories['nonsense']['false_positive_rate']:.0%}",
        "synthetic share": f"{generated} of {rows}",
        "looks": "Scored once" if looks == 1 else f"Scored {looks} times",
        "renders": f"re-rendered {renders} times",
    }


@pytest.mark.parametrize("fact", sorted(expected_facts()))
def test_the_card_states(fact: str) -> None:
    value = expected_facts()[fact]
    assert value in CARD.read_text(encoding="utf-8"), (
        f"MODEL_CARD.md does not state the {fact} ({value!r}). Update the card."
    )


def test_the_card_declares_what_it_is_not() -> None:
    card = CARD.read_text(encoding="utf-8")
    assert "Not a diagnostic instrument" in card
    assert "## Out of scope" in card
