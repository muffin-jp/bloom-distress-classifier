"""Attack the skip band: can a genuinely distressed note avoid the LLM?

Every note in ``data/redteam.jsonl`` is distress by construction, so exactly one
outcome is a failure: a note scoring below ``low`` and skipping the LLM. Nothing
else in this system can add a missed crisis.

Two scores per note:

* **whole note**, which is what the service does today; and
* **worst segment**, the largest score over the note's sentences and sliding word
  windows — the candidate rule for the skip band, which would skip only when
  *every* segment is below ``low``.

Reporting both says whether segment-level scoring would close the hole.

This is a probe, not an estimate. The notes were written after reading the
model's explanations, so they are adversarial by design and say nothing about
how often real players write this way. They are deliberately kept out of
training: a probe that a model has been trained on has stopped being a probe.
Re-run it against every artifact, before the test set is touched.

    uv run --extra embed python scripts/redteam.py

Exits non-zero if any note skips the LLM.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dc.artifact import load, read_scope, read_segmentation
from dc.cascade import Route, Thresholds, route, route_note
from dc.features import load_embedder
from dc.serve import score_notes

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTES_PATH = REPO_ROOT / "data" / "redteam.jsonl"
REPORT_DIR = REPO_ROOT / "reports"


@dataclass(frozen=True)
class Probe:
    id: str
    family: str
    text: str
    whole: float
    worst_segment: float
    worst_text: str
    out_of_scope: str

    def skips(self, thresholds: Thresholds) -> bool:
        """Under the whole-note rule — what the first artifact served."""
        return route(self.whole, thresholds) is Route.SKIP_LLM

    def skips_with_segments(self, thresholds: Thresholds) -> bool:
        """Under the shipped rule: segments, and the scope rule that outranks them."""
        if self.out_of_scope:
            return False
        return route_note(self.whole, self.worst_segment, thresholds) is Route.SKIP_LLM


def load_notes(path: Path = NOTES_PATH) -> list[dict[str, str]]:
    notes = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [n["id"] for n in notes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate note id in the red-team set")
    return notes


def probe(notes: list[dict[str, str]]) -> list[Probe]:
    """Score every note through the same path the service uses.

    The segmentation comes from the artifact rather than from this file's imports:
    the probe has to attack the rule that would actually ship, not the rule this
    script happens to have been written against.
    """
    artifact = load()
    embedder = load_embedder()
    scored = score_notes(
        [note["text"] for note in notes],
        artifact.model,
        embedder,
        segmentation=read_segmentation(artifact.metadata),
        scope=read_scope(artifact.metadata),
    )
    return [
        Probe(
            id=note["id"],
            family=note["family"],
            text=note["text"],
            whole=score.whole,
            worst_segment=score.worst,
            worst_text=score.worst_segment,
            out_of_scope=score.out_of_scope,
        )
        for note, score in zip(notes, scored, strict=True)
    ]


def render(probes: list[Probe], thresholds: Thresholds) -> str:
    skipped = [p for p in probes if p.skips(thresholds)]
    still = [p for p in skipped if p.skips_with_segments(thresholds)]
    families = sorted({p.family for p in probes})
    by_family = Counter(p.family for p in skipped)
    counts = Counter(p.family for p in probes)

    lines = [
        "# Red-team: the skip band",
        "",
        f"{len(probes)} notes, every one distress by construction. A note that scores below "
        f"`low` ({thresholds.low:.4f}) skips the LLM and is answered with generated "
        "encouragement — the only failure in this system that can add a missed crisis.",
        "",
        f"**{len(still)} of {len(probes)} skip the LLM** under the shipped rule: each note's "
        "sentences and sliding word windows are scored, a note skips only when every segment "
        "is below `low`, and a note in a non-Latin script escalates whatever it scores. Under "
        f"the whole-note rule this replaced, **{len(skipped)}** skip.",
        "",
        "> **Read this as a guarantee, not a finding.** `low` is fitted so that no note in",
        "> this set can skip — see `reports/cascade.md`. These notes constrained the",
        "> artifact, so they cannot also test it. Attack notes written *after* the fit are",
        "> what would make this a probe again.",
        "",
        f"{sum(1 for p in probes if p.out_of_scope)} of the notes are escalated by the scope "
        "rule rather than by their score.",
        "",
        "| Family | Notes | Skip (whole-note rule) | Skip (shipped) |",
        "| --- | --- | --- | --- |",
    ]
    for family in families:
        segment_skips = sum(
            1 for p in skipped if p.family == family and p.skips_with_segments(thresholds)
        )
        lines.append(f"| {family} | {counts[family]} | {by_family[family]} | {segment_skips} |")

    lines += [
        "",
        "## Notes that skip the LLM",
        "",
        "`worst segment` is the highest-scoring sentence or window inside the note.",
        "",
        "| Note | Whole | Worst segment | The segment |",
        "| --- | --- | --- | --- |",
    ]
    for p in sorted(skipped, key=lambda p: p.whole):
        lines.append(f"| {p.text} | **{p.whole:.4f}** | {p.worst_segment:.4f} | {p.worst_text} |")
    if not skipped:
        lines.append("| _none_ | | | |")

    lines += [
        "",
        "## How to read this",
        "",
        "These notes were written after reading the model's explanations, so they are "
        "adversarial by design. They measure whether a hole exists, not how often players "
        "fall into it. They are kept out of training on purpose: a probe a model has trained "
        "on is no longer a probe.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=NOTES_PATH)
    args = parser.parse_args()

    artifact = load()
    bands = artifact.metadata["thresholds"]
    thresholds = Thresholds(float(bands["low"]), float(bands["high"]))

    probes = probe(load_notes(args.notes))
    skipped = [p for p in probes if p.skips(thresholds)]
    still = [p for p in skipped if p.skips_with_segments(thresholds)]

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "redteam.md").write_text(render(probes, thresholds), encoding="utf-8")
    payload: dict[str, Any] = {
        "notes": len(probes),
        "low": thresholds.low,
        "skipped": [p.id for p in skipped],
        "skipped_with_segment_scoring": [p.id for p in still],
        "minimum_whole_score": min(p.whole for p in probes),
        "minimum_worst_segment": min(p.worst_segment for p in probes),
        "out_of_scope": [p.id for p in probes if p.out_of_scope],
        "probes": [
            {
                "id": p.id,
                "family": p.family,
                "text": p.text,
                "whole": p.whole,
                "worst_segment": p.worst_segment,
                "worst_text": p.worst_text,
                "out_of_scope": p.out_of_scope,
            }  # fmt: skip
            for p in probes
        ],
    }
    (REPORT_DIR / "redteam.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"{len(probes)} distress notes probed against low={thresholds.low:.4f}.")
    print(f"  skip under the shipped rule:   {len(still)}")
    print(f"  (whole-note rule, for scale:   {len(skipped)})")
    print(f"  lowest whole-note score:       {min(p.whole for p in probes):.4f}")
    print(f"  lowest worst-segment score:    {min(p.worst_segment for p in probes):.4f}")
    print(f"  escalated by the scope rule:   {sum(1 for p in probes if p.out_of_scope)}")
    # The gate is the rule that ships. `skipped` is kept alongside it because the
    # gap between the two is the only evidence that segmentation is doing anything.
    if still:
        print("\nNotes that still skip the LLM:")
        for p in sorted(still, key=lambda p: p.worst_segment)[:5]:
            print(f"  {p.worst_segment:.4f}  (whole {p.whole:.4f})  {p.text[:64]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
