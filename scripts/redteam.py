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

import numpy as np

from dc.artifact import load
from dc.cascade import Route, Thresholds, route
from dc.features import load_embedder
from dc.text import split_segments

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

    def route(self, thresholds: Thresholds) -> Route:
        return route(self.whole, thresholds)

    def skips(self, thresholds: Thresholds) -> bool:
        return self.route(thresholds) is Route.SKIP_LLM

    def skips_with_segments(self, thresholds: Thresholds) -> bool:
        return self.worst_segment < thresholds.low


def load_notes(path: Path = NOTES_PATH) -> list[dict[str, str]]:
    notes = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [n["id"] for n in notes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate note id in the red-team set")
    return notes


def probe(notes: list[dict[str, str]]) -> list[Probe]:
    artifact = load()
    embedder = load_embedder()
    segments_per_note = [split_segments(note["text"]) for note in notes]
    flat = [segment for segments in segments_per_note for segment in segments]
    scores = artifact.model.predict_proba_features(np.asarray(embedder.embed(flat), dtype=float))

    out: list[Probe] = []
    cursor = 0
    for note, segments in zip(notes, segments_per_note, strict=True):
        window = scores[cursor : cursor + len(segments)]
        cursor += len(segments)
        best = int(np.argmax(window))
        out.append(
            Probe(
                id=note["id"],
                family=note["family"],
                text=note["text"],
                whole=float(window[0]),  # split_segments always yields the whole note first
                worst_segment=float(window[best]),
                worst_text=segments[best],
            )
        )
    return out


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
        f"**{len(skipped)} of {len(probes)} skip the LLM today.** Scoring each note's segments "
        f"instead — sentences and sliding word windows, skipping only when every segment is "
        f"below `low` — leaves **{len(still)}**.",
        "",
        "| Family | Notes | Skip today | Skip with segment scoring |",
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
        "probes": [
            {
                "id": p.id,
                "family": p.family,
                "text": p.text,
                "whole": p.whole,
                "worst_segment": p.worst_segment,
                "worst_text": p.worst_text,
            }  # fmt: skip
            for p in probes
        ],
    }
    (REPORT_DIR / "redteam.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"{len(probes)} distress notes probed against low={thresholds.low:.4f}.")
    print(f"  skip the LLM today:            {len(skipped)}")
    print(f"  skip with segment scoring:     {len(still)}")
    print(f"  lowest whole-note score:       {min(p.whole for p in probes):.4f}")
    print(f"  lowest worst-segment score:    {min(p.worst_segment for p in probes):.4f}")
    if skipped:
        print("\nWorst offenders:")
        for p in sorted(skipped, key=lambda p: p.whole)[:5]:
            print(f"  {p.whole:.4f}  (segment {p.worst_segment:.4f})  {p.text[:64]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
