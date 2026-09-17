"""The adversarial probe set: notes written to slip past the skip band.

Every note in ``data/redteam.jsonl`` is distress by construction, so there is only
one failure mode to look for — a note that skips the LLM. The set lives here
rather than inside the script that reports on it because it is now read twice:
once by :mod:`dc.train`, which will not let ``low`` rise above what these notes
score, and once by ``scripts/redteam.py``, which checks that it did not.

These notes are never training data. Fitting a threshold against them is not the
same as training on them — no weight sees them — but it does spend their
independence: after a fit constrained this way, the probe reports a guarantee
rather than a finding. Writing new attack notes is what makes it a probe again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PROBE_PATH", "ProbeNote", "load_probe_notes"]

PROBE_PATH = Path(__file__).resolve().parents[2] / "data" / "redteam.jsonl"


@dataclass(frozen=True)
class ProbeNote:
    id: str
    family: str
    text: str


def load_probe_notes(path: Path = PROBE_PATH) -> list[ProbeNote]:
    notes = [
        ProbeNote(id=str(raw["id"]), family=str(raw["family"]), text=str(raw["text"]))
        for raw in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    ids = [note.id for note in notes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate note id in the red-team set")
    if not notes:
        raise ValueError(f"no probe notes in {path}")
    return notes
