"""The test set is looked at once, and every look is written down.

A test set loses its value each time someone looks at it and then changes
something. The usual protection is discipline, and discipline does not leave a
record. This module does:

* scoring the test split appends a line to ``reports/test_ledger.json`` — which
  artifact was measured, on which data, by which evaluation code;
* a second look is refused unless it states a reason, and the reason is recorded
  next to the first;
* the artifact and the data must be committed before the test set is spent, so
  that what was measured can always be recovered.

A ledger with more than one look is not a failure. An undisclosed second look is.
The model card reports the ledger as it stands.

Re-rendering the report from saved predictions is not a look — no test row is
read — but it is recorded too. If the evaluation code changed after the look, the
numbers in the report came from different code than the one hashed at the look,
and that should be visible rather than something a reader has to notice.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

__all__ = [
    "EVALUATOR_FILES",
    "LEDGER_PATH",
    "MEASURED_INPUTS",
    "LedgerError",
    "Look",
    "Render",
    "append_look",
    "append_render",
    "authorize",
    "files_sha256",
    "read_looks",
    "read_renders",
    "uncommitted",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "reports" / "test_ledger.json"

#: The code that decides what an evaluation reports. Hashed into every look, so
#: the committed evaluator can be checked against the one that produced a number.
EVALUATOR_FILES = (
    "src/dc/evaluate.py",
    "src/dc/metrics.py",
    "src/dc/cascade.py",
    "src/dc/calibration.py",
)

#: Must be committed before the test set is spent.
MEASURED_INPUTS = ("artifacts", "data")


class LedgerError(RuntimeError):
    """The test set may not be looked at under these conditions."""


@dataclass(frozen=True)
class Look:
    number: int
    on: str
    commit: str | None
    artifact_sha256: str
    dataset_sha256: str
    evaluator_sha256: str
    reason: str


@dataclass(frozen=True)
class Render:
    """A report re-rendered from saved predictions. Not a look: no test row is read."""

    on: str
    predictions_sha256: str
    artifact_sha256: str
    evaluator_sha256: str
    note: str


def files_sha256(paths: Sequence[Path]) -> str:
    """One hash over named files, in order. A missing file hashes as missing."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LedgerError(f"{path.name} is not a ledger")
    return cast("dict[str, Any]", raw)


def _write(path: Path, looks: Sequence[Look], renders: Sequence[Render]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "looks": [asdict(entry) for entry in looks],
        "renders": [asdict(entry) for entry in renders],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_looks(path: Path = LEDGER_PATH) -> list[Look]:
    return [Look(**entry) for entry in cast("list[dict[str, Any]]", _read(path).get("looks", []))]


def read_renders(path: Path = LEDGER_PATH) -> list[Render]:
    entries = cast("list[dict[str, Any]]", _read(path).get("renders", []))
    return [Render(**entry) for entry in entries]


def authorize(looks: Sequence[Look], *, again: str | None) -> str:
    """The reason to record for this look, or refuse.

    The first look needs no justification. Every later one does, because a second
    look after changing something is exactly how a test set quietly becomes a
    validation set.
    """
    if not looks:
        return (again or "").strip() or "first look"
    if again is None or not again.strip():
        previous = "; ".join(f"#{look.number} on {look.on}: {look.reason}" for look in looks)
        raise LedgerError(
            f"The test set has already been evaluated {len(looks)} time(s) ({previous}). "
            "Re-render the report from saved predictions with `--render`, or look again "
            'with `--again "<reason>"` — the reason is recorded, and the model card must '
            "disclose it."
        )
    return again.strip()


def append_look(look: Look, path: Path = LEDGER_PATH) -> None:
    looks = read_looks(path)
    if look.number != len(looks) + 1:
        raise LedgerError(f"look #{look.number} does not follow the {len(looks)} recorded")
    _write(path, [*looks, look], read_renders(path))


def append_render(render: Render, path: Path = LEDGER_PATH) -> None:
    looks = read_looks(path)
    if not looks:
        raise LedgerError("nothing to re-render: the test set has not been scored")
    _write(path, looks, [*read_renders(path), render])


def uncommitted(paths: Sequence[str], repo: Path = REPO_ROOT) -> list[str]:
    """Uncommitted changes under ``paths``. Unknown state counts as uncommitted."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ["<git status unavailable>"]
    return [line for line in status.splitlines() if line.strip()]


def today() -> str:
    return date.today().isoformat()
