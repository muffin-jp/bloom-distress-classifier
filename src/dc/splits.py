"""The frozen train/test assignment.

Two rules decide every row, and both exist to stop a specific, silent failure:

**Golden rows are always test.** The 44 imported cases are the release gate
already running in ``bloom-langgraph``'s CI. Training on them would keep that
gate reporting 100% while measuring nothing, and it would fail green — the worst
failure mode available.

**Groups never straddle the line.** A paraphrase shares its source row's
``origin_id``; the whole group goes to one side. If a paraphrase of a training
row landed in test, every test metric would be inflated and the inflation would
be invisible.

Why a hashed order rather than ``random.shuffle``
-------------------------------------------------
``splits.json`` is a committed artifact, so its stability matters more than its
randomness. Groups are ordered by ``sha256(salt:origin_id)`` and the first
slice of each category goes to test. That is deterministic across Python
versions and — crucially — **adding rows in milestone 2 does not reshuffle the
groups already assigned**, because a group's hash does not depend on the rest of
the file. Only the cut point moves. A seeded ``shuffle`` would reassign
everything on every import, quietly changing what "test" means mid-project.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, cast

from dc.schema import Category, Row, load_dataset

__all__ = [
    "DEFAULT_SALT",
    "DEFAULT_TEST_FRACTION",
    "Split",
    "build_assignment",
    "check_leakage",
    "dominant_category",
    "drifted_groups",
    "group_rows",
    "load_all_rows",
    "load_assignment",
    "split_rows",
    "write_assignment",
]

Split = Literal["train", "test"]

#: Bump only to deliberately re-draw the split. Changing it invalidates every
#: number already reported, so it belongs in the model card when it moves.
DEFAULT_SALT = "bloom-distress-classifier/v1"
DEFAULT_TEST_FRACTION = 0.2

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_PATH = DATA_DIR / "labelled.jsonl"
SEED_PATH = DATA_DIR / "seed.jsonl"
SPLITS_PATH = DATA_DIR / "splits.json"


def group_rows(rows: list[Row]) -> dict[str, list[Row]]:
    """Bucket rows by ``origin_id``. A family always stays whole.

    Review can make a family heterogeneous: a reviewer looks at a generated
    variant, decides it drifted from its seed, and gives it a different
    category. That is review working, not data corruption — and it forces a
    choice between two things that cannot both hold:

    * a family never straddles the train/test line, and
    * a family has exactly one category, so stratification is exact.

    The first wins, always. A split family is silent, unfalsifiable metric
    inflation. Imperfect stratification is a rounding error in a class balance
    that gets reported anyway. So a drifted family is kept together and
    stratified by its dominant category; :func:`drifted_groups` reports the
    drift so it stays visible rather than becoming invisible.
    """
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.origin_id].append(row)
    return dict(groups)


def dominant_category(members: list[Row]) -> Category:
    """The category a family is stratified by: most common, ties broken by name.

    Deterministic on purpose — the split is a committed artifact, so the same
    rows must always produce the same assignment.
    """
    counts = Counter(member.category for member in members)
    top = max(counts.values())
    return min((c for c, n in counts.items() if n == top), key=lambda c: c.value)


def drifted_groups(rows: list[Row]) -> dict[str, list[str]]:
    """Families whose members disagree about category, after review.

    Not an error. Reported so a reviewer can see which generated variants
    wandered from their seed, and decide whether the seed's mode needs
    rewriting.
    """
    groups = group_rows(rows)
    return {
        origin_id: sorted({member.category.value for member in members})
        for origin_id, members in sorted(groups.items())
        if len({member.category for member in members}) > 1
    }


def _group_hash(origin_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{origin_id}".encode()).hexdigest()


def build_assignment(
    rows: list[Row],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    salt: str = DEFAULT_SALT,
) -> dict[str, Split]:
    """Assign every row id to ``"train"`` or ``"test"``.

    Golden groups go to test unconditionally. The rest are held out per category
    so the test set carries the same class mix as the data — including the hard
    ``game-frustration`` negatives, which are the ones worth measuring.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    groups = group_rows(rows)
    test_groups: set[str] = set()
    candidates: dict[Category, list[str]] = defaultdict(list)

    for origin_id, members in groups.items():
        if any(member.is_golden for member in members):
            test_groups.add(origin_id)
        else:
            candidates[dominant_category(members)].append(origin_id)

    for _category, origin_ids in sorted(candidates.items()):
        ordered = sorted(origin_ids, key=lambda oid: _group_hash(oid, salt))
        # Round up so a small category still contributes at least one test group
        # rather than silently disappearing from the test set.
        n_test = math.ceil(len(ordered) * test_fraction)
        test_groups.update(ordered[:n_test])

    assignment: dict[str, Split] = {}
    for origin_id, members in groups.items():
        split: Split = "test" if origin_id in test_groups else "train"
        for member in members:
            assignment[member.id] = split
    return assignment


def check_leakage(rows: list[Row], assignment: dict[str, Split]) -> list[str]:
    """Return every violation found, empty list if the assignment is sound.

    Returned rather than raised so a caller can report all problems at once;
    ``__main__`` and the tests turn a non-empty result into a failure.
    """
    problems: list[str] = []

    missing = sorted({row.id for row in rows} - set(assignment))
    if missing:
        problems.append(f"{len(missing)} row(s) have no split assignment: {missing[:5]}")

    unknown = sorted(set(assignment) - {row.id for row in rows})
    if unknown:
        problems.append(f"{len(unknown)} assigned id(s) are not in the dataset: {unknown[:5]}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.id in assignment:
            group_splits[row.origin_id].add(assignment[row.id])
    straddling = sorted(oid for oid, splits in group_splits.items() if len(splits) > 1)
    if straddling:
        problems.append(
            f"{len(straddling)} origin group(s) straddle the train/test line "
            f"(paraphrase leakage): {straddling[:5]}"
        )

    leaked_golden = sorted(
        row.id for row in rows if row.is_golden and assignment.get(row.id) != "test"
    )
    if leaked_golden:
        problems.append(
            f"{len(leaked_golden)} golden row(s) are not in test: {leaked_golden[:5]}. "
            "The golden cases are the release gate — training on them makes it "
            "report 100% while measuring nothing."
        )

    if not any(row.is_golden for row in rows):
        problems.append(
            "no golden rows in the dataset. The 44 imported release-gate cases "
            "must be present and in test; without them the golden-row check "
            "above passes vacuously and the gate measures nothing."
        )

    for split in ("train", "test"):
        if not any(value == split for value in assignment.values()):
            problems.append(f"the {split!r} split is empty")

    return problems


def split_rows(rows: list[Row], assignment: dict[str, Split], split: Split) -> list[Row]:
    """The rows on one side of the line, in file order."""
    return [row for row in rows if assignment.get(row.id) == split]


def write_assignment(assignment: dict[str, Split], path: Path = SPLITS_PATH) -> None:
    """Write splits.json sorted and newline-terminated, so diffs stay readable."""
    payload: dict[str, Any] = {
        "salt": DEFAULT_SALT,
        "test_fraction": DEFAULT_TEST_FRACTION,
        "assignment": dict(sorted(assignment.items())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_assignment(path: Path = SPLITS_PATH) -> dict[str, Split]:
    """Read a committed splits.json back."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "assignment" not in raw:
        raise ValueError(f"{path.name}: not a splits file (no 'assignment' key)")
    assignment: dict[str, Split] = {}
    for key, value in cast("dict[str, Any]", raw["assignment"]).items():
        if value not in ("train", "test"):
            raise ValueError(f"{path.name}: unknown split {value!r} for id {key!r}")
        assignment[str(key)] = value
    return assignment


def load_all_rows(seed_path: Path = SEED_PATH, dataset_path: Path = DATASET_PATH) -> list[Row]:
    """The whole dataset: the imported golden cases plus the reviewed rows.

    The two files are kept apart for good reasons — ``seed.jsonl`` is an
    immutable import from the companion repo, ``labelled.jsonl`` is review
    output that grows — but every consumer needs both. Reading only one is how
    the golden cases disappeared from the split without anything failing:
    ``check_leakage`` reported "all golden rows are in test" and it was true,
    because there were none. A guarantee that passes vacuously is worse than one
    that fails, so the merge lives here rather than in each caller.
    """
    rows: list[Row] = []
    if seed_path.exists():
        rows.extend(load_dataset(seed_path))
    if dataset_path.exists():
        rows.extend(load_dataset(dataset_path))
    if not rows:
        raise ValueError(
            f"No dataset found. Expected {seed_path.name} (run `make seed`) "
            f"and/or {dataset_path.name} (run `make review`)."
        )
    duplicates = [item for item, n in Counter(row.id for row in rows).items() if n > 1]
    if duplicates:
        raise ValueError(
            f"id(s) {sorted(duplicates)} appear in both {seed_path.name} and "
            f"{dataset_path.name}. The golden cases must not be re-reviewed."
        )
    return rows


def main() -> None:
    rows = load_all_rows()
    assignment = build_assignment(rows)

    if all(row.is_golden for row in rows):
        # Expected until milestone 2: the seed is golden by definition, so every
        # row is test-only and there is nothing to train on. Refuse to write a
        # degenerate splits.json rather than commit one that means nothing.
        print(
            "The dataset contains only golden rows, so the split is all-test and "
            "there is no train set yet. Build data/labelled.jsonl via `make review`, "
            "then re-run. Nothing written."
        )
        raise SystemExit(1)

    drifted = drifted_groups(rows)
    if drifted:
        print(f"{len(drifted)} family/families diverged during review (kept together):")
        for origin_id, categories in list(drifted.items())[:10]:
            print(f"  {origin_id}: {categories}")

    problems = check_leakage(rows, assignment)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)

    write_assignment(assignment)
    n_test = sum(1 for value in assignment.values() if value == "test")
    n_golden = sum(1 for row in rows if row.is_golden)
    print(f"Read {len(rows)} row(s) ({n_golden} golden + {len(rows) - n_golden} reviewed).")
    print(f"Wrote {SPLITS_PATH.name}: {len(assignment) - n_test} train / {n_test} test.")
    print(f"  of which {n_golden} golden row(s), all in test.")


if __name__ == "__main__":
    main()
