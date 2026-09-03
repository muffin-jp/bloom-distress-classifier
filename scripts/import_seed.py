"""Import the golden cases from bloom-langgraph into data/seed.jsonl.

These 44 rows are the release gate already running in the companion repo's CI.
They arrive here to be *measured against*, never trained on — ``dc.splits``
forces every one of them into the test set.

Two groups are dropped, and both drops are deliberate (BUILD_SPEC.md §4.1):

* the single ``regeneration`` case, which exercises the reflection loop rather
  than the routing decision; and
* the six chip-only ``normal-feeling`` cases, which carry no ``freeText``. The
  production graph never invokes the classifier without free text, so these sit
  outside the model's input distribution entirely. Keeping them would score a
  path the model does not serve.

Usage::

    uv run python scripts/import_seed.py [--reviewed-by NAME] [--source PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, cast

from dc.schema import Category, Feeling, Provenance, Row

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent / "bloom-langgraph" / "evals" / "dataset.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "seed.jsonl"

#: Not a routing case — it forces a regeneration in the companion repo's harness.
EXCLUDED_CATEGORY = "regeneration"


def _read_source(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw: Any = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name}:{lineno}: expected a JSON object")
        rows.append(cast("dict[str, Any]", raw))
    return rows


def convert(raw: dict[str, Any], *, reviewed_by: str, reviewed_on: date) -> Row:
    """Map one companion-repo case onto our row schema.

    ``expectedPath`` is the companion repo's ground truth; we re-derive the label
    from ``category`` and assert the two agree, so a drift in either repo shows
    up here rather than as a mysterious metric change later.
    """
    category = Category(str(raw["category"]))
    label = 1 if category is Category.DISTRESS else 0
    expected_path = str(raw["expectedPath"])
    if (expected_path == "support") != bool(label):
        raise ValueError(
            f"{raw['id']}: expectedPath={expected_path!r} disagrees with "
            f"category={category.value!r}. One of the two repos has drifted."
        )
    return Row(
        id=str(raw["id"]),
        # A seed row is its own origin: nothing was paraphrased from anything.
        origin_id=str(raw["id"]),
        feeling=Feeling(str(raw["feeling"])),
        free_text=str(raw["freeText"]).strip(),
        label=label,
        category=category,
        provenance=Provenance.SEED,
        reviewed_by=reviewed_by,
        reviewed_on=reviewed_on,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--reviewed-by",
        default="uv",
        help="who vouches for these labels (they are inherited, not re-derived)",
    )
    args = parser.parse_args()

    source: Path = args.source
    if not source.exists():
        raise SystemExit(
            f"Source dataset not found at {source}. Pass --source, or clone "
            "bloom-langgraph next to this repo."
        )

    raw_rows = _read_source(source)
    kept: list[Row] = []
    dropped_no_text = 0
    dropped_excluded = 0
    today = date.today()

    for raw in raw_rows:
        if str(raw.get("category")) == EXCLUDED_CATEGORY:
            dropped_excluded += 1
            continue
        if not str(raw.get("freeText") or "").strip():
            dropped_no_text += 1
            continue
        kept.append(convert(raw, reviewed_by=args.reviewed_by, reviewed_on=today))

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(f"Read {len(raw_rows)} case(s) from {source}.")
    print(f"  dropped {dropped_excluded} ({EXCLUDED_CATEGORY}), {dropped_no_text} chip-only.")
    print(f"Wrote {len(kept)} golden row(s) to {out.relative_to(REPO_ROOT)}.")
    counts = Counter(row.category.value for row in kept)
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    print(f"  positive: {sum(row.label for row in kept)}")


if __name__ == "__main__":
    main()
