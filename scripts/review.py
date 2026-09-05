"""Human review: the only way a candidate becomes training data.

Candidates arrive with a *proposed* label — from the author's intent, or from
the teacher. Nothing here trusts either one. A row enters ``data/labelled.jsonl``
when a person has looked at it and put their name on it, and not before.

The queue is ordered so the rows that most need judgement come first: the ones
where the teacher contradicts the author, then the ones where the teacher
contradicted *itself* across runs, then the least confident. Working in that
order means the ambiguous cases get attention while attention is still fresh.

Every decision is appended immediately, so stopping halfway loses nothing and
re-running picks up where you left off.

Usage::

    uv run python scripts/review.py --reviewed-by uv
    uv run python scripts/review.py --stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dc.candidates import Candidate, load_candidates, promote, review_order
from dc.schema import Category, Row, load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CANDIDATE_DIR = DATA_DIR / "candidates"
LABELLED = DATA_DIR / "labelled.jsonl"

NEGATIVE_CATEGORIES = [c for c in Category if c is not Category.DISTRESS]


def _already_reviewed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row.id for row in load_dataset(path)}


def _append(path: Path, row: Row) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _render(candidate: Candidate, position: int, total: int) -> None:
    flags: list[str] = []
    if candidate.is_contested:
        flags.append("CONTESTED")
    if candidate.is_unstable:
        flags.append("UNSTABLE")
    header = f"[{position}/{total}] {candidate.id}"
    if flags:
        header += "  ** " + " ".join(flags) + " **"
    print("\n" + "=" * 78)
    print(header)
    print(f"  feeling : {candidate.feeling.value}")
    print(f"  note    : {candidate.free_text}")
    print(f"  mode    : {candidate.note or '-'}")
    print(f"  author  : {candidate.category.value} (label {candidate.proposed_label})")
    if candidate.teacher_votes:
        votes = "".join(str(v) for v in candidate.teacher_votes)
        confidence = (
            f"{candidate.teacher_confidence:.2f}"
            if candidate.teacher_confidence is not None
            else "-"
        )
        print(f"  teacher : label {candidate.teacher_label} votes {votes} conf {confidence}")


def _prompt_negative_category() -> Category | None:
    options = {str(index): category for index, category in enumerate(NEGATIVE_CATEGORIES, start=1)}
    print("  which negative category?")
    for key, category in options.items():
        print(f"    {key}) {category.value}")
    choice = input("  > ").strip()
    return options.get(choice)


def review(reviewed_by: str) -> None:
    candidates = load_candidates(*sorted(CANDIDATE_DIR.glob("*.jsonl")))
    done = _already_reviewed(LABELLED)
    queue = [c for c in review_order(candidates) if c.id not in done]

    if not queue:
        print(f"Nothing to review. {len(done)} row(s) already in {LABELLED.name}.")
        return

    print(f"{len(queue)} candidate(s) to review, {len(done)} already done.")
    print("  [enter]/a accept   d distress   n not distress   s skip   q quit")

    for position, candidate in enumerate(queue, start=1):
        _render(candidate, position, len(queue))
        action = input("  > ").strip().lower()

        if action in ("q", "quit"):
            print("Stopped. Progress is saved.")
            return
        if action in ("s", "skip"):
            continue

        if action in ("", "a", "accept"):
            category = candidate.category
        elif action in ("d", "1"):
            category = Category.DISTRESS
        elif action in ("n", "0"):
            if candidate.category is Category.DISTRESS:
                chosen = _prompt_negative_category()
                if chosen is None:
                    print("  unrecognised — skipping")
                    continue
                category = chosen
            else:
                category = candidate.category
        else:
            print("  unrecognised — skipping")
            continue

        _append(LABELLED, promote(candidate, category=category, reviewed_by=reviewed_by))
        flipped = " (flipped)" if category is not candidate.category else ""
        print(f"  -> {category.value}{flipped}")

    print(f"\nQueue empty. {LABELLED.name} is ready for `make splits`.")


def stats() -> None:
    candidates = load_candidates(*sorted(CANDIDATE_DIR.glob("*.jsonl")))
    done = _already_reviewed(LABELLED)
    pending = [c for c in candidates if c.id not in done]
    print(f"candidates : {len(candidates)}")
    print(f"reviewed   : {len(done)}")
    print(f"pending    : {len(pending)}")
    print(f"  contested: {sum(1 for c in pending if c.is_contested)}")
    print(f"  unstable : {sum(1 for c in pending if c.is_unstable)}")
    if not any(c.teacher_votes for c in candidates):
        print("\nThe teacher has not run yet, so the queue is only in id order.")
        print("Run scripts/propose_labels.py first to surface the ambiguous rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-by", default="", help="your name — it goes on every row")
    parser.add_argument("--stats", action="store_true", help="queue summary, review nothing")
    args = parser.parse_args()

    if args.stats:
        stats()
        return
    if not args.reviewed_by.strip():
        raise SystemExit("--reviewed-by is required: an unreviewed row is not training data.")
    review(args.reviewed_by)


if __name__ == "__main__":
    main()
