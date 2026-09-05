"""Run the teacher over candidates to order the human review queue.

This does **not** label anything. It records what `claude-haiku-4-5` thought, so
a reviewer reads the contested and unstable rows first instead of working
alphabetically. The label is still assigned by a person in `scripts/review.py`.

Each candidate is classified ``--votes`` times (default 3). Repeat runs are the
cheap trick that earns its keep: where the teacher flips its own answer on
identical text, the example is genuinely ambiguous, and those are the rows worth
a human's full attention. At roughly $0.0005 a call, three votes over a thousand
candidates costs about $1.50.

Usage::

    uv run --extra teacher python scripts/propose_labels.py --dry-run
    uv run --extra teacher python scripts/propose_labels.py --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from anthropic import AsyncAnthropic

from dc.candidates import Candidate, load_candidates
from dc.teacher import (
    CONFIDENCE_SCHEMA,
    PRODUCTION_SYSTEM_PROMPT,
    TEACHER_MODEL,
    build_user_message,
    check_prompt_drift,
    parse_reply,
    text_from,
)

CANDIDATE_DIR = Path(__file__).resolve().parents[1] / "data" / "candidates"

# Anthropic first-party rates for claude-haiku-4-5, USD per million tokens.
INPUT_USD_PER_MTOK = 1.00
OUTPUT_USD_PER_MTOK = 5.00
# Measured shape of one call: the system prompt and schema dominate, the note is
# capped at 200 chars, and the reply is a two-field JSON object.
EST_INPUT_TOKENS = 420
EST_OUTPUT_TOKENS = 20

MAX_CONCURRENCY = 8
MAX_ATTEMPTS = 4


def estimate_usd(n_calls: int) -> float:
    return n_calls * (
        EST_INPUT_TOKENS * INPUT_USD_PER_MTOK / 1_000_000
        + EST_OUTPUT_TOKENS * OUTPUT_USD_PER_MTOK / 1_000_000
    )


async def _classify_once(client: AsyncAnthropic, candidate: Candidate) -> tuple[int, float | None]:
    """One teacher call, with backoff on transient failures."""
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=128,
                system=PRODUCTION_SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": CONFIDENCE_SCHEMA}},
                # Sampling is left at the API default rather than pinned to 0:
                # the point of repeat votes is to surface where the teacher is
                # unstable, and temperature 0 would hide exactly that.
                messages=[
                    {
                        "role": "user",
                        "content": build_user_message(candidate.feeling.value, candidate.free_text),
                    }
                ],
            )
            return parse_reply(text_from(response))
        except Exception as exc:  # noqa: BLE001 - retry every transient failure alike
            last = exc
            await asyncio.sleep(min(2**attempt + random.random(), 30.0))
    raise RuntimeError(f"teacher failed for {candidate.id} after {MAX_ATTEMPTS} attempts: {last}")


async def _classify_candidate(
    client: AsyncAnthropic,
    candidate: Candidate,
    votes: int,
    limiter: asyncio.Semaphore,
) -> Candidate:
    async with limiter:
        results = [await _classify_once(client, candidate) for _ in range(votes)]
    labels = tuple(label for label, _ in results)
    confidences = [c for _, c in results if c is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    return candidate.model_copy(
        update={"teacher_votes": labels, "teacher_confidence": mean_confidence}
    )


async def run(paths: list[Path], votes: int) -> list[Candidate]:
    candidates = load_candidates(*paths)
    client = AsyncAnthropic()
    limiter = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [_classify_candidate(client, c, votes, limiter) for c in candidates]
    return list(await asyncio.gather(*tasks))


def write_back(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = candidate.model_dump(mode="json", exclude_defaults=False)
            payload["teacher_votes"] = list(candidate.teacher_votes)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="cost estimate only, no API calls")
    parser.add_argument("--yes", action="store_true", help="required to actually spend money")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    drift = check_prompt_drift()
    if drift is not None:
        raise SystemExit(f"ERROR: {drift}")

    paths: list[Path] = list(args.paths) or sorted(CANDIDATE_DIR.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No candidate files found in {CANDIDATE_DIR}.")

    candidates = load_candidates(*paths)
    n_calls = len(candidates) * args.votes
    print(f"{len(candidates)} candidate(s) x {args.votes} vote(s) = {n_calls} calls")
    print(f"Estimated cost: ${estimate_usd(n_calls):.2f} at {TEACHER_MODEL} rates.")

    if args.dry_run or not args.yes:
        print("Nothing sent. Re-run with --yes to spend.")
        return

    updated = asyncio.run(run(paths, args.votes))
    by_path: dict[Path, list[Candidate]] = {path: [] for path in paths}
    index = {c.id: c for c in updated}
    for path in paths:
        for original in load_candidates(path):
            by_path[path].append(index[original.id])
    for path, rows in by_path.items():
        write_back(path, rows)
        print(f"Updated {path.name}: {len(rows)} row(s).")

    contested = sum(1 for c in updated if c.is_contested)
    unstable = sum(1 for c in updated if c.is_unstable)
    print(f"{contested} contested (teacher disagrees with the author) — review these first.")
    print(f"{unstable} unstable (teacher flipped across runs) — the genuinely ambiguous ones.")


if __name__ == "__main__":
    main()
