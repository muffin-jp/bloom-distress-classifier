"""Run the teacher over candidates to order the human review queue.

This does **not** label anything. It records what `claude-haiku-4-5` thought, so
a reviewer reads the contested and unstable rows first instead of working
alphabetically. The label is still assigned by a person in `scripts/review.py`.

Each candidate is classified ``--votes`` times (default 3). Repeat runs are the
cheap trick that earns its keep: where the teacher flips its own answer on
identical text, the example is genuinely ambiguous, and those are the rows worth
a human's full attention.

Three things keep a long run from wasting money:

* a **preflight** call validates the request shape before the batch starts, so a
  malformed schema costs one call rather than a run;
* only *transient* failures are retried — a 400 fails fast instead of backing
  off four times against an error that can never succeed;
* results are written even when some candidates fail, and candidates that
  already carry votes are skipped, so an interrupted run resumes instead of
  starting over.

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

from dc.api import is_retryable
from dc.candidates import Candidate, load_candidates
from dc.env import load_env
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


async def _call_teacher(
    client: AsyncAnthropic, feeling: str, free_text: str
) -> tuple[int, float | None]:
    """One teacher call. Raises on failure; the caller decides about retrying."""
    response = await client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=128,
        system=PRODUCTION_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": CONFIDENCE_SCHEMA}},
        # Sampling is left at the API default rather than pinned to 0: the point
        # of repeat votes is to surface where the teacher is unstable, and
        # temperature 0 would hide exactly that.
        messages=[{"role": "user", "content": build_user_message(feeling, free_text)}],
    )
    return parse_reply(text_from(response))


async def _classify_once(client: AsyncAnthropic, candidate: Candidate) -> tuple[int, float | None]:
    """One vote, retrying only failures that could plausibly succeed next time."""
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await _call_teacher(client, candidate.feeling.value, candidate.free_text)
        except Exception as exc:
            last = exc
            # A malformed request, a bad key, or an unparseable reply will fail
            # identically on every attempt. Backing off just hides the error.
            if not is_retryable(exc):
                raise
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(min(2**attempt + random.random(), 30.0))
    raise RuntimeError(f"teacher failed for {candidate.id} after {MAX_ATTEMPTS} attempts: {last}")


async def _classify_candidate(
    client: AsyncAnthropic, candidate: Candidate, votes: int, limiter: asyncio.Semaphore
) -> Candidate:
    async with limiter:
        results = [await _classify_once(client, candidate) for _ in range(votes)]
    labels = tuple(label for label, _ in results)
    confidences = [c for _, c in results if c is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    return candidate.model_copy(
        update={"teacher_votes": labels, "teacher_confidence": mean_confidence}
    )


async def preflight(client: AsyncAnthropic) -> None:
    """One real call before the batch, so a bad request costs $0.0005 not $1.

    The 400 that motivated this ("For 'number' type, properties maximum, minimum
    are not supported") was a property of every request in the run. Discovering
    that on call 1 instead of after N retries across N candidates is the
    difference between a typo and a wasted afternoon.
    """
    print("Preflight: sending one call to validate the request shape ...")
    label, confidence = await _call_teacher(client, "tired", "preflight check, please ignore")
    print(f"  ok — label {label}, confidence {confidence}")


async def run(
    batches: dict[Path, list[Candidate]], votes: int
) -> tuple[dict[Path, list[Candidate]], list[str]]:
    """Classify every pending candidate, keeping whatever succeeds.

    ``return_exceptions=True`` is the important part: one bad candidate must not
    discard the work already paid for. Failures are reported and their
    candidates are written back untouched, so a re-run picks them up.
    """
    client = AsyncAnthropic()
    await preflight(client)

    limiter = asyncio.Semaphore(MAX_CONCURRENCY)
    updated: dict[Path, list[Candidate]] = {}
    failures: list[str] = []

    for path, candidates in batches.items():
        pending = [c for c in candidates if not c.teacher_votes]
        results = await asyncio.gather(
            *(_classify_candidate(client, c, votes, limiter) for c in pending),
            return_exceptions=True,
        )
        done: dict[str, Candidate] = {}
        for candidate, result in zip(pending, results, strict=True):
            if isinstance(result, Candidate):
                done[candidate.id] = result
            else:
                failures.append(f"{candidate.id}: {result}")
        updated[path] = [done.get(c.id, c) for c in candidates]
        print(f"  {path.name}: {len(done)}/{len(pending)} pending candidate(s) classified")

    return updated, failures


def write_back(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = candidate.model_dump(mode="json")
            payload["teacher_votes"] = list(candidate.teacher_votes)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="cost estimate only, no API calls")
    parser.add_argument("--yes", action="store_true", help="required to actually spend money")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-classify candidates that already have votes (default: skip them)",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    load_env()

    drift = check_prompt_drift()
    if drift is not None:
        raise SystemExit(f"ERROR: {drift}")

    paths: list[Path] = list(args.paths) or sorted(CANDIDATE_DIR.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No candidate files found in {CANDIDATE_DIR}.")

    batches: dict[Path, list[Candidate]] = {}
    for path in paths:
        loaded = load_candidates(path)
        if args.force:
            loaded = [c.model_copy(update={"teacher_votes": ()}) for c in loaded]
        batches[path] = loaded

    total = sum(len(rows) for rows in batches.values())
    pending = sum(1 for rows in batches.values() for c in rows if not c.teacher_votes)
    n_calls = pending * args.votes
    print(f"{total} candidate(s), {pending} without votes.")
    print(f"{pending} x {args.votes} vote(s) = {n_calls} calls")
    print(f"Estimated cost: ${estimate_usd(n_calls):.2f} at {TEACHER_MODEL} rates.")

    if pending == 0:
        print("Nothing to do. Use --force to re-classify.")
        return
    if args.dry_run or not args.yes:
        print("Nothing sent. Re-run with --yes to spend.")
        return

    updated, failures = asyncio.run(run(batches, args.votes))
    for path, rows in updated.items():
        write_back(path, rows)
        print(f"Updated {path.name}: {len(rows)} row(s).")

    everything = [c for rows in updated.values() for c in rows]
    contested = sum(1 for c in everything if c.is_contested)
    unstable = sum(1 for c in everything if c.is_unstable)
    print(f"{contested} contested (teacher disagrees with the author) — review these first.")
    print(f"{unstable} unstable (teacher flipped across runs) — the genuinely ambiguous ones.")

    if failures:
        print(f"\n{len(failures)} candidate(s) failed and kept their previous state:")
        for failure in failures[:10]:
            print(f"  {failure}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        print("Re-run to retry just those — candidates with votes are skipped.")


if __name__ == "__main__":
    main()
