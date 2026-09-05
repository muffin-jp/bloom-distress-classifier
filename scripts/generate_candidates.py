"""Expand the curated taxonomy into more candidates, for human review.

The curated pack is a *taxonomy*: one or two examples of each way a game
complaint and a real crisis can look alike. This script asks a model for more
sentences in each mode, so the dataset gets volume without the taxonomy getting
blurrier.

Two rules keep the expansion honest:

**A variant inherits its seed's ``origin_id``.** Variants of the same mode are
near-paraphrases of each other, so the whole family must land on one side of the
train/test line. The consequence is deliberate and worth stating: because modes
do not split, the test set contains *held-out modes* rather than held-out
rephrasings of modes it has already seen. That is a harder and more honest test
than the alternative, which would inflate every number.

**A variant is a candidate, never a row.** The generator proposes text; it does
not get to decide the label. ``scripts/review.py`` is the only way in.

Usage::

    uv run --extra teacher python scripts/generate_candidates.py --dry-run
    uv run --extra teacher python scripts/generate_candidates.py --per-seed 6 --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic

from dc.api import is_retryable
from dc.candidates import Candidate, load_candidates
from dc.env import load_env
from dc.schema import Provenance
from dc.teacher import text_from

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "candidates"
CURATED = DATA_DIR / "curated.jsonl"
GENERATED = DATA_DIR / "generated.jsonl"

GENERATOR_MODEL = "claude-opus-5"
# Anthropic first-party rates for claude-opus-5, USD per million tokens.
INPUT_USD_PER_MTOK = 5.00
OUTPUT_USD_PER_MTOK = 25.00
EST_INPUT_TOKENS = 700
EST_OUTPUT_PER_VARIANT = 45

MAX_CONCURRENCY = 4
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You write synthetic training data for a safety classifier used by a mobile puzzle game. After clearing a stage, a player picks a feeling and may add a short note. The classifier decides whether that note signals serious personal distress or is ordinary game frustration.

You will be given one seed note and the confusion mode it illustrates. Write more notes in the SAME mode and the SAME frame, as a different player would phrase them.

Rules:
- Keep the frame fixed. A "frame:game" seed stays about the game. A "frame:life" seed stays about the player's life. Never blur the two — the whole value of this data is that the frames stay clean.
- Match the register: short, lowercase, typed on a phone right after playing. No therapy language, no polish.
- Vary the wording genuinely. Do not restate the seed with synonyms swapped.
- 200 characters maximum, one sentence or two.
- No real names, places, or anything that could identify a person.

Respond with JSON only."""

VARIANTS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {"type": "string", "maxLength": 200},
        }
    },
    "required": ["variants"],
    "additionalProperties": False,
}


def estimate_usd(n_seeds: int, per_seed: int) -> float:
    input_cost = n_seeds * EST_INPUT_TOKENS * INPUT_USD_PER_MTOK / 1_000_000
    output_cost = n_seeds * per_seed * EST_OUTPUT_PER_VARIANT * OUTPUT_USD_PER_MTOK / 1_000_000
    return input_cost + output_cost


async def _variants_for(
    client: AsyncAnthropic, seed: Candidate, per_seed: int, limiter: asyncio.Semaphore
) -> list[str]:
    user = (
        f"Confusion mode: {seed.note or 'unspecified'}\n"
        f"Selected feeling: {seed.feeling.value}\n"
        f'Seed note: """{seed.free_text}"""\n\n'
        f"Write {per_seed} more notes in this mode and frame."
    )
    async with limiter:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.messages.create(
                    model=GENERATOR_MODEL,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    output_config={
                        "effort": "medium",
                        "format": {"type": "json_schema", "schema": VARIANTS_SCHEMA},
                    },
                    messages=[{"role": "user", "content": user}],
                )
                loaded: Any = json.loads(text_from(response))
                if not isinstance(loaded, dict):
                    raise ValueError("reply was not a JSON object")
                raw: Any = cast("dict[str, Any]", loaded).get("variants")
                if not isinstance(raw, list):
                    raise ValueError("no variants array in reply")
                out: list[str] = []
                for item in cast("list[Any]", raw):
                    if isinstance(item, str) and item.strip() and len(item.strip()) <= 200:
                        out.append(item.strip())
                return out
            except Exception as exc:
                # A malformed request or bad key fails identically every time.
                # Surface it now rather than backing off against a wall — and
                # rather than returning [] and quietly producing a short file.
                if not is_retryable(exc):
                    raise
                if attempt == MAX_ATTEMPTS - 1:
                    print(f"  ! giving up on {seed.id} after transient failures: {exc}")
                    return []
                await asyncio.sleep(min(2**attempt + random.random(), 20.0))
    return []


async def run(seeds: list[Candidate], per_seed: int) -> list[Candidate]:
    client = AsyncAnthropic()
    limiter = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await asyncio.gather(
        *(_variants_for(client, seed, per_seed, limiter) for seed in seeds)
    )
    generated: list[Candidate] = []
    for seed, texts in zip(seeds, results, strict=True):
        for index, text in enumerate(texts, start=1):
            generated.append(
                Candidate(
                    id=f"{seed.id}-v{index:02d}",
                    # Inherits the seed's group: variants never straddle the split.
                    origin_id=seed.origin_id,
                    feeling=seed.feeling,
                    free_text=text,
                    category=seed.category,
                    provenance=Provenance.LLM_AUGMENTED,
                    note=seed.note,
                )
            )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", type=int, default=6)
    parser.add_argument("--seeds", type=Path, default=CURATED)
    parser.add_argument("--out", type=Path, default=GENERATED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="required to actually spend money")
    args = parser.parse_args()
    load_env()

    seeds = load_candidates(args.seeds)
    print(f"{len(seeds)} seed(s) x {args.per_seed} variants = ~{len(seeds) * args.per_seed} rows")
    print(f"Estimated cost: ${estimate_usd(len(seeds), args.per_seed):.2f} on {GENERATOR_MODEL}.")

    if args.dry_run or not args.yes:
        print("Nothing sent. Re-run with --yes to spend.")
        return

    generated = asyncio.run(run(seeds, args.per_seed))
    with args.out.open("w", encoding="utf-8") as handle:
        for candidate in generated:
            handle.write(
                json.dumps(
                    candidate.model_dump(mode="json", exclude={"teacher_votes"}),
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(generated)} candidate(s) to {args.out}.")
    print("None of them are training data until reviewed: `make review`.")


if __name__ == "__main__":
    main()
