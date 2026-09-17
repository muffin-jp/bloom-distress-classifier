"""Latency of the local path, and of the LLM call it replaces.

Kept out of the training report on purpose: timings vary run to run and machine
to machine, and the training report must reproduce byte for byte. This writes a
snapshot to ``reports/latency.md`` instead.

The local path is measured the way a request would pay for it — one note at a
time, sequentially, after warm-up, through ``dc.serve`` and the artifact's own
segmentation and scope rules, so what is timed is what ships. A note is not one
embedding: it is the note plus each of its sentences and sliding word windows, in
one batched call. The whole-note path is timed alongside it, because the
difference between the two is what the red-team fix costs. The LLM path makes real
``claude-haiku-4-5`` calls with production's exact prompt and schema, so it costs
money and needs ``--yes``. It is timed from this machine, which is not where
production runs; read it as indicative.

Usage::

    uv run --extra embed python scripts/bench_latency.py
    uv run --extra embed --extra teacher python scripts/bench_latency.py --llm-calls 20 --yes
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from dc.artifact import is_servable, load, read_scope, read_segmentation
from dc.cascade import Thresholds, route
from dc.features import build_features, load_embedder
from dc.serve import score_notes
from dc.splits import load_all_rows, load_assignment, split_rows

REPORT = Path(__file__).resolve().parents[1] / "reports" / "latency.md"
USD_PER_CALL = 0.0005  # measured shape of a production classify call on claude-haiku-4-5


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "n": float(values.size),
    }


def _time(one: Any, texts: list[str], warmup: int) -> dict[str, float]:
    for text in texts[:warmup]:
        one(text)
    samples: list[float] = []
    for text in texts:
        start = time.perf_counter()
        one(text)
        samples.append((time.perf_counter() - start) * 1000)
    return percentiles(samples)


def bench_local(texts: list[str], warmup: int) -> tuple[dict[str, float], dict[str, float], float]:
    """The shipped path, the whole-note path it replaced, and segments per note."""
    artifact = load()
    if not is_servable(artifact.metadata):
        raise SystemExit("The artifact has no thresholds. Run `make train` first.")
    bands = artifact.metadata["thresholds"]
    thresholds = Thresholds(float(bands["low"]), float(bands["high"]))
    segmentation = read_segmentation(artifact.metadata)
    scope = read_scope(artifact.metadata)
    embedder = load_embedder()
    by_text = {
        row.free_text: row for row in split_rows(load_all_rows(), load_assignment(), "train")
    }

    def shipped(text: str) -> None:
        [note] = score_notes(
            [text], artifact.model, embedder, segmentation=segmentation, scope=scope
        )
        note.route(thresholds)

    def whole_note(text: str) -> None:
        features = build_features([by_text[text]], embedder, include_feeling=False)
        route(float(artifact.model.predict_proba_features(features)[0]), thresholds)

    segments = float(np.mean([len(segmentation.split(text)) for text in texts]))
    return _time(shipped, texts, warmup), _time(whole_note, texts, warmup), segments


def bench_llm(texts: list[str], feelings: list[str]) -> dict[str, float]:
    from anthropic import Anthropic

    from dc.env import load_env
    from dc.teacher import (
        PRODUCTION_SCHEMA,
        PRODUCTION_SYSTEM_PROMPT,
        TEACHER_MODEL,
        build_user_message,
    )

    load_env()
    client = Anthropic()
    samples: list[float] = []
    for text, feeling in zip(texts, feelings, strict=True):
        start = time.perf_counter()
        client.messages.create(
            model=TEACHER_MODEL,
            max_tokens=64,
            system=PRODUCTION_SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": PRODUCTION_SCHEMA}},
            messages=[{"role": "user", "content": build_user_message(feeling, text)}],
        )
        samples.append((time.perf_counter() - start) * 1000)
    return percentiles(samples)


def render(
    local: dict[str, float],
    whole: dict[str, float],
    segments: float,
    llm: dict[str, float] | None,
    escalate: float | None,
) -> str:
    def row(name: str, stats: dict[str, float]) -> str:
        return (
            f"| {name} | {stats['p50']:.1f} ms | {stats['p95']:.1f} ms "
            f"| {stats['mean']:.1f} ms | {stats['max']:.1f} ms | {int(stats['n'])} |"
        )

    lines = [
        "# Latency",
        "",
        f"A snapshot from {platform.system()} {platform.machine()}, Python "
        f"{platform.python_version()}. One note at a time, sequentially, after warm-up.",
        "",
        "| Path | p50 | p95 | Mean | Max | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
        row("**Local model, as shipped** — split, embed all segments, score, route", local),
        row("Local model, whole note only — the rule this replaced", whole),
    ]
    if llm is None:
        lines.append(
            "| `claude-haiku-4-5` — production call | not measured | | | | "
            "run with `--llm-calls N --yes` |"
        )
    else:
        lines.append(row("`claude-haiku-4-5` — production call", llm))
    lines += [
        "",
        f"A note becomes **{segments:.1f} texts** on average — itself, its sentences, and "
        f"sliding word windows — embedded in one batched call. That costs "
        f"**{local['p50'] - whole['p50']:+.1f} ms** at p50 against scoring the note alone, "
        "which is what the red-team fix is worth paying.",
        "",
        "The local path runs in-process on CPU. The LLM path includes the network "
        "round-trip from this machine, which is not where production runs.",
    ]
    if llm is not None and escalate is not None:
        expected = local["mean"] + escalate * llm["mean"]
        lines += [
            "",
            f"Under the fitted cascade, {escalate:.0%} of notes escalate. Every note pays "
            f"the local path; only those pay the call. Expected mean latency per note: "
            f"**{expected:.0f} ms**, against {llm['mean']:.0f} ms when every note calls "
            "the LLM.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--llm-calls", type=int, default=0)
    parser.add_argument("--yes", action="store_true", help="required for --llm-calls")
    args = parser.parse_args()

    train = split_rows(load_all_rows(), load_assignment(), "train")
    sample = train[: args.notes]
    texts = [row.free_text for row in sample]

    print(f"Timing the local path over {len(texts)} note(s) ...")
    local, whole, segments = bench_local(texts, args.warmup)
    print(f"  shipped (segmented) p50 {local['p50']:.1f} ms · p95 {local['p95']:.1f} ms")
    print(f"  whole note only     p50 {whole['p50']:.1f} ms · p95 {whole['p95']:.1f} ms")
    print(f"  segments per note   {segments:.1f}")

    llm: dict[str, float] | None = None
    if args.llm_calls > 0:
        n = min(args.llm_calls, len(sample))
        print(f"{n} LLM call(s), estimated ${n * USD_PER_CALL:.3f}.")
        if not args.yes:
            print("Nothing sent. Re-run with --yes to spend.")
        else:
            llm = bench_llm(texts[:n], [row.feeling.value for row in sample[:n]])
            print(f"  p50 {llm['p50']:.0f} ms · p95 {llm['p95']:.0f} ms")

    metadata: dict[str, Any] = load().metadata
    escalate = metadata["thresholds"]["validation"]["escalate_share"]
    REPORT.write_text(render(local, whole, segments, llm, float(escalate)), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(REPORT.parents[1])}.")


if __name__ == "__main__":
    main()
