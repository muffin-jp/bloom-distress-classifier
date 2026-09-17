# Latency

A snapshot from Darwin arm64, Python 3.12.13. One note at a time, sequentially, after warm-up.

| Path | p50 | p95 | Mean | Max | Notes |
| --- | --- | --- | --- | --- | --- |
| **Local model, as shipped** — split, embed all segments, score, route | 5.9 ms | 16.9 ms | 8.0 ms | 79.4 ms | 200 |
| Local model, whole note only — the rule this replaced | 4.7 ms | 6.5 ms | 4.9 ms | 11.4 ms | 200 |
| `claude-haiku-4-5` — production call | not measured | | | | run with `--llm-calls N --yes` |

A note becomes **3.1 texts** on average — itself, its sentences, and sliding word windows — embedded in one batched call. That costs **+1.2 ms** at p50 against scoring the note alone, which is what the red-team fix is worth paying.

The local path runs in-process on CPU. The LLM path includes the network round-trip from this machine, which is not where production runs.
