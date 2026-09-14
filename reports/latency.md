# Latency

A snapshot from Darwin arm64, Python 3.12.13. One note at a time, sequentially, after warm-up.

| Path | p50 | p95 | Mean | Max | Notes |
| --- | --- | --- | --- | --- | --- |
| Local model — embed, score, route | 4.8 ms | 18.6 ms | 18.0 ms | 625.8 ms | 200 |
| `claude-haiku-4-5` — production call | not measured | | | | run with `--llm-calls N --yes` |

The local path runs in-process on CPU. The LLM path includes the network round-trip from this machine, which is not where production runs.
