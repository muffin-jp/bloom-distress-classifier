# Latency

A snapshot from Darwin arm64, Python 3.12.13. One note at a time, sequentially, after warm-up.

| Path | p50 | p95 | Mean | Max | Notes |
| --- | --- | --- | --- | --- | --- |
| **Local model, as shipped** — split, embed all segments, score, route | 5.7 ms | 16.1 ms | 7.3 ms | 21.0 ms | 200 |
| Local model, whole note only — the rule this replaced | 4.7 ms | 5.2 ms | 4.9 ms | 9.9 ms | 200 |
| `claude-haiku-4-5` — production call | 784.0 ms | 1343.4 ms | 901.8 ms | 2252.8 ms | 20 |

A note becomes **3.1 texts** on average — itself, its sentences, and sliding word windows — embedded in one batched call. That costs **+1.0 ms** at p50 against scoring the note alone, which is what the red-team fix is worth paying.

The local path runs in-process on CPU. The LLM path includes the network round-trip from this machine, which is not where production runs.

Under the fitted cascade, 86% of notes escalate. Every note pays the local path; only those pay the call. Expected mean latency per note: **778 ms**, against 902 ms when every note calls the LLM.
