# bloom-safety-classifier

Distilling an LLM safety classifier into a small, explainable, threshold-tuned model —
dataset, evaluation, and model card.

Companion to [`bloom-langgraph`](../bloom-langgraph), whose *Guided Encouragement* feature routes
every free-text note through a `claude-haiku-4-5` call before generating anything: distress goes
to fixed, human-reviewed words; everything else goes to the encouragement branch. That decision
is unmeasured beyond routing accuracy on 44 cases, costs a model call on the hot path, and cannot
explain itself. This repo addresses all three.

**Not a diagnostic instrument.** It chooses between two pre-approved response *paths* for an
in-game message and makes no claim about anyone's mental state. See `BUILD_SPEC.md` §3.

## Status

Milestone 1 of 8 — scaffold, schema, splits. No model yet.

| # | Milestone | State |
| --- | --- | --- |
| 1 | Scaffold, strict schema, splits + leakage guards | ✅ |
| 2 | Dataset to 600+ reviewed rows | — |
| 3 | Baselines 0–3 + the Haiku teacher | — |
| 4 | Training, CV, calibration, ablations | — |
| 5 | Cost model, threshold fitting, cascade | — |
| 6 | Single test-set evaluation, bootstrap CIs | — |
| 7 | Explanations + model card | — |
| 8 | Integration PR into `bloom-langgraph` | — |

Results table goes here once milestone 6 lands.

## What is here so far

- **`src/dc/schema.py`** — the row model and a strict loader. Six invariants, each guarding a
  specific silent failure; the most important is that `label == 1` **iff** `category == "distress"`,
  so a mislabelled row is a contradiction the loader catches rather than a plausible mistake.
- **`src/dc/splits.py`** — the frozen train/test assignment. Two rules: golden rows are always
  test, and paraphrase groups never straddle the line. Groups are ordered by a salted hash rather
  than shuffled, so adding rows in milestone 2 will not reshuffle what is already assigned.
- **`data/seed.jsonl`** — the 44 golden cases imported from the companion repo. These are the
  release gate already running in its CI; training on them would keep that gate reporting 100%
  while measuring nothing, so `dc.splits` forces every one into the test set.

## Running

```bash
uv sync
make seed      # import the golden cases from ../bloom-langgraph
make splits    # (re)build data/splits.json
make check     # ruff + pyright strict + pytest
```

## Design

`BUILD_SPEC.md` is the plan of record: data provenance and ethics (§4), baselines (§5), why the
model is deliberately linear (§6), what explainability can and cannot mean over embedding
features (§7), the evaluation protocol (§8), and the cost model behind the operating threshold
(§9).
