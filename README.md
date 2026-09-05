# bloom-safety-classifier

Distilling an LLM safety classifier into a small, explainable, threshold-tuned model —
dataset, evaluation, and model card.

Companion to [`bloom-langgraph`](../bloom-langgraph), whose *Guided Encouragement* feature routes
every free-text note through a `claude-haiku-4-5` call before generating anything: distress goes
to fixed, human-reviewed words; everything else goes to the encouragement branch. That decision
is unmeasured beyond routing accuracy on 44 cases, costs a model call on the hot path, and cannot
explain itself. This repo addresses all three.

**Not a diagnostic instrument.** It chooses between two pre-approved response *paths* for an
in-game message and makes no claim about anyone's mental state.

## Status

Milestone 2 of 8 — dataset tooling and the curated taxonomy. No model yet.

| # | Milestone | State |
| --- | --- | --- |
| 1 | Scaffold, strict schema, splits + leakage guards | ✅ |
| 2 | Dataset to 600+ reviewed rows | 🚧 tooling done, 92 candidates awaiting review |
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
- **`src/dc/candidates.py`** — a candidate is a row nobody has vouched for yet. It carries a
  *proposed* label and cannot be loaded as training data; `promote()` is the only way across, and
  it requires a reviewer's name.
- **`src/dc/teacher.py`** — the classifier being distilled, with its prompt copied verbatim from
  production and a drift guard that fails the tests if the two diverge.
- **`data/candidates/curated.jsonl`** — 92 hand-written candidates, described below.

### The curated taxonomy

The hard part of this problem is that a player writing *"I want to die, this stage took me 40
tries"* and a player writing *"I want to die. I can't do this anymore"* produce nearly the same
string and need opposite responses.

So the hand-written pack is not a pile of examples — it is a **taxonomy of confusion modes**, and
18 of them are written as *matched pairs*: identical surface phrasing, one in a game frame and one
in a life frame. Death idiom, giving up, hopelessness, worthlessness, exhaustion, pointlessness,
numbness, isolation, sleeplessness, finality, and more. Every trivial baseline fails on these, and
they are where the error analysis will live.

`tests/test_curated_pack.py` protects that argument: if the pairs stop being matched, or the hard
class stops outweighing the positive class, the tests fail rather than the numbers quietly getting
easier.

### How a row becomes training data

```
curated.jsonl ─┐
               ├─→ propose_labels.py ──→ review.py ──→ labelled.jsonl
generated.jsonl┘   (teacher orders       (a human       (training data)
                    the queue)            decides)
```

The teacher never assigns a label. It classifies each candidate several times and records where it
**contradicts the author** and where it **contradicts itself across runs** — those rows go to the
top of the review queue, because they are the genuinely ambiguous ones. Reviewing a proposal is
fast; assigning a label from scratch is not. That is the only reason the teacher is in the loop.

Because a human corrects the teacher before the student ever sees a label, the labels end up
*better* than the teacher's — which is what makes it possible for the student to beat it.

## Running

Only the two API scripts need a credential; everything else is offline. Either export
`ANTHROPIC_API_KEY`, or `cp .env.example .env` and fill it in, or run `ant auth login` once and
leave both unset — the SDK resolves all three.

```bash
uv sync
make seed                   # import the golden cases from ../bloom-langgraph
make check                  # ruff + pyright strict + pytest
make stats                  # review queue summary

# These two call the API and cost money. The bare targets only estimate;
# the -run targets actually spend. Extra flags go through ARGS.
make generate               # cost estimate for candidate expansion
make generate-run           # ...actually spend  (ARGS="--per-seed 8")
make propose                # cost estimate for the teacher pass
make propose-run            # ...actually spend  (ARGS="--votes 5")

make review REVIEWER=uv     # the only path from candidate to training data
make splits                 # (re)build data/splits.json once labelled.jsonl exists
```

## Design

Four commitments the rest of this repo is built around.

**The model is deliberately linear** — logistic regression over frozen `all-MiniLM-L6-v2`
embeddings. 800 examples cannot train 22.7M parameters without memorising them, and a
385-parameter artifact is auditable, needs no GPU, and adds sub-millisecond inference rather
than a second model to serve.

**Explainability is honest about its limits.** A coefficient over embedding dimension 197 means
nothing to anyone. Decisions are explained by nearest labelled neighbours in the same embedding
space, by a parallel TF-IDF model whose coefficients *are* words, and by a per-decision record —
not by pointing at the deployed model's weights.

**Every baseline must be beaten in order**: majority class, then a hand-built crisis-keyword
regex, then TF-IDF, then embeddings. If the keyword list wins, the honest conclusion is that no
model was needed.

**The operating threshold comes from a stated cost model, not from 0.5.** A missed distress case
is treated as ~20x costlier than a false positive — a false positive receives the reviewed
support message, which is warm and appropriate on its own. That ratio is a product decision,
written down so it can be argued with and changed without retraining.
