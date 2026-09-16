# bloom-safety-classifier

A 385-weight classifier that sits in front of an LLM safety check and decides, for each
free-text note a player writes, whether that check is needed at all.

It was built to replace most calls to the `claude-haiku-4-5` distress classifier in Bloom's
*Guided Encouragement* feature ([`bloom-langgraph`](../bloom-langgraph)), and it covers the
whole lifecycle: dataset, baselines, training, a cost-based cascade, a single test-set
evaluation, explanations, and a [model card](MODEL_CARD.md).

> **Not a diagnostic instrument.** It chooses between pre-approved responses to a note typed
> into a puzzle game. It makes no claim about anyone's mental state.

## Results

Scored **once** on a held-out test set of 191 notes, including the 44 cases that gate
releases in the companion repo. The look is recorded in `reports/test_ledger.json`.

| | Result | Holds at 95%? |
| --- | --- | --- |
| Model PR-AUC (target ≥ 0.90) | **0.953** — cross-validation said 0.960 | no — the interval reaches 0.87 |
| Crisis-keyword rule PR-AUC | 0.368 | the model beats it: yes |
| Distress cases missed | **0 of 52** | recall is at least 0.944 |
| Golden release-gate cases | 10 of 10 routed to support, without the LLM | a gate, not an estimate |
| LLM calls needed under the cascade | **32%** of notes | — |
| Non-distress notes sent to support | 27 of 139 (19%) | — |

**What that means.** The model generalises to confusion modes it never trained on and missed no
distress case. About two notes in three would need no LLM call: 26% skip it, and 41% go
straight to the support message. The cost is false alarms: about one in
five ordinary notes would get the support message. That rate follows from a deliberately low
support cutoff, which a product decision set; it is not a flaw in how the model ranks notes.

![Precision–recall on the test set](reports/plots/pr_curve.svg)

## How it works

```
note ─→ all-MiniLM-L6-v2 (frozen) ─→ 385 weights ─→ score
                                                     │
          score < 0.0181  ─→  encouragement           (no LLM call)
          in between      ─→  claude-haiku-4-5 decides (today's behaviour)
          score > 0.163   ─→  reviewed support message (no LLM call)
```

- **`low` is a safety constraint.** Skipping the LLM is the only route that can add a missed
  crisis, so `low` sits at half the lowest score any validation distress case received.
- **`high` is a cost decision.** It minimises expected cost with a missed crisis treated as 20
  times an unneeded support message. That ratio sits just past a cliff, and
  [`reports/cascade.md`](reports/cascade.md) shows where it is and the four labels the ratio
  depends on.
- **Failure lands on today's behaviour.** A score that is not a finite probability escalates
  to the LLM.

## Explaining a decision

The model's weights are 384 embedding dimensions no one can read, so a decision is explained by
what the note resembles and by the words its score moves with. Here is the same complaint
written about a stage, then about life:

```
$ make explain NOTE="30 tries on this stage and nothing works, i'm so tired of it"
Score    0.039 → escalate  (LLM consulted)
Nearest training notes — 0% distress
Words    toward distress  “nothing” +1.03, “i'm” +0.65, “tired” +0.62
         away from it     “stage” -0.69, “tries” -0.65, “this stage” -0.36

$ make explain NOTE="nothing works anymore and i'm so tired of everything lately"
Score    0.999 → support  (LLM not consulted)
Nearest training notes — 100% distress
Words    toward distress  “nothing” +1.18, “lately” +1.08, “i'm” +0.74
```

The words come from a surrogate fitted to the model's scores, which tracks the model at R²
0.68, so they are an approximation and are always labelled as one. Decision records store a
hash of the note, never the note.

## What building it found

**A keyword list is not enough.** A good-faith crisis-keyword rule caught 13% of distress cases
in cross-validation. Real phrasing varies far more than any list anticipates.

**A frozen embedding is enough.** A linear model over frozen MiniLM embeddings reached 0.96
PR-AUC, with a 3% false-positive rate on game frustration, so fine-tuning was never needed.

**The obvious extra feature was a trap.** In this dataset, the feeling chip a player picks
almost gives away the label. Cross-validation *rewarded* using it. A counterfactual probe showed
why that is unsafe: re-score real distress notes with only the chip changed to `frustrated`,
and recall falls from 0.96 to 0.05. Feeling features were ruled out before the model was chosen.

**The test set was looked at once, on purpose.** Scoring writes a ledger entry before any number
is shown; a second look is refused without a recorded reason. The report is rendered from saved
predictions, so fixing it never reads a test row again — and every re-render is logged.

**The errors point to data, not thresholds.** Of 27 false alarms, 22 resemble ordinary notes
and cleared the low support cutoff. The other five resemble distress: game notes in
first-person, ongoing-state language, like *"i keep starting over and honestly i just don't
have it in me tonight"* (0.994). The model has learned how a note is said more than what it
is about.

**One of my own conclusions did not survive.** The first error analysis blamed seven errors on
matched pairs split across train and test. The nearest-neighbour explanations did not support
that, and the analysis was corrected in the open.

**Integration exposed two things the evaluation could not.** The first was a mistake: the
code claimed the embedder revision matched production's, and it did not. The two snapshots were
then verified byte-identical, and the served classifier accepts only verified-equivalent
revisions. The second is a real conflict. With the classifier switched on,
`bloom-langgraph`'s own release gate fails: seven golden cases that expect encouragement reach
support, so game frustration stays out of support only 80% of the time against a 100% gate.
The cost model called those false alarms cheap, and the gate calls them failures. The
integration ships dark, behind a flag that stays off until that is settled.

## Before shipping

The [model card](MODEL_CARD.md) sets out what is still open. In short: settle what a false
alarm costs, with the 20:1 cost model and the companion gate side by side; red-team the skip
band, since on test 7 of 20 injection notes scored low enough to skip the LLM; collect game
notes in first-person, ongoing-state language and ordinary off-topic sentences; and add a second
reviewer, because every label so far comes from one person, alongside a dataset that is 80%
synthetic.

## Reports

| | |
| --- | --- |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, data, evaluation, thresholds, failure modes, ethics |
| [`reports/test.md`](reports/test.md) | The test evaluation, with every error listed and diagnosed |
| [`reports/explanations.md`](reports/explanations.md) | What the model responds to, and each test error explained |
| [`reports/cascade.md`](reports/cascade.md) | The cost model, both thresholds, and the cliff |
| [`reports/training.md`](reports/training.md) | Model selection, the feeling-chip probe, calibration |
| [`reports/baselines.md`](reports/baselines.md) | Baselines 0–4 under cross-validation |

## Running

```bash
uv sync
make check                  # ruff + pyright strict + pytest — all offline
make vendor-model           # one ~90MB download of the pinned MiniLM weights

make baselines              # baselines 0-4, cross-validated on train
make train                  # select, fit thresholds, write artifacts/ and reports/
make explain NOTE="..."     # explain one note
make explain-errors         # explain the test set's recorded errors
make evaluate-render        # re-render the test report from saved predictions
```

Building the dataset calls the API and costs money. Each command prints an estimate and sends
nothing until confirmed:

```bash
make generate                          # estimate: expand the hand-written taxonomy
make generate-run                      # ...and spend
make propose                           # estimate: record the teacher's votes, to order review
make propose-run                       # ...and spend
make review REVIEWER=<name>            # the only path from candidate to training data
```

`make evaluate` scores the test set, and it has already been run. It refuses a second look
unless you pass `ARGS='--again "<reason>"'`; the reason is recorded, and the model card must
disclose it.

## Layout

```
src/dc/
  schema.py  splits.py            strict rows; a split that never leaks a paraphrase
  candidates.py  teacher.py       unreviewed rows; the teacher prompt, with a drift guard
  baselines.py  features.py       baselines 0-4; the pinned embedder
  selection.py  calibration.py    grouped CV, a rule fixed in advance; the feeling probe
  cascade.py                      routes, the cost model, both thresholds
  model.py  artifact.py           385 weights as npz + json, no pickle
  train.py                        the eight-step walkthrough spine
  evaluate.py  ledger.py          the test set, spent once, and the record of it
  explain.py  plots.py            neighbours, a word surrogate, decision records; SVG charts
scripts/                          dataset building, baselines, latency
data/  artifacts/  reports/       committed; the audit surface
```

## Status

| # | Milestone | |
| --- | --- | --- |
| 1 | Scaffold, schema, leakage-proof split | ✅ |
| 2 | Dataset: 688 rows, reviewed | ✅ |
| 3 | Baselines | ✅ |
| 4 | Training, calibration, the feeling-chip probe | ✅ |
| 5 | Cost model, thresholds, cascade | ✅ |
| 6 | One test-set evaluation | ✅ |
| 7 | Explanations and model card | ✅ |
| 8 | Integration into `bloom-langgraph` | ✅ merged dark — its release gate fails with the flag on |
