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

Milestone 5 of 8 — a servable model and a fitted cascade. The test set has not been touched yet.

| # | Milestone | State |
| --- | --- | --- |
| 1 | Scaffold, strict schema, splits + leakage guards | ✅ |
| 2 | Dataset to 600+ reviewed rows | ✅ 688 rows, 191 positive (28%) |
| 3 | Baselines 0–3 + the Haiku teacher | ✅ |
| 4 | Training, CV, calibration, ablations | ✅ |
| 5 | Cost model, threshold fitting, cascade | ✅ |
| 6 | Single test-set evaluation, bootstrap CIs | — |
| 7 | Explanations + model card | — |
| 8 | Integration PR into `bloom-langgraph` | — |

### Baselines — 5-fold CV on train (497 rows, 28% positive)

| Baseline | Recall | Precision | PR-AUC | Missed |
| --- | --- | --- | --- | --- |
| majority | 0.00 | 0.00 | 0.28 | 139 |
| crisis-keyword regex | 0.13 | 0.64 | 0.33 | 121 |
| TF-IDF + LR | 0.91 | 0.88 | 0.94 | 13 |
| **MiniLM + LR** | **0.93** | 0.92 | **0.96** | **10** |
| `claude-haiku-4-5` (production) | 0.87 | 0.99 | 0.90 | 18 |

Three things worth noting.

**The keyword baseline is nearly useless — recall 0.13.** It was written in good
faith, not as a strawman, and it still misses 121 of 139 distress cases: real phrasing
varies far more than any hand-written list anticipates. That is the number that
justifies building a model at all.

**The frozen embedding space does separate the frames.** MiniLM + LR reaches 0.96
PR-AUC with a 3% false-positive rate on `game-frustration` — the hard class where
the wording is shared. So a linear probe is enough, and fine-tuning stays in
reserve rather than being assumed.

**The candidate is ahead of the model it distils** (0.96 vs 0.90 PR-AUC, 10 missed
vs 18) — which is possible only because a human corrected the teacher before the
student saw a label.

`reports/baselines.md` carries the per-category breakdown, bootstrap intervals, and
the caveats these numbers need — chiefly that 86% of the training rows came from a
generator, so the *ordering* is more trustworthy than the magnitudes. **The test
split is untouched**; it is spent once, at milestone 6.

### Training — `src/dc/train.py`

The spine reads top to bottom as eight steps: load, split, embed, baseline, select,
threshold, fit, save. Each calls a module and says *why*. After step 2 no test row is
in scope, and a test fails if one is ever embedded.

**The chip a player picks is a label leak in this dataset, and the model would learn
it.** Four of the seven feelings carry no distress at all. That comes from how the rows
were authored, not from anything true about players. Given the chip, the best config
gains +0.015 PR-AUC in cross-validation. CV rewards the shortcut because the shortcut
appears in every fold, so CV can't be what rejects it. A counterfactual probe can:
keep the text of each held-out distress case, change only the chip, and score it again.

| Chip swapped to | Recall before | Recall after |
| --- | --- | --- |
| `frustrated` | 0.96 | **0.05** |
| `proud` | 0.96 | 0.26 |
| `relieved` | 0.96 | 0.26 |
| `disappointed` | 0.96 | 0.32 |

A player who picks `frustrated` and then writes a real crisis note would be caught 5% of
the time. So feeling features are ineligible under the selection rule, which was fixed in
`dc.selection` before any result existed.

**Selection.** All ten text-only configs land within one standard error of the best
(0.950–0.962 PR-AUC). The differences between them are fold noise, not evidence. The rule
takes the lowest Brier score inside that band, **C=10, balanced**, which leaves out
C=0.01: it ranks almost as well, but its Brier score is six times worse. Calibration out
of fold: Brier 0.038, ECE 0.054. 72% of rows sit in the two outermost score bins, where
the model is close to calibrated. The thin middle bins are where it isn't.

**The artifact.** `artifacts/model.npz` + `model.json`: 385 weights, 3.6KB, no pickle.
Serving needs numpy only, and it matches scikit-learn to about 1e-7. Its operating thresholds come from the cost model below.

### The cascade — `reports/cascade.md`

The model does not replace the LLM. It sits in front of it and picks one of three routes
for each note:

| Score | Route |
| --- | --- |
| p < 0.0181 | **skip the LLM** — straight to encouragement |
| between | **escalate** — `claude-haiku-4-5` decides, exactly as it does today |
| p > 0.163 | **support** — straight to the reviewed support message |

A score exactly on a threshold escalates, and so does any score that isn't a finite
probability. Every way the local model can fail lands on today's behaviour.

**`low` is a safety constraint, not an optimisation.** Skipping the LLM is the only route
that can add a missed crisis, so `low` sits at half the lowest score any validation
distress case received. Zero misses on 139 cases is still a finite sample, so the report
says it the honest way: the true share of distress cases that would skip the LLM is below
2.1% with 95% confidence. The case that set `low` is an
indirect disclosure of abuse at home, scored at 0.036: the kind of note a frozen embedding
under-reads, and the kind the constraint exists to protect.

**`high` is chosen by expected cost, with a missed crisis treated as 20× an unneeded
support message.** On validation, expected values:

| Policy | Expected recall | Expected false alarms | LLM calls |
| --- | --- | --- | --- |
| LLM alone (today) | 0.863 | 1.3 | 100% |
| **cascade** | 1.000 | 61.0 | 34% |

**Those numbers need two warnings.** First, recall 1.000 describes how the thresholds were
built, not how they will perform: both were fitted on these same rows. Second, 20:1 sits
just past a cliff. Above 13.2:1 the cost model sends 40% of all notes
to support. Below it, `high` would be 0.510, with 4 more expected missed crises and 53
fewer false alarms. That whole trade rests on **4 labelled-distress notes where the teacher
voted "not distress" three times out of three**. One of them is work venting, which the
production prompt explicitly classes as not distress. The report lists all four. Whether
they are distress, and so whether 20:1 is the right ratio, is a product decision. It was
not made here.

**Latency** (`make bench`, a snapshot): the local path — embed, score, route — is 4.8 ms at
p50 on a laptop CPU. The LLM path isn't timed by default because it costs money; the report
explains how to run it.

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

make vendor-model           # one ~90MB download of the pinned MiniLM weights
make baselines              # baselines 0-4, CV on train  (ARGS="--skip-embedding")
make train                  # select, fit thresholds, write artifacts/ and reports/
make bench                  # latency snapshot  (ARGS="--llm-calls 20 --yes" costs money)
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
