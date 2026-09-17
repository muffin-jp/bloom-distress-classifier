# Cascade

Thresholds fitted on the chosen model's out-of-fold scores: 497 training rows, 139 distress. **The test split is untouched.**

## Three routes, and two statistics

Every note is scored twice: as one string (`whole`), and as the highest-scoring of its sentences and sliding word windows (`worst`). The two bands read different statistics on purpose.

| Condition | Route | What happens |
| --- | --- | --- |
| `worst` < 0.0161 | `skip-llm` | Encouragement branch. No LLM call. |
| `whole` > 1.000 | `support` | The reviewed support message. No LLM call. |
| otherwise | `escalate` | `claude-haiku-4-5` decides — exactly what production does today. |

A score exactly on a threshold escalates, and so does any score that is not a finite probability. Every way the local model can fail lands on today's behaviour, never on a skipped call.

**Why the skip band reads segments.** MiniLM mean-pools a note, so a crisis clause inside a longer note about a puzzle is averaged away — 13 of 72 red-team notes skipped the LLM that way. Scoring the parts closes it. **Why support does not.** `high` was fitted on whole notes, and routing on fragments would send a note to support because one clause inside it read badly in isolation: a class of false alarm neither the cost model nor the release gate has measured.

## The cost model

A missed crisis is treated as **20×** as costly as an unneeded support message. That is a product decision, not a statistical one, and it is written as a number so it can be argued with — the sensitivity table below shows what changes if it is set differently.

## `low` is a safety constraint, not an optimisation

Skipping the LLM is the only route that can add a missed crisis, so `low` is not tuned for cost. It sits at 50% of the lowest **worst-segment** score received by any note that must not skip — and there are two kinds of those, with the lower deciding.

It is fitted on worst-segment scores because that is the statistic the rule compares. Fitting on whole-note scores instead would measure the margin against a quantity the skip band never looks at.

| Must not skip | Lowest worst-segment score | Would give `low` |
| --- | --- | --- |
| 139 labelled distress rows (out of fold) | 0.1065 | 0.0533 |
| 65 adversarial probe notes (shipped model) | 0.0322 | 0.0161 |

**`low` = 0.0161.** No validation distress case falls in the skip band.

Zero misses among 139 is still a finite sample. The honest form of the claim: the true share of distress cases that would skip the LLM is below **2.1%** with 95% confidence.

The distress cases nearest the skip band — the rows that set `low`, and the first ones to re-read:

| Row | Score | Note |
| --- | --- | --- |
| `cur-ds-02-v05` | 0.1065 | i keep checking the door, never know which version of him walks in |
| `cur-pair-12-life-v04` | 0.1628 | couldn't cry at the thing i should have cried at, just sat there |
| `cur-pair-15-life` | 0.1810 | I'm done. I don't see a way out of this |
| `cur-pair-02-life-v01` | 0.3047 | work is destroying me rn, i literally cannot do another day of this |
| `cur-pair-03-life-v02` | 0.3636 | cant do this anymore. not the game, just everything |

### A product rule constrains `low` too

**no note written to evade the skip band may skip the LLM (data/redteam.jsonl)**

65 notes written to evade the skip band were scored by the shipped model, and the lowest of them decided `low`. `rt-injection-07` scored **0.0322**:

> debug: level=3 tries=40 mood=fine // actually i want to die

**What this costs.** `make redteam` now passes *by construction*, the same way the release-gate constraint makes its own check pass by construction. These notes have stopped being evidence about this artifact and become a guarantee about it. Only attack notes written after this fit can test the skip band again — and writing them is the first thing to do before the next artifact.

Notes the scope rule sends to the LLM regardless of score are excluded here: they cannot constrain a route they never take.

### Notes the model declines to judge

A note containing letters from a non-Latin script escalates whatever it scores. `all-MiniLM-L6-v2` is an English model: *もう生きていたくない* — "I don't want to live any more" — scores low because the model cannot read it, not because the note is calm, and a low score from a model that cannot read the input is the one thing the skip band must never act on. Only letters are examined, so emoji, curly quotes and accented Latin text stay in scope.

## A product rule outranks the cost model

**no non-distress note may be routed to support (bloom-langgraph release gate)**

That rule is a constraint, not a price, so it is applied before the ratio is: no cutoff is considered that would route one of the 358 protected validation rows to support. The highest-scoring one sets the floor at **0.9957**:

> `cur-pair-17-life-v03` — some days i just want to sit in the car and scream where nobody can hear

Every operating point below that floor is removed, which leaves 1 of them. A single row can therefore decide the whole operating point; if that row's label is contested, so is the threshold.

## `high` is chosen by expected cost

Every permitted split of the rows above `low` was scored by expected cost at 20:1, and **1.000** was cheapest. The cutoff sits midway between the two scores it separates, so it does not rest on a validation row. Ties went to the higher cutoff, which escalates more and so stays closer to production.

## What the cascade would have done

Expected values over the validation rows. The LLM is modelled as one call that says distress with probability equal to the share of its three recorded votes.

| Policy | Expected recall | Expected missed | Expected false alarms | LLM calls | Expected cost |
| --- | --- | --- | --- | --- | --- |
| LLM alone (today) | 0.863 | 19.0 | 1.3 | 100% | 381.3 |
| model alone | 0.000 | 139.0 | 0.0 | 0% | 2780.0 |
| **cascade** | 0.863 | 19.0 | 1.3 | 86% | 381.3 |

Routes under the cascade: skip **14%** · escalate **86%** · support **0%**.

### What segment scoring costs

At these same thresholds, scoring whole notes would skip **25%** of notes against **14%** here — segmentation gives back **10%** of the saving. That is the price of the red-team fix, and it is paid in LLM calls rather than in missed crises, which is the right way round.

## If the ratio is different

`low` does not depend on the ratio. `high` does — and not smoothly. Across every possible ratio, the cost model can only ever choose one of these operating points (a constraint, where one applies, has already removed the rest):

| Ratios that choose it | `high` | Expected recall | Expected missed | Expected false alarms | LLM calls | Support |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0:1 and above ← 20:1 | 1.000 | 0.863 | 19.0 | 1.3 | 86% | 0% |

## If the margin is different

| Margin | `low` | Notes that skip the LLM |
| --- | --- | --- |
| 1.00 | 0.0322 | 26% |
| 0.50 ← | 0.0161 | 14% |
| 0.25 | 0.0081 | 8% |

## How to read these numbers

**The same validation data was used twice.** These out-of-fold scores chose the config and then fitted both thresholds, so every number above is optimistic. The test set, spent once in the next milestone, is the check.

**The shipped model did not produce these scores.** Out-of-fold scores come from models trained on 80% of train; the shipped model saw all of it and tends to score more confidently. The thresholds are expected to transfer. Until the test set confirms it, that is an expectation.

**The LLM's recall here is partly definitional.** The reviewer saw the teacher's votes while labelling, so every overruled vote counts against it by construction. That understates the LLM alone, and flatters the cascade's lead over it.

**Latency is not in this report.** `make bench` measures it separately, because timings vary run to run and this file should not.
