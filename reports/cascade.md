# Cascade

Thresholds fitted on the chosen model's out-of-fold scores: 497 training rows, 139 distress. **The test split is untouched.**

## Three routes

| Score | Route | What happens |
| --- | --- | --- |
| p < 0.0181 | `skip-llm` | Encouragement branch. No LLM call. |
| 0.0181 ≤ p ≤ 0.163 | `escalate` | `claude-haiku-4-5` decides — exactly what production does today. |
| p > 0.163 | `support` | The reviewed support message. No LLM call. |

A score exactly on a threshold escalates, and so does any score that is not a finite probability. Every way the local model can fail lands on today's behaviour, never on a skipped call.

## The cost model

A missed crisis is treated as **20×** as costly as an unneeded support message. That is a product decision, not a statistical one, and it is written as a number so it can be argued with — the sensitivity table below shows what changes if it is set differently.

## `low` is a safety constraint, not an optimisation

Skipping the LLM is the only route that can add a missed crisis, so `low` is not tuned for cost. The lowest score any validation distress case received was **0.0363**; `low` keeps 50% of it, **0.0181**. No validation distress case falls in the skip band.

Zero misses among 139 is still a finite sample. The honest form of the claim: the true share of distress cases that would skip the LLM is below **2.1%** with 95% confidence.

The distress cases nearest the skip band — the rows that set `low`, and the first ones to re-read:

| Row | Score | Note |
| --- | --- | --- |
| `cur-ds-02-v05` | 0.0363 | i keep checking the door, never know which version of him walks in |
| `cur-pair-15-life` | 0.0871 | I'm done. I don't see a way out of this |
| `cur-pair-13-life-v06` | 0.1070 | my phone hasnt rung in weeks and i dont think it will |
| `cur-pair-12-life-v04` | 0.1628 | couldn't cry at the thing i should have cried at, just sat there |
| `cur-ds-07-v05` | 0.1789 | can't catch my breath, played this to calm down and it didn't work |

## `high` is chosen by expected cost

Every distinct split of the rows above `low` was scored by expected cost at 20:1, and **0.163** was cheapest. The cutoff sits midway between the two scores it separates, so it does not rest on a validation row. Ties went to the higher cutoff, which escalates more and so stays closer to production.

## What the cascade would have done

Expected values over the validation rows. The LLM is modelled as one call that says distress with probability equal to the share of its three recorded votes.

| Policy | Expected recall | Expected missed | Expected false alarms | LLM calls | Expected cost |
| --- | --- | --- | --- | --- | --- |
| LLM alone (today) | 0.863 | 19.0 | 1.3 | 100% | 381.3 |
| model alone | 0.978 | 3.0 | 61.0 | 0% | 121.0 |
| **cascade** | 1.000 | 0.0 | 61.0 | 34% | 61.0 |

Routes under the cascade: skip **26%** · escalate **34%** · support **40%**.

**Recall 1.000 describes how the thresholds were built, not how they will perform.** `low` was placed under the lowest-scoring distress case in these rows, and `high` was optimised on the same rows. Expect misses on the test set.

## If the ratio is different

`low` does not depend on the ratio. `high` does — and not smoothly. Across every possible ratio, the cost model can only ever choose one of these operating points:

| Ratios that choose it | `high` | Expected recall | Expected missed | Expected false alarms | LLM calls | Support |
| --- | --- | --- | --- | --- | --- | --- |
| below 0.1:1 | 1.000 | 0.863 | 19.0 | 0.3 | 74% | 0% |
| 0.1:1 – 0.4:1 | 0.871 | 0.930 | 9.7 | 1.0 | 55% | 19% |
| 0.4:1 – 1.0:1 | 0.799 | 0.947 | 7.3 | 2.0 | 52% | 22% |
| 1.0:1 – 2.1:1 | 0.714 | 0.954 | 6.3 | 3.0 | 50% | 24% |
| 2.1:1 – 13.2:1 | 0.510 | 0.971 | 4.0 | 8.0 | 46% | 28% |
| 13.2:1 and above ← 20:1 | 0.163 | 1.000 | 0.0 | 61.0 | 34% | 40% |

**20:1 sits above a switch at 13.2:1.** One step more conservative, `high` would be 0.510: 4.0 more expected missed crises, and 53 fewer false alarms. The switch sits exactly where one expected catch is worth 13.2 false alarms — so choosing 20:1 is choosing to send 53 more players the support message to catch those 4.0.

That trade rests on 4 distress case(s) where the teacher's votes disagreed with the reviewer. They are the rows to re-read before accepting 20:1 — and they are exactly the disagreements the caveat below calls partly definitional.

| Row | Score | Teacher votes for distress | Note |
| --- | --- | --- | --- |
| `cur-pair-12-life-v04` | 0.163 | 0% | couldn't cry at the thing i should have cried at, just sat there |
| `cur-ds-07-v05` | 0.179 | 0% | can't catch my breath, played this to calm down and it didn't work |
| `cur-pair-02-life-v01` | 0.265 | 0% | work is destroying me rn, i literally cannot do another day of this |
| `cur-pair-08-life-v04` | 0.287 | 0% | woke up with my face wet again, no idea what i dreamed |

## If the margin is different

| Margin | `low` | Notes that skip the LLM |
| --- | --- | --- |
| 1.00 | 0.0363 | 40% |
| 0.50 ← | 0.0181 | 26% |
| 0.25 | 0.0091 | 15% |

## How to read these numbers

**The same validation data was used twice.** These out-of-fold scores chose the config and then fitted both thresholds, so every number above is optimistic. The test set, spent once in the next milestone, is the check.

**The shipped model did not produce these scores.** Out-of-fold scores come from models trained on 80% of train; the shipped model saw all of it and tends to score more confidently. The thresholds are expected to transfer. Until the test set confirms it, that is an expectation.

**The LLM's recall here is partly definitional.** The reviewer saw the teacher's votes while labelling, so every overruled vote counts against it by construction. That understates the LLM alone, and flatters the cascade's lead over it.

**Latency is not in this report.** `make bench` measures it separately, because timings vary run to run and this file should not.
