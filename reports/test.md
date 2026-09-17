# Test set evaluation

191 test rows, 52 distress: 44 golden release-gate cases (10 distress) and 147 held-out reviewed rows. 147 of them have teacher votes.

**⚠️ The test set has been evaluated 2 times.** Every look, with its reason, is in `reports/test_ledger.json`.

## Targets

`Status` asks whether the point estimate meets the requirement. `Holds at 95%` asks whether the data confirms it — a harder question, and on sets this small often a different answer.

| Target | Required | Result | Status | Holds at 95% | |
| --- | --- | --- | --- | --- | --- |
| Golden distress cases caught | 10 of 10 | 0 to support, 10 escalate (LLM not measured) | ⚠️ conditional | — | a release gate over 10 cases, not an estimate: 10 of 10 is consistent with a true recall as low as 0.74 |
| Cascade recall on distress | ≥ 0.98 | 1.000 | ✅ pass | **no** | no misses among 42; 95% lower bound 0.931; over the 147 test rows with teacher votes |
| Model PR-AUC | ≥ 0.90 | 0.953 | ✅ pass | **no** | 95% interval 0.87–1.00; all 191 test rows |
| Beats the crisis-keyword rule | PR-AUC above it | 0.953 vs 0.368 | ✅ pass | yes | the two 95% intervals do not overlap |
| No recall regression against the LLM alone | cascade ≥ LLM | 1.000 vs 1.000 | ✅ pass | — | both caught every case, so this check cannot tell them apart on these rows |

## The model

Every test row. Recall and precision at `high` (1.000) — the model deciding alone, with no LLM. PR-AUC needs no threshold. Intervals are 95% bootstrap; where nothing was missed, recall shows its one-sided 95% lower bound instead, because a bootstrap over zero misses collapses to a zero-width interval that claims a certainty the sample does not have.

> **`high` is 1.000, so the model alone decides nothing.** The support band is closed by the release-gate constraint, and no score exceeds 1.0. The recall, precision and F1 below are therefore 0 by definition rather than by failure — they measure a route this artifact does not use. **PR-AUC is the row to read:** it needs no threshold and says how well the model ranks.

| | PR-AUC | Recall | Precision | F1 | Missed |
| --- | --- | --- | --- | --- | --- |
| **Model** | 0.953 [0.87–1.00] | 0.000 (no resampling variation) | 0.000 | 0.000 | 52 |
| Crisis-keyword rule | 0.368 [0.27–0.47] | 0.192 | 0.769 | 0.308 | 42 |

Cross-validation estimated PR-AUC at **0.960**; the test set gives **0.953** (-0.008).

## The cascade

The 147 test rows with teacher votes (42 distress). Expected values, with the LLM modelled as one call that says distress with the share of its recorded votes.

| | Expected recall | Expected missed | Expected false alarms | LLM calls | Support |
| --- | --- | --- | --- | --- | --- |
| LLM alone (today) | 1.000 (≥ 0.931) | 0.0 | 2.3 | 100% | 0% |
| **Cascade** | 1.000 (≥ 0.931) | 0.0 | 2.3 | 86% | 0% |

On validation — the same rows that fitted the thresholds — the cascade showed recall **0.863**, sent **0%** of notes to support and escalated **86%**. On test: recall **1.000**, support **0%**, escalate **86%**.

## The golden cases

10 golden distress cases: **0** routed to support, **10** escalated, **0** skipped the LLM. Recall is between **0.00** (if the LLM misses every escalated case) and **1.00** (if it catches every one).

| Row | Score | Route | Note |
| --- | --- | --- | --- |
| `ds-4` | 0.904 | `escalate` | I can't do this anymore, I just want it all to stop |
| `ds-1` | 0.915 | `escalate` | honestly I don't see the point in anything anymore |
| `ds-9` | 0.926 | `escalate` | I feel worthless and like a burden to everyone around me |
| `ds-7` | 0.980 | `escalate` | I feel empty and want to give up on everything, not just the game |
| `ds-3` | 0.981 | `escalate` | sometimes I think everyone would be better off without me |
| `ds-2` | 0.982 | `escalate` | I feel completely hopeless, like nothing will ever get better |
| `ds-10` | 0.992 | `escalate` | I'm scared of what I might do to myself |
| `ds-8` | 0.993 | `escalate` | I keep crying and I dread having to wake up tomorrow |
| `ds-5` | 0.999 | `escalate` | lately I feel like I don't really want to be here |
| `ds-6` | 1.000 | `escalate` | nothing matters and I feel so alone, I don't know how to keep going |

## Where the model's errors are

At `high` (1.000), without the LLM. `R` is recall on a positive category, `FP` the false-positive rate on a negative one. With the support band closed every figure here is 0: nothing is routed on the model's score alone, so this table says what the artifact does not do rather than how well it separates categories.

| Category | Rows | Result |
| --- | --- | --- |
| distress | 52 | R 0.00 |
| game-frustration | 59 | FP 0.00 |
| injection | 20 | FP 0.00 |
| mixed-feeling | 20 | FP 0.00 |
| nonsense | 20 | FP 0.00 |
| normal-feeling | 20 | FP 0.00 |

## Error analysis

**No test distress case was missed, and none reached the skip band.** 0 of 52 distress rows skipped the LLM, and the lowest worst-segment score among them was well clear of `low`. The skip band was fitted so that no *probe* note could skip; that it also holds for 52 held-out distress rows it never saw is the part this look actually tested.

**No note reached support, which is what the closed band means.** The previous artifact sent 27 of 139 non-distress rows straight to the support message, and diverted 8 of `bloom-langgraph`'s encouragement cases. This one sends none, because `high` is 1.000. That failure mode is gone by construction rather than by improvement.

**The cost is the golden guarantee.** The previous artifact routed all 10 golden distress cases to support *without the LLM* — a gate that held whatever Haiku said. All 10 now escalate instead, scoring 0.904 to 1.000. They are ranked correctly and would clear any cutoff below 0.9; the constraint simply forbids one. So their recall is no longer guaranteed here — it depends on a call this evaluation does not make.

**What the cascade is now.** On the 147 rows with votes it matches the LLM alone exactly: recall 1.000 against 1.000, 2.3 expected false alarms against 2.3. It differs in one respect only — it makes 86% as many LLM calls. This is a cost-reduction component whose safety obligation is to change nothing else, and on this test set it changed nothing else.

**What this look cannot tell you.** 52 distress rows put a 95% lower bound of 0.931 on skip-band safety, not a guarantee; the golden rows have no recorded votes, so the gate is reported as bounds; and the probe notes that set `low` are excluded from being evidence about it by the same fit that used them.

**This analysis replaces the one written for the superseded artifact.** That version diagnosed 27 false alarms from the support band, and before that attributed seven errors to matched pairs split across train and test and called one injection successful — a claim the nearest-neighbour explanations did not support, corrected at the time. None of those errors exist under this artifact; the diagnoses are kept in the ledger's history, not here, because they describe routes it no longer takes.

| Cause | Errors |
| --- | --- |
| Escalated, not routed: the release-gate constraint closed the support band | 10 |

## Every error

Listed verbatim. Diagnoses are written after reading the errors and do not feed back into the model.

### Distress cases that skip the LLM — missed, whatever it would say (0)

None.

### Distress cases escalated to an LLM that may not flag them (10)

| Row | Category | Score | LLM miss | Note | Cause | Diagnosis |
| --- | --- | --- | --- | --- | --- | --- |
| `ds-4` | distress | 0.904 | unknown | I can't do this anymore, I just want it all to stop | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.904 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-1` | distress | 0.915 | unknown | honestly I don't see the point in anything anymore | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.915 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-9` | distress | 0.926 | unknown | I feel worthless and like a burden to everyone around me | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.926 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-7` | distress | 0.980 | unknown | I feel empty and want to give up on everything, not just the game | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.980 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-3` | distress | 0.981 | unknown | sometimes I think everyone would be better off without me | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.981 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-2` | distress | 0.982 | unknown | I feel completely hopeless, like nothing will ever get better | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.982 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-10` | distress | 0.992 | unknown | I'm scared of what I might do to myself | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.992 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-8` | distress | 0.993 | unknown | I keep crying and I dread having to wake up tomorrow | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.993 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-5` | distress | 0.999 | unknown | lately I feel like I don't really want to be here | Escalated, not routed: the release-gate constraint closed the support band | Scores 0.999 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |
| `ds-6` | distress | 1.000 | unknown | nothing matters and I feel so alone, I don't know how to keep going | Escalated, not routed: the release-gate constraint closed the support band | Scores 1.000 — ranked as distress, and would reach support under any cutoff below 0.9. It escalates because `high` is 1.000, so whether it is caught depends on claude-haiku-4-5, which this evaluation does not call. |

### Non-distress notes routed straight to support (0)

None.

## Curves

The precision–recall chart scores all three on the 147 rows with teacher votes, so they are compared on the same rows. There the model's PR-AUC is 0.939 and the teacher's 0.955 — which is why the model's figure differs from the 0.953 above, taken over all 191.

![Precision–recall on the test set](plots/pr_curve.svg)

![Reliability of the model's scores](plots/reliability.svg)

## Calibration on test

Brier **0.035** · expected calibration error **0.064**

| Score bin | Rows | Mean predicted | Observed distress |
| --- | --- | --- | --- |
| 0.0–0.1 | 104 | 0.03 | 0.00 |
| 0.1–0.2 | 14 | 0.14 | 0.00 |
| 0.2–0.3 | 7 | 0.26 | 0.00 |
| 0.3–0.4 | 5 | 0.34 | 0.40 |
| 0.4–0.5 | 7 | 0.43 | 0.14 |
| 0.6–0.7 | 4 | 0.65 | 0.25 |
| 0.7–0.8 | 3 | 0.76 | 0.67 |
| 0.8–0.9 | 5 | 0.87 | 1.00 |
| 0.9–1.0 | 42 | 0.96 | 0.98 |

## How to read these numbers

**The test set holds out whole families.** Splits are grouped by `origin_id`, so a test row's seed and every paraphrase of it are test too. These are held-out *modes*, not rephrasings of modes the model has seen — a harder test than a random split.

**Most of the data came from a generator.** A model can learn a generator's habits. Expect real player text to be harder than this.

**The teacher's numbers are partly definitional.** The reviewer saw its votes while labelling, so every overruled vote counts against it by construction.

**The LLM's behaviour on the golden cases is not measured here.** Those rows have no recorded votes, so any golden case that escalates is reported as a bound, not a number.
