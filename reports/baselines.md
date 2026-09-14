# Baselines (5-fold CV on train)

497 training rows, 139 positive (28%). 5-fold StratifiedGroupKFold grouped by `origin_id`, seed 0.

**The test split is untouched.** It is evaluated once, at the end.
Metrics are at threshold 0.5; the operating point is chosen later, on a
cost model. Compare on PR-AUC, which needs no threshold. Intervals are
95% percentile bootstrap over 1000 resamples.

| Baseline | Recall | Precision | F1 | PR-AUC | Missed | n |
| --- | --- | --- | --- | --- | --- | --- |
| majority | 0.00 [0.00-0.00] | 0.00 | 0.00 | 0.28 [0.23-0.32] | 139 | 497 |
| keyword | 0.13 [0.08-0.19] | 0.64 | 0.22 | 0.33 [0.27-0.39] | 121 | 497 |
| tfidf+lr | 0.91 [0.85-0.95] | 0.88 | 0.89 | 0.94 [0.90-0.98] | 13 | 497 |
| minilm+lr | 0.93 [0.88-0.97] | 0.92 | 0.92 | 0.96 [0.91-0.99] | 10 | 497 |
| teacher(haiku) | 0.87 [0.81-0.92] | 0.99 | 0.93 | 0.90 [0.86-0.95] | 18 | 497 |

## Recall and false positives by category

| Baseline | distress | game-frustration | injection | mixed-feeling | nonsense | normal-feeling |
| --- | --- | --- | --- | --- | --- | --- |
| majority | R 0.00 | FP 0.00 | FP 0.00 | FP 0.00 | FP 0.00 | FP 0.00 |
| keyword | R 0.13 | FP 0.06 | FP 0.00 | FP 0.00 | FP 0.00 | FP 0.00 |
| tfidf+lr | R 0.91 | FP 0.05 | FP 0.10 | FP 0.05 | FP 0.05 | FP 0.02 |
| minilm+lr | R 0.93 | FP 0.03 | FP 0.10 | FP 0.00 | FP 0.00 | FP 0.02 |
| teacher(haiku) | R 0.87 | FP 0.01 | FP 0.00 | FP 0.00 | FP 0.00 | FP 0.00 |

`R` is recall on a positive category, `FP` the false-positive rate on a
negative one. `game-frustration` is the column that matters: it is where
a lexical rule collapses, because the words are the same.

## How to read these numbers

Three caveats, in descending order of how much they should temper the
table above.

**Most of this data came from a generator.** 552 of 644 reviewed rows are
LLM variants of 92 hand-written seeds. Grouped CV stops a paraphrase from
scoring its own sibling, but train and validation still share a generator,
and a model can learn that generator's habits rather than the distinction
being taught. Trust the *ordering* of these baselines more than their
magnitudes; expect the absolute numbers to fall on real player text.

**The teacher's recall is partly definitional.** The labels are the
reviewer's, and the reviewer saw the teacher's votes while deciding. Every
row where the reviewer overruled it counts against it by construction. Read
its row as *disagreement with the reviewed labels*, not as an error rate.

**Threshold 0.5 is a placeholder, not a choice.** The operating point comes
from a stated cost model on validation data, later. Nothing here is tuned,
which is why PR-AUC is the column to compare on.
