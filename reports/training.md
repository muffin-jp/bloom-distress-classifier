# Training

497 training rows, 139 positive (28%). 5-fold StratifiedGroupKFold grouped by `origin_id`, seed 0. Every config is scored on the same folds.

**The test split is untouched.** It is evaluated once, at the end.

## The bar

A crisis-keyword rule needs no training and scores PR-AUC **0.327** on these rows. The chosen model scores **0.960** mean fold PR-AUC, so a model was worth building.

## Why the chip a player picked is not a feature

In this dataset the feeling is close to a label leak — an artefact of how
rows were authored, not a fact about players:

| Feeling | Rows | Distress | P(distress) |
| --- | --- | --- | --- |
| `anxious` | 28 | 21 | 75% |
| `custom` | 161 | 97 | 60% |
| `disappointed` | 42 | 0 | 0% |
| `frustrated` | 105 | 0 | 0% |
| `proud` | 49 | 0 | 0% |
| `relieved` | 42 | 0 | 0% |
| `tired` | 70 | 21 | 30% |

Given the chip, the best config reaches **0.977** PR-AUC against **0.962** without it (+0.015). Cross-validation rewards the shortcut, because the shortcut is present in every fold — which is exactly
why cross-validation cannot be the thing that rejects it.

**The probe.** Take the held-out distress cases, leave the text untouched, and change only the chip. Scored by `C=10 cw=balanced+feeling` at threshold 0.5:

| Chip swapped to | Recall before | Recall after | Mean score before → after |
| --- | --- | --- | --- |
| `proud` | 0.96 | 0.26 | 0.91 → 0.34 |
| `relieved` | 0.96 | 0.26 | 0.91 → 0.35 |
| `frustrated` | 0.96 | 0.05 | 0.91 → 0.16 |
| `disappointed` | 0.96 | 0.32 | 0.91 → 0.39 |

A player who picks `frustrated` and then writes a real crisis note is caught 5% of the time instead of 96%. That is the shortcut, measured. The selection rule makes every feeling config ineligible.

## Selection

Rule, fixed before any result existed: Feeling features ineligible (label leak, see probe); best mean fold PR-AUC; keep configs within 1 SE; lowest Brier among them; ties to smaller C, then no class reweighting.

| Config | PR-AUC (mean ± SE) | Brier | |
| --- | --- | --- | --- |
| `C=1 cw=balanced` | 0.962 ± 0.020 | 0.064 | within 1 SE |
| `C=1 cw=none` | 0.961 ± 0.020 | 0.064 | within 1 SE |
| `C=10 cw=balanced` | 0.960 ± 0.020 | 0.038 | **chosen** |
| `C=10 cw=none` | 0.960 ± 0.020 | 0.040 | within 1 SE |
| `C=0.1 cw=balanced` | 0.956 ± 0.019 | 0.157 | within 1 SE |
| `C=0.1 cw=none` | 0.955 ± 0.020 | 0.144 | within 1 SE |
| `C=0.01 cw=balanced` | 0.953 ± 0.020 | 0.235 | within 1 SE |
| `C=0.01 cw=none` | 0.953 ± 0.020 | 0.194 | within 1 SE |
| `C=100 cw=balanced` | 0.951 ± 0.023 | 0.040 | within 1 SE |
| `C=100 cw=none` | 0.950 ± 0.023 | 0.041 | within 1 SE |
| `C=10 cw=balanced+feeling` | 0.977 ± 0.013 | 0.029 | ineligible (feeling) |
| `C=100 cw=balanced+feeling` | 0.976 ± 0.012 | 0.028 | ineligible (feeling) |
| `C=10 cw=none+feeling` | 0.975 ± 0.013 | 0.028 | ineligible (feeling) |
| `C=100 cw=none+feeling` | 0.974 ± 0.014 | 0.027 | ineligible (feeling) |
| `C=1 cw=balanced+feeling` | 0.970 ± 0.013 | 0.051 | ineligible (feeling) |
| `C=1 cw=none+feeling` | 0.969 ± 0.013 | 0.048 | ineligible (feeling) |
| `C=0.1 cw=balanced+feeling` | 0.882 ± 0.061 | 0.116 | ineligible (feeling) |
| `C=0.1 cw=none+feeling` | 0.880 ± 0.061 | 0.111 | ineligible (feeling) |
| `C=0.01 cw=none+feeling` | 0.864 ± 0.063 | 0.176 | ineligible (feeling) |
| `C=0.01 cw=balanced+feeling` | 0.864 ± 0.063 | 0.204 | ineligible (feeling) |

Best eligible mean was `C=1 cw=balanced` at 0.962; 10 config(s) fell within one standard error of it, and `C=10 cw=balanced` had the lowest Brier score among them.

## Calibration of the chosen model (out-of-fold)

Brier **0.038** · expected calibration error **0.054**

| Score bin | Rows | Mean predicted | Observed distress |
| --- | --- | --- | --- |
| 0.0–0.1 | 269 | 0.03 | 0.01 |
| 0.1–0.2 | 42 | 0.14 | 0.07 |
| 0.2–0.3 | 27 | 0.25 | 0.11 |
| 0.3–0.4 | 16 | 0.35 | 0.00 |
| 0.4–0.5 | 5 | 0.44 | 0.20 |
| 0.5–0.6 | 7 | 0.55 | 0.71 |
| 0.6–0.7 | 10 | 0.64 | 0.80 |
| 0.7–0.8 | 12 | 0.74 | 0.83 |
| 0.8–0.9 | 21 | 0.85 | 0.95 |
| 0.9–1.0 | 88 | 0.96 | 0.99 |

72% of rows sit in the two outermost bins, where the model is close to calibrated. The middle bins — exactly where an "uncertain, ask the LLM" band would sit — hold few rows and carry the largest gaps (worst: 0.3–0.4, predicted 0.35, observed 0.00, 16 rows).

Two consequences. Thresholds are fitted empirically on these out-of-fold scores, so they do not depend on the scores being probabilities. But these scores should not be *described* as probabilities anywhere a person reads them without recalibrating first.

## Artifact

`artifacts/model.npz` + `artifacts/model.json`, with operating thresholds `low` 0.0181 and `high` 0.163 fitted from the cost model. How they were chosen, and what the cascade would have done on validation, is in `reports/cascade.md`.
