# Model Card / Executive Summary — Credit Risk Screening Model

## Model Details
- **Model type:** HistGradientBoostingClassifier (scikit-learn), with sigmoid (Platt) probability calibration
- **Model version:** `histgb_v1_calibrated_sigmoid`
- **Training library versions:** scikit-learn 1.8.0, xgboost 3.4.1 (backup model only), joblib 1.5.3
- **Tuned hyperparameters:** `learning_rate=0.2, max_depth=3, max_iter=200, min_samples_leaf=10, max_leaf_nodes=63, l2_regularization=0`
- **Preprocessing:** packaged inside the same artifact — sentinel imputation (999) for three "months-since-event" fields, median imputation elsewhere, one-hot encoding for categoricals (`term`, `purpose`, `home_ownership`, `verification_status`), no scaling (unneeded for tree-based models)
- **Calibration:** `CalibratedClassifierCV(method='sigmoid', cv=3)`, cross-fit on training data
- **Training data:** 190,156 LendingClub loans (2007-2014 originations), stratified 80% of the labeled, resolved-outcome subset
- **Backup candidate:** tuned XGBoost, nearly equivalent performance, documented but not packaged for deployment
- **Interpretability benchmark:** tuned Logistic Regression (`solver='newton-cholesky'`), used for explainability cross-checking, not deployed

## Intended Use
A **screening and prioritization tool** for routing loan applications toward manual underwriting
attention based on estimated credit risk. **Not validated or intended as an automatic approve/decline
mechanism.** Scored applications should feed a human-in-the-loop review process.

**Out of scope:** any lender, loan product, applicant population, or time period other than
2007-2014 LendingClub-style unsecured consumer loans, without independent re-validation.

## Training Data Summary
- Source: LendingClub loan data, 2007-2014 originations
- Target: binary — "bad" (Charged Off, Default, Late 31-120 days, or the equivalent
  "does not meet credit policy" charged-off variant) vs. "good" (Fully Paid or its credit-policy
  variant); loans with unresolved outcomes (`Current`, `In Grace Period`, `Late 16-30 days`) were
  excluded from the labeled set entirely (see project notebook, Section 2)
- Class balance: 78.56% good / 21.44% bad
- ~49% of the raw dataset was excluded due to unresolved outcomes — the model is trained on an
  older-vintage-skewed population, not the most recent originations

## Performance (Final, Calibrated Model, Untouched Test Set — 47,539 loans)

| Metric | Value | What it measures |
|---|---|---|
| **ROC-AUC** | **0.7118** | Discrimination / ranking ability — moderate, not strong, by conventional credit-scoring benchmarks |
| **Brier Score** | **0.1519** | Calibration — beats the base-rate-only reference (0.1684); confirms probabilities are genuinely usable in absolute terms |
| **F1 (at threshold 0.20)** | 0.4467 | Harmonic mean of precision/recall at the production threshold |
| **Precision** | 0.3296 | ~33% of flagged applications are actually bad — expect substantial review volume of ultimately-good loans |
| **Recall** | 0.6927 | ~69% of actual bad loans are caught; ~31% are still missed |
| **Accuracy** | — | Not the primary metric for this imbalanced problem; reported alongside F1/precision/recall in the notebook for completeness |

**Confusion matrix (test set, threshold 0.20):** TN=22,983, FP=14,362, FN=3,133, TP=7,061

## Calibration
The uncalibrated model's raw probabilities were confirmed **unreliable in absolute terms** —
Brier score 0.2157, *worse* than a trivial constant-prediction baseline (0.1684), a direct
consequence of training with `class_weight='balanced'`. Sigmoid calibration (chosen over isotonic
regression, which performed statistically identically but with more overfitting risk on limited
calibration data) corrected this to Brier 0.1519. **ROC-AUC was preserved throughout calibration**
(0.7104 → 0.7118, within noise) — calibration is a probability-scale correction, not a change
in the model's underlying risk-ranking behavior.

## The Threshold: 0.20 (0.48 is OBSOLETE)
The operating threshold was **re-derived after calibration**, not carried over — probability
calibration shifts the scale on which any given threshold operates, so reusing a pre-calibration
threshold would be a scale-mismatch error. The threshold was chosen via a training-data-only
cost-minimization sweep, using an empirically-derived cost ratio (see "Business Value" below), not
by naively maximizing F1. **Any reference to threshold 0.48 elsewhere in this project's history
refers to the pre-calibration configuration and must not be used with this artifact.**

## Business Value and Its Limits
An empirical cost ratio was derived from training data alone: missing a bad loan (false negative)
costs **~3.73x** more on average ($7,621 average net loss per charged-off loan) than wrongly
flagging a good one (false positive, $2,040 average forgone interest) — this justified a
recall-favoring threshold. Applying this ratio to the **final calibrated model's** test-set confusion matrix (threshold 0.20)
yields estimated figures of ~$23.9M (avoided-default value, from 7,061 correctly-flagged bad loans netted
against 3,133 still-missed bad loans) vs. ~$29.3M (foregone-interest opportunity cost, from 14,362
wrongly-flagged good loans).

**These are historical backtest estimates, not projected savings.** They describe what the
historical outcomes would suggest if this exact model had scored these exact historical loans —
they are not a guarantee of future financial impact, do not assume a flagged loan's loss is
actually avoided (that depends on the manual review outcome), and should not be presented to
stakeholders as a P&L projection. A pilot deployment with measured realized outcomes is the
appropriate way to establish an actual financial case.

## Key Explainability Findings
- **`installment_to_income`** (engineered affordability ratio) is the strongest genuinely
  applicant-level risk driver — a real debt-burden signal, not a proxy for another party's decision.
- **`term`** is the single strongest overall driver in both the tree ensemble (SHAP) and the
  Logistic Regression benchmark, but is **confirmed to be a partial proxy for LendingClub's own
  risk grading** (60-month loans are heavily concentrated in riskier LendingClub-assigned grades
  and carry higher rates/larger amounts) — **this is not evidence that loan term independently
  causes default risk**, and must never be presented as such.
- **`ever_delinquent`'s Logistic Regression coefficient must be read jointly with
  `mths_since_last_delinq`**, not in isolation — the isolated coefficient's sign is an artifact of
  the sentinel-value encoding scale, not a genuine finding that delinquency lowers risk (the
  combined effect confirms the model behaves correctly: any delinquency is riskier than none,
  and recent delinquency is riskier than old).
- DTI, income, revolving utilization, and credit-inquiry recency all show conventional,
  business-interpretable, direction-consistent associations with risk across both models.

## Fairness Considerations
**This dataset contains no protected-class attributes** (no race, ethnicity, gender, age, or
similar) — none were inferred or constructed, consistent with responsible modeling practice.
An error-rate disparity check was performed using the only legitimate available group attributes:
geography (state) and home ownership status.

- Geographic false-positive rates ranged 33.8%-41.4% across the 12 highest-volume states;
  false-negative rates ranged 24.2%-34.2%. Home ownership showed a similar pattern (MORTGAGE:
  lowest bad rate and lowest FPR; RENT/OWN: higher on both).
- **These disparities track closely with each group's underlying bad rate** — a known,
  statistically-expected property of any single-threshold classifier applied across groups with
  differing base rates, not necessarily evidence of inappropriate disparate treatment.
- **This analysis is not, and does not substitute for, a formal fair-lending / compliance
  assessment.** It should inform ongoing monitoring, not stand in for legal/compliance review
  before any production deployment decision.

## Limitations (Full List)
1. Modest discrimination (ROC-AUC ≈ 0.71) — not a highly precise risk-detection system.
2. Substantial false-positive volume (~66% of flagged applications are actually good loans) —
   review capacity must be sized accordingly.
3. Meaningful residual false negatives (~31% of bad loans missed even at the recall-favoring threshold).
4. Trained on a censoring-affected population (Current/unresolved loans excluded) skewed toward
   older vintages.
5. Self-reported income and other applicant-provided fields were not independently verified for
   every record.
6. `term` is a partial risk-grading proxy, not a fully independent applicant-behavior signal.
7. Scoped specifically to 2007-2014 LendingClub unsecured consumer loans — not validated for any
   other lender, product, or time period.
8. Fairness analysis is limited to non-protected available attributes (geography, home ownership)
   and does not constitute a compliance clearance.

## Deployment Recommendation
Deploy as a **decision-support / screening layer** feeding a human-review workflow, with:
- Ongoing monitoring of ROC-AUC, Brier score, and the geography/home-ownership error-rate gaps
  documented above, watching for drift from the reported baseline figures.
- A formal fair-lending/compliance review before any production go-live decision.
- A pilot period with realized-outcome tracking before any of the backtest dollar figures are used
  in a financial business case.
- Re-validation before extending use beyond the training population's scope (product, lender, era).
