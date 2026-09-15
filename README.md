# Credit Risk Model — Deployment Package

## What this is
A calibrated HistGradientBoosting model that estimates the probability a loan will default
(charge off, formal default, or reach serious delinquency) versus be fully repaid, trained on
LendingClub-originated loans issued **2007-2014**. It is a **screening / prioritization tool**,
not an automatic approve/decline system — see "Intended Use" below.

## Files in this package
| File | Purpose |
|---|---|
| `credit_risk_model.joblib` | The complete, fitted pipeline: preprocessing → HistGradientBoosting → sigmoid calibration, as a single serialized object. |
| `prediction.py` | Inference module. Import `predict_credit_risk` and `load_model` from here. |
| `FEATURE_SCHEMA.md` | Full list of required input fields, types, and descriptions. |
| `MODEL_CARD.md` | Model card / executive summary: performance, calibration, fairness, limitations. |
| `README.md` | This file. |

## Quick start
```python
from prediction import load_model, predict_credit_risk
import pandas as pd

model = load_model(".")                      # loads credit_risk_model.joblib
applications = pd.read_csv("my_applications.csv")   # must match FEATURE_SCHEMA.md
results = predict_credit_risk(applications, model=model)
print(results)
```

Output columns:
- `probability_bad_risk` — calibrated probability (0-1) that the loan would default. This is a
  genuinely calibrated probability (see MODEL_CARD.md, "Calibration") — it can be read as "roughly
  X% of applications scored near this value historically defaulted," not just as a rank.
- `risk_category` — `"High Risk"` or `"Low Risk"`, based on the production threshold (0.20).
- `screening_decision` — `"High Risk / Review"` or `"Low Risk / Standard Processing"`. **Never**
  `"Approve"` or `"Decline"` — this tool flags applications for prioritized human review, it does
  not make the lending decision itself.

## The decision threshold
**Production threshold: 0.20.** This is stored in `prediction.py`'s `DEPLOYMENT_CONFIG` dict — a
single source of truth. Do not hard-code a different threshold anywhere else in downstream code.

**An earlier threshold of 0.48 appears in the project's development history and is OBSOLETE.**
It was selected before probability calibration was applied and is not compatible with this
(calibrated) model — the two thresholds live on different probability scales. If you see `0.48`
referenced anywhere in older documentation or code, it does not apply to this artifact.

## Intended use
- **Do:** use the output to prioritize applications for manual underwriting review, rank
  applications by relative risk, or feed a broader decision-support workflow alongside other
  information the model doesn't have access to.
- **Do not:** use this as a sole or automatic approve/decline mechanism. Discrimination is modest
  (ROC-AUC ≈ 0.71) and the false-positive/false-negative rates are non-trivial (see MODEL_CARD.md)
  — human review is an essential part of the intended workflow, not an optional add-on.
- **Do not:** apply this model to a different lender, loan product, or time period without
  re-validation. It is scoped to the 2007-2014 LendingClub population specifically.

## Requirements
Python 3.x with `scikit-learn`, `pandas`, `numpy`, `joblib`. The exact scikit-learn version used
to train this artifact should be used (or a compatible version) to load it — see
`MODEL_CARD.md` for the pinned version used during development.

## Support / questions
See `MODEL_CARD.md` for full performance, calibration, and fairness details, and the full
project notebook for the complete development history and rationale behind every design choice.
