"""
Credit Risk Prediction — Inference Module
==========================================

Loads the packaged, calibrated HistGradientBoosting credit-risk model and
exposes a single function, `predict_credit_risk`, that takes raw applicant/
loan features and returns a calibrated probability of default plus a
business-facing screening decision.

This module is a SCREENING / PRIORITIZATION tool. It does not make final
lending decisions. "High Risk" applications should be routed to manual
underwriting review, not automatically declined.

Model scope: trained on LendingClub-originated loans issued 2007-2014.
Do not apply to other lenders, products, or time periods without
re-validation (see README.md and MODEL_CARD.md).
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Deployment configuration — single source of truth for the operating
# threshold and model metadata. Do not hard-code 0.20 (or any threshold)
# anywhere else; import DEPLOYMENT_CONFIG['threshold'] instead.
# ---------------------------------------------------------------------------
DEPLOYMENT_CONFIG = {
    "model_file": "credit_risk_model.joblib",
    "threshold": 0.20,
    "threshold_note": (
        "0.20 is the PRODUCTION threshold, valid only for this calibrated model. "
        "An earlier, now-OBSOLETE threshold of 0.48 was selected before calibration "
        "and must never be used with this artifact — the two are not interchangeable, "
        "since calibration changed the probability scale itself (see MODEL_CARD.md)."
    ),
    "positive_class_label": "bad_risk",  # 1 = charged off / default / serious delinquency; 0 = fully paid
    "model_version": "histgb_v1_calibrated_sigmoid",
    "training_population": "LendingClub-originated loans, issued 2007-2014",
}

REQUIRED_FEATURES = [
    "term", "loan_to_income", "installment_to_income", "dti", "revol_util", "loan_amnt",
    "annual_inc", "inq_last_6mths", "installment", "tot_cur_bal", "employment_info_complete",
    "total_rev_hi_lim", "delinq_2yrs", "mths_since_last_major_derog", "ever_major_derog",
    "open_acc", "total_acc", "mths_since_last_delinq", "ever_delinquent", "pub_rec",
    "ever_public_record", "mths_since_last_record", "collections_12_mths_ex_med", "acc_now_delinq",
    "emp_length_num", "revol_bal", "tot_coll_amt", "purpose", "home_ownership", "verification_status",
]


def load_model(model_dir="."):
    """Load the packaged preprocessing + calibrated model pipeline from disk."""
    model_path = Path(model_dir) / DEPLOYMENT_CONFIG["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Expected '{DEPLOYMENT_CONFIG['model_file']}' "
            f"in the given model_dir."
        )
    return joblib.load(model_path)


def _validate_schema(input_data: pd.DataFrame):
    missing = [c for c in REQUIRED_FEATURES if c not in input_data.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature(s): {missing}. "
            f"See FEATURE_SCHEMA.md for the full expected schema and field descriptions."
        )
    extra = [c for c in input_data.columns if c not in REQUIRED_FEATURES]
    if extra:
        # Not fatal — the pipeline only reads the columns it needs — but worth surfacing
        # since it often indicates a schema mismatch upstream (e.g. leakage columns
        # accidentally included).
        print(f"Note: {len(extra)} extra column(s) in input will be ignored: {extra}")


def predict_credit_risk(input_data, model=None, model_dir=".", threshold=None):
    """
    Score one or more loan applications for credit risk.

    Parameters
    ----------
    input_data : pandas.DataFrame
        One row per application. Must contain all columns in REQUIRED_FEATURES
        (see FEATURE_SCHEMA.md). Raw, unprocessed values — all imputation,
        encoding, and scaling happen inside the packaged pipeline.
    model : fitted estimator, optional
        A pre-loaded model (e.g. from `load_model()`). If not provided, the
        model is loaded fresh from `model_dir` on every call — pass a
        pre-loaded model in any latency-sensitive/batch context.
    model_dir : str, optional
        Directory containing `credit_risk_model.joblib`, used only if `model`
        is not provided.
    threshold : float, optional
        Overrides the default production threshold (DEPLOYMENT_CONFIG['threshold']
        = 0.20). Present for testing/what-if analysis only — production use
        should rely on the default.

    Returns
    -------
    pandas.DataFrame
        One row per input application, with columns:
        - 'probability_bad_risk': calibrated probability of default/charge-off (0-1)
        - 'risk_category': 'High Risk' or 'Low Risk', based on the threshold
        - 'screening_decision': business-facing label — 'High Risk / Review' or
          'Low Risk / Standard Processing'. Never 'Approve' or 'Decline' — this
          tool screens and prioritizes, it does not make the lending decision.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame (one row per application).")

    _validate_schema(input_data)

    if model is None:
        model = load_model(model_dir)

    active_threshold = DEPLOYMENT_CONFIG["threshold"] if threshold is None else threshold

    proba = model.predict_proba(input_data[REQUIRED_FEATURES])[:, 1]

    results = pd.DataFrame({
        "probability_bad_risk": proba,
    }, index=input_data.index)
    results["risk_category"] = np.where(proba >= active_threshold, "High Risk", "Low Risk")
    results["screening_decision"] = np.where(
        proba >= active_threshold,
        "High Risk / Review",
        "Low Risk / Standard Processing",
    )
    return results


if __name__ == "__main__":
    # Minimal smoke test when run directly: python prediction.py
    model = load_model(".")
    sample = pd.read_csv("sample_input.csv")
    output = predict_credit_risk(sample, model=model)
    print(output.to_string())
