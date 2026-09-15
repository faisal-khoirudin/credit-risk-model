# Feature Schema — Credit Risk Model Input

All 30 fields below are **required** for every scored application. Values should be raw
(unprocessed) — imputation, encoding, and scaling all happen inside the packaged pipeline.
Do not pre-impute, pre-encode, or pre-scale before calling `predict_credit_risk`.

| Field | Type | Description |
|---|---|---|
| `term` | string | Loan repayment term. One of `"36 months"` / `"60 months"` (with leading space, matching source format). |
| `loan_to_income` | float | Engineered: `loan_amnt / annual_inc`. |
| `installment_to_income` | float | Engineered: `(installment * 12) / annual_inc`. |
| `dti` | float | Debt-to-income ratio (borrower-reported, excluding mortgage and this loan). |
| `revol_util` | float | Revolving line utilization rate (%). |
| `loan_amnt` | float | Requested loan amount ($). |
| `annual_inc` | float | Self-reported annual income ($). Must be > 0 (used as a ratio denominator). |
| `inq_last_6mths` | int | Number of credit inquiries in the last 6 months. |
| `installment` | float | Monthly payment owed if the loan originates ($). |
| `tot_cur_bal` | float | Total current balance of all accounts ($). May be missing (NaN) — handled internally. |
| `employment_info_complete` | int (0/1) | Engineered: 1 if both employer title and employment length are provided. |
| `total_rev_hi_lim` | float | Total revolving high credit/credit limit ($). May be missing. |
| `delinq_2yrs` | int | Number of 30+ day delinquencies in the past 2 years. |
| `mths_since_last_major_derog` | float | Months since most recent 90+ day derogatory rating. May be missing (= never occurred). |
| `ever_major_derog` | int (0/1) | Engineered: 1 if `mths_since_last_major_derog` is present. |
| `open_acc` | int | Number of open credit lines. |
| `total_acc` | int | Total number of credit lines ever. |
| `mths_since_last_delinq` | float | Months since last delinquency. May be missing (= never occurred). |
| `ever_delinquent` | int (0/1) | Engineered: 1 if `mths_since_last_delinq` is present. **Must be interpreted jointly with `mths_since_last_delinq`, not in isolation** (see MODEL_CARD.md). |
| `pub_rec` | int | Number of derogatory public records. |
| `ever_public_record` | int (0/1) | Engineered: 1 if `mths_since_last_record` is present. |
| `mths_since_last_record` | float | Months since last public record. May be missing (= never occurred). |
| `collections_12_mths_ex_med` | int | Collections in the last 12 months, excluding medical. |
| `acc_now_delinq` | int | Number of accounts currently delinquent. |
| `emp_length_num` | int (0-10) | Employment length in years; 0 = "< 1 year", 10 = "10+ years". |
| `revol_bal` | float | Total revolving credit balance ($). |
| `tot_coll_amt` | float | Total collection amounts ever owed ($). May be missing. |
| `purpose` | string | Loan purpose category (e.g. `"debt_consolidation"`, `"small_business"`, `"credit_card"`, etc. — 14 categories, must match training-time spelling exactly; unseen categories are handled gracefully via one-hot `handle_unknown='ignore'`). |
| `home_ownership` | string | One of `"MORTGAGE"`, `"RENT"`, `"OWN"`, `"OTHER"` (values `"NONE"`/`"ANY"` should be pre-mapped to `"OTHER"` if present). |
| `verification_status` | string | Income verification status as recorded by the originating platform. |

## Missing values
Fields marked "may be missing" should be passed as `NaN` — the pipeline applies domain-appropriate
imputation internally (a fixed sentinel for the three `mths_since_*` fields, since missing means
"this event never happened," and median imputation for other numeric fields with incidental
missingness). Do not fill these yourself before calling the model.

## What NOT to include
Do not pass `grade`, `sub_grade`, `int_rate`, `funded_amnt`, `funded_amnt_inv`, or any post-origination
performance field (`total_pymnt`, `recoveries`, `last_pymnt_d`, etc.) — these were excluded from the
model by design (see MODEL_CARD.md, "Excluded Features") and passing them has no effect (the pipeline
only reads the 30 columns above) but their presence usually signals a schema mismatch worth investigating.
