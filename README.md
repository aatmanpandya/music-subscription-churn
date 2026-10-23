# KKBox Music Subscription Churn

Which subscribers of KKBox (a Taiwanese music streaming service) will not renew, why, and what to do about it.

## Dataset

**[WSDM – KKBox's Churn Prediction Challenge (Kaggle)](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data)**

The data isn't in this repository; it's about 9 GB compressed and governed by the competition rules. Download it from the link above.

| File | Rows | Contents | Used for |
|---|---|---|---|
| `train.csv` | 993K | Churn labels, memberships expiring Feb 2017 | Training cohort |
| `train_v2.csv` | 971K | Churn labels, memberships expiring Mar 2017 | Out-of-time test cohort |
| `members_v3.csv` | 6.8M | City, age, gender, sign-up channel, registration date | Both cohorts |
| `transactions.csv` | 21.5M | Payments: plan length, price, auto-renew, cancellations, expiry | Both cohorts |
| `transactions_v2.csv` | 1.4M | Further payments, including 361K dated before March missing from the file above | Out-of-time cohort only (see below) |
| `user_logs.csv` | 392M (30 GB) | Daily listening, Jan 2015 – Feb 2017 | Both cohorts |
| `user_logs_v2.csv` | 18.4M | Daily listening, March 2017 | Not used: it falls after both cutoffs |
| `sample_submission_*.csv` | | Kaggle submission templates | Not used: no information |

## Production pipeline

The model is also built out as a production-style system; see **[PRODUCTION.md](PRODUCTION.md)**. It covers:
- schema-validated training with an MLflow model registry
- batch scoring and a FastAPI service in Docker
- drift monitoring that caught a real calibration incident on the March cohort (6.5% predicted vs 9.0% actual churn)
- leak-free recalibration (ECE 0.025 → 0.001)

## Run it

1. Download the data and unpack every `.7z` archive so the CSVs sit directly in `data/`.
2. Set up and run. On macOS, XGBoost also needs `brew install libomp`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_features.py                  # ~2 min: raw CSVs -> data/*.parquet
.venv/bin/jupyter lab notebooks/churn_analysis.ipynb
```

| Path | What it is |
|---|---|
| `src/build_features.py` | DuckDB SQL over the raw files, one row of features per subscriber per cohort |
| `notebooks/churn_analysis.ipynb` | EDA, hypothesis tests, models, risk tiers, segments, out-of-time test |

## Method

- **Cohort and label:** the 992,931 subscribers whose membership expired in February 2017. KKBox counts a user as churned if they did not buy a new subscription within 30 days of expiry.
- **Leakage guard:** all features use data up to **31 January 2017**, before the churn window opens. The raw transactions run to 28 February, so they include some renewals made inside the window.
- **Models:** logistic regression and XGBoost on an 80/20 stratified split. XGBoost's tree depth was chosen on a validation slice, with early stopping.
- **Out-of-time test:** the trained models score the March 2017 cohort (`train_v2.csv`), with features built from data up to 28 February.
- **Transaction sources:** the February cohort uses `transactions.csv` only. Having rows in `transactions_v2.csv` partly signals renewing into March, so adding it to training would leak information.

## Results

| | |
|---|---|
| Labelled subscribers | 992,931 |
| Baseline churn | 6.39% |
| Data processed | 21.5M transaction rows (15.0M before the cutoff); 392M listening-log rows / 30 GB (234M before the cutoff) |
| Features | 27 (23 numeric, 4 categorical) |
| Logistic regression, Feb test | ROC-AUC 0.865, PR-AUC 0.523 |
| XGBoost, Feb test | ROC-AUC 0.899, PR-AUC 0.578 |
| XGBoost, March out-of-time | ROC-AUC 0.873, PR-AUC 0.601 |
| XGBoost, March users never in the training cohort (89K) | ROC-AUC 0.863 |
| Top 10% riskiest subscribers (Feb test) | contain 67.6% of churners (6.8x lift) |
| "At risk" (predicted ≥ 2x baseline = 12.8%) | 11.5% of subscribers, 70.6% of churners |
| Auto-renew vs manual renewal | 3.0% vs 32.4% churn |
| 5 action segments | 12.3% of subscribers, 69.2% of churners |

**Drivers:**
- Auto-renew status and a cancellation on the last transaction are by far the strongest features.
- Plan length matters: churn is 73% on sub-30-day plans and 48% on prepaid long plans, against 5% on monthly plans.
- Listening activity barely matters on its own (Cohen's d ≈ 0.1), but it matters a lot among manual renewers. For manual monthly renewers, churn is 77% with zero active days and 19% with 16+ active days. For auto-renewers it is 2–5% whatever they do.

**Out-of-time:**
- The ranking holds a month later.
- The probabilities don't: March churn was 9.0%, while the model predicted 6.5% on average. Probability thresholds should be recalibrated on recent data before use.

**Segments and actions** (each action to be validated with an A/B test):

| Segment | Users | Churn | Share of churners | Action |
|---|---|---|---|---|
| Cancelled last plan | 9,542 | 79.5% | 12.0% | Win-back offer before expiry, ask why |
| Short / trial plan | 3,627 | 73.3% | 4.2% | End-of-trial conversion offer |
| Manual, prepaid long plan | 22,756 | 47.6% | 17.1% | Early reminder plus loyalty renewal price |
| Manual monthly, disengaged | 7,929 | 54.4% | 6.8% | Re-engage (playlists, releases), then remind |
| Manual monthly, engaged | 77,798 | 23.8% | 29.2% | Incentive to switch on auto-renew |
| Auto-renewers | 871,279 | 2.2% | 30.8% | No proactive spend |

## Limitations

- **Short time span:** two monthly cohorts (Feb and Mar 2017).
- **Anonymised fields:** user IDs and payment-method codes are anonymised.
- **Bad age data:** `bd` (age) is junk for about 60% of users.
- **Corrupt listening rows:** some `total_secs` values are corrupt, so they are clipped to 0–24 hours per day.
- **No causal claims:** the segments are descriptive.
