# Monitoring report: March 2017 cohort, labels not yet available

**Overall status: WARN**

- Model: `kkbox-churn` v1 (alias `champion`, calibration `identity`)
- Batch: `features_oot.parquet`, 970,960 subscribers
- Reference: training cohort `features.parquet` (frozen profile stored with the model)

| Component | Status |
|---|---|
| feature drift | WARN |
| prediction drift | OK |
| calibration | n/a |
| ranking performance | n/a |

Thresholds: PSI warn 0.1 / alert 0.25; calibration gap warn 10% / alert 20% of the actual rate; ECE/actual warn 10% / alert 20%; AUC drop warn 0.01 / alert 0.03.

## Prediction drift

- Score PSI vs reference: **0.007** (OK)
- Mean raw score: reference 6.37% -> batch 6.51% (+2.2%)

## Calibration and performance (labels available)

Labels not supplied: not evaluated. Re-run with labels once the churn window has closed.

## Feature drift (ranked by PSI)

1 feature(s) at ALERT, 2 at WARN, of 27 (raw worst-PSI status ALERT; the component status caps ALERTs on features with under 1% of model gain at WARN, because drift in an input the model barely uses cannot move its predictions much).

| # | feature | PSI | status | model importance | null rate ref -> batch | mean ref -> batch |
|---|---|---|---|---|---|---|
| 1 | subscriber_tenure_days | 0.697 | ALERT | 0.3% | 0.2% -> 0.3% | 485 -> 501 |
| 2 | days_to_expiry | 0.191 | WARN | 0.4% | 0.2% -> 0.3% | 16.9 -> 19.1 |
| 3 | n_transactions | 0.159 | WARN | 0.3% | 0.2% -> 0.3% | 15.2 -> 15.8 |
| 4 | list_price | 0.013 | OK | 0.6% | 0.2% -> 0.3% | 150 -> 152 |
| 5 | days_since_last_listen | 0.011 | OK | 0.3% | 12.8% -> 12.4% | 23.5 -> 24.1 |
| 6 | hours_change | 0.008 | OK | 0.1% | 0.0% -> 0.0% | 0.588 -> -0.447 |
| 7 | account_age_days | 0.003 | OK | 0.1% | 11.7% -> 11.3% | 1.25e+03 -> 1.26e+03 |
| 8 | payment_method | 0.003 | OK | 0.4% | 0.2% -> 0.3% |  |
| 9 | active_days_prev30 | 0.002 | OK | 0.3% | 0.0% -> 0.0% | 13 -> 13 |
| 10 | amount_paid | 0.002 | OK | 0.7% | 0.2% -> 0.3% | 149 -> 152 |

![psi](psi.png)

Categorical codes unseen in training: payment_method 0.000%
