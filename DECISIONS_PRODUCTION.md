# Decisions log: production pipeline, monitoring and recalibration

A running log of the choices made while turning the churn notebook into a production-style system. Each entry says what was decided, what the alternatives were, why, which numbers settled it, and what went wrong on the way. Written to be read by someone learning how the project was thought through, so the reasoning is spelled out. Dates and order follow the work.

Vocabulary used throughout:
- **Calibration**: whether a predicted 10% really means 10 in 100 churn. Different from ranking quality (AUC), which only asks whether risky people score higher than safe ones.
- **Drift**: the data the model sees in production differs from the data it was trained on.
- **Reference**: the training-time snapshot that new batches are compared with.
- **Champion / challenger**: the model version production uses, and a candidate waiting to replace it.

---

## 0. Ground rules and the starting point

**Decision.** Work on a new local branch `production-pipeline`, commit only there, never push, never touch `main`. New code goes in new folders (`pipeline/`, `monitoring/`, `service/`, `tests/`); the notebook, `src/`, `README.md` and the existing parquets are read-only.

**Why.** The repo is public and the notebook is the project's finished story. A production layer should sit beside it so a reader can see both and nothing already published breaks.

**Starting facts that shaped everything.**
- The raw Kaggle CSVs have been deleted, so `src/build_features.py` cannot be rerun. The pipeline therefore starts from the three feature parquets, and feature building is documented as upstream. The alternative (rewriting feature building to run from something) was impossible without the data.
- `features.parquet` (Feb, 992,931 rows, churn 6.39%) is training; `features_oot.parquet` (Mar, 970,960 rows, churn 8.99%) is the out-of-time set. `features_oot_v1tx.parquet` is March built from only the original transactions file; I used it once, as a diagnostic (section 5).
- The parquet columns are DuckDB nullable integers (`Int64`), not plain floats like the notebook converts them to. That detail matters for preprocessing (section 2).

---

## 1. Reproducing the notebook's model

**Decision.** Re-implement the notebook's modelling exactly (same features in the same order, same split seeds, same XGBoost settings, same depth grid) and check the numbers match before building anything on top.

**Why.** If the pipeline's model differs from the notebook's, every later number (drift, recalibration) is about a different model than the one the README describes.

**Numbers that settled it.**

| | Notebook output | Pipeline |
|---|---|---|
| Validation AUC at depth 4, 6, 8 | 0.8977, 0.8973, 0.8983 | 0.8977, 0.8973, 0.8983 |
| Trees at early stopping | 1025, 345, 307 | 1025, 345, 307 |
| Feb test ROC-AUC / PR-AUC | 0.899 / 0.578 | 0.8987 / 0.5785 |
| Mar ROC-AUC / PR-AUC | 0.873 / 0.601 | 0.8733 / 0.6008 |
| Mar mean predicted | ~0.065 | 0.0651 |

I read the notebook's stored cell outputs to compare, because the notebook prints only three decimals. The validation AUCs and tree counts are identical, which is a much stricter check than the final AUC. Depth 8 wins by 0.0006 on validation AUC, as in the notebook. This is a small margin, and I note it because "depth chosen from {4,6,8}" is not a strong choice (depths 4, 6 and 8 are within 0.001 of each other).

**Differences, honestly.** XGBoost ran with 4 threads instead of the notebook's 8, because the machine is shared with other jobs. Histogram-based XGBoost can in principle give slightly different results across thread counts; in practice the output was identical at displayed precision on three separate runs (a smoke run and two clean-start runs). I did not test 8 threads.

**Detail: trimming the booster.** `XGBClassifier.predict_proba` uses only the trees up to the early-stopping point, but a plain `Booster.predict` uses every tree trained (best iteration plus the 100 extra trees early stopping needs to confirm the plateau). The pipeline keeps `booster[: best_iteration + 1]` and asserts that its predictions equal the classifier's. Without this the saved model would silently score differently from the evaluated one.

---

## 2. One preprocessing path, categories pinned

**Decision.** A single `Preprocessor` class (`pipeline/features.py`) builds the model matrix for training, batch scoring, monitoring and the API. Its categorical vocabularies are learned once from the training file, saved as JSON in the model bundle, and reused everywhere.

**Why.** The notebook built the matrix inline for training and re-built it with a second, slightly different function for March. That works until the two drift apart. XGBoost splits on the integer code of each category, so a different vocabulary or order in a scoring batch would change predictions without any error. This is the classic "training-serving skew" bug, and the fix is to make it impossible by having one code path and one saved vocabulary.

**Choices inside it.**
- Categories are learned from the whole training file before the split, as the notebook does. Only the names of categories leak this way, never information about labels, and keeping it identical preserves the exact reproduction in section 1.
- A value never seen in training becomes missing (NaN), as the notebook does. The monitor counts the share of such values. In March it was 0.000%, so this has not mattered yet.
- Code-valued columns (city, registered_via, payment_method) arrive as `Int64`, `int64` or `float64` depending on who wrote the parquet. I convert to a clean integer string (`"5"`, not `"5.0"`) so that all three give the same vocabulary. Sort order of those strings is the same as the notebook's `"5.0"` strings, so the integer codes XGBoost sees are unchanged (the exact AUC match confirms it). A test checks that `Int64` and `float64` inputs produce identical matrices.

---

## 3. Schema validation

**Decision.** `pipeline/schema.py` holds the contract: which columns exist, numeric dtypes, a sanity range per column, whole-number and 0/1 checks, a null-rate limit per column, allowed gender values, and (for labelled data) a 0/1 label and unique ids. It collects every violation into one `SchemaValidationError` rather than stopping at the first. The scoring CLI exits with code 3 on a violation and writes nothing.

**Why "fail loudly".** A scoring job that quietly produces numbers from broken input is worse than one that stops: someone will act on those numbers. Listing all violations at once saves a debugging round trip per problem.

**Schema problems vs drift.** A deliberately drawn line:
- Schema violation: the data is structurally wrong or physically impossible (a missing column, text in a numeric field, `hours_30 = 900` when a month has 720 hours, an `auto_renew` of 7). Do not score.
- Drift: the data is valid but looks different from training. Score it, flag it.
Mixing the two would make the schema reject real but unusual batches, or make drift monitoring miss broken ones.

**How the ranges were set.** Wider than anything seen in either cohort, because a range check that rejects a legitimate extreme customer is worse than none. Real extremes I found while setting them:
- `days_to_expiry` has a minimum of -17,197 in February (123 rows are below -400): some membership expiry dates are far in the past, an upstream data quirk preserved to match the notebook. In March the maximum is 2,113 (February's was 59). The bound is [-20000, 4000].
- `n_transactions` is up to 71 in February and 244 in March; `avg_unique_songs_30` up to 587 then 1,308. Bounds are 5000 each.
- `hours_30` can reach 720.0 (24 hours for 30 days), so 720 is inclusive.

**Null limits.** About twice the training null rate, or exactly zero for columns the feature builder always fills (`active_days_30`, `hours_30`, flags): a null there means an upstream join broke. The zero limit applies to a one-row API request as well; the statistical limits apply only to batches of at least 1,000 rows, since a null rate over 3 rows means nothing.

**Alternative considered.** A library like Great Expectations or pandera. For 27 columns a 150-line module with no new dependency is easier to read and to test; a library earns its keep when many teams share contracts.

---

## 4. MLflow: SQLite store, bundles, aliases

**Decision.** Local MLflow with a SQLite database (`mlflow.db`) for runs and the registry and a local folder (`mlartifacts/`) for files. The registered model is a `pyfunc` wrapping a bundle folder (booster, `preprocessing.json`, `calibration.json`, `metadata.json`, `reference_profile.json`). Versions are promoted by moving the alias `champion`; `challenger` marks a candidate and `previous` the last champion.

**Why SQLite rather than the plain file store.** The task allowed either. The model registry in current MLflow is built around a database backend, and the plain file store is being phased out; I did not test the file-store path. SQLite costs nothing to set up and has no server.

**Why aliases rather than stages.** MLflow's older Staging/Production stages are deprecated in favour of aliases, which are movable named pointers to a version. The consumer says "give me `champion`" and never needs to know the version number, which is what lets a promotion (or a rollback to `previous`) happen without redeploying anything.

**Why the preprocessing, calibrator and reference profile live inside the model version.** A version should be an immutable, self-contained snapshot. If the vocabulary or calibrator were stored elsewhere, then "v2" would not fully identify what produced a score. It also means the recalibrated model is a clean new version (same booster, new calibrator) instead of a patch.

**The reference profile.** Each version carries a compact frozen description of the training data (bin proportions per feature, null rates, score distribution on the held-out Feb test split, reference AUC and calibration, XGBoost gain importance), so the monitor never needs the training set. The reference score distribution uses the held-out 20%, not the training rows, because the model's scores on data it was fitted to are over-confident.

**Dead end: `search_model_versions` omits aliases.** My first `registry list` printed empty alias lists after training, although the alias existed (scoring by it worked). The search endpoint does not return aliases; the fix is one `get_model_version` per version. Also after promoting I clear the `challenger` alias if it points at the promoted version, otherwise v2 showed as both challenger and champion.

**Detail: `code_paths`.** The pyfunc wrapper imports `pipeline.scorer`. MLflow copies the `pipeline/` folder into the model so `mlflow.pyfunc.load_model("models:/kkbox-churn@champion")` works without the repo on the path. A test loads it that way. Scoring itself uses the downloaded bundle directly, which is what lets the API image skip MLflow.

---

## 5. Monitoring

### 5.1 Why PSI

**Decision.** Population Stability Index per feature, with bins frozen from the reference, thresholds 0.1 (warn) and 0.25 (alert).

**What PSI is.** Split a variable into bins using the reference data. Compare the share of rows per bin in the reference (e) and the new batch (a): PSI is the sum over bins of (a - e) x ln(a / e). Zero means identical; bigger means more shifted.

**Alternatives.**
- Kolmogorov-Smirnov test: numeric only, and with a million rows it rejects for differences too small to matter (a p-value says "different", not "different enough to care").
- Chi-square on categories: same sample-size problem.
- Jensen-Shannon / Wasserstein distance: fine, but have no widely used thresholds.
- PSI works the same way for numeric, categorical and score distributions, returns one number per feature that can be ranked, and has conventional thresholds from credit-risk practice. Its weaknesses: the thresholds are rules of thumb, and results depend on how the bins are cut (below).

**Binning choices.**
- Numeric: up to 10 reference-quantile bins. Many features are heavily tied (hours of listening is 0 for many people), so duplicate edges are merged. Features with 12 or fewer distinct values (flags, day counts) get one bin per value instead.
- A separate bin for missing values, so that a change in the null rate is itself detected (a test covers it).
- Categories: one bin per training category (including "missing") plus an "unseen" bin.
- Empty bins are floored at 0.0001 so the logarithm exists.
- The bins are frozen in the reference profile, so PSI values are comparable between months.

### 5.2 What the March run showed

Final numbers (from the registered v1 on the 970,960 March subscribers):

| Component | Result |
|---|---|
| Overall | ALERT |
| Feature drift | WARN (see below) |
| Prediction drift | OK, score PSI 0.007, mean raw score 6.37% to 6.51% |
| Calibration | ALERT, predicted 6.51% vs actual 8.99%, gap 27.6% of actual, ECE 0.0248 vs 0.0012 reference |
| Ranking | WARN, ROC-AUC 0.8733 vs 0.8987 (drop 0.0254, between the warn and alert thresholds of 0.01 and 0.03) |

Top PSI: `subscriber_tenure_days` 0.697, `days_to_expiry` 0.191, `n_transactions` 0.159, then everything else below 0.02.

### 5.3 The first version raised a false-ish alarm: investigating `subscriber_tenure_days`

My first rule was simply "overall feature status = worst per-feature PSI". That gave feature drift ALERT, driven by `subscriber_tenure_days` at 0.697. Before accepting that, I looked at why.

- The distribution's upper tail moved by exactly the cutoff's move. Deciles 7 to 10 in February were 731, 740, 751, 761 days; in March 759, 767, 779, 789. The shift is about 28 days, which is how far the cutoff moved (31 Jan to 28 Feb).
- Cause: tenure is "days from first transaction to the cutoff", and the transaction data begins on a fixed date, so everyone whose first purchase is at or before the data start has tenure equal to the distance from that fixed start to the cutoff, which grew by 28 days.
- It is not caused by the second transactions file: on `features_oot_v1tx` the PSI is 0.696.
- The model barely uses it: 0.3% of total gain. The two dominant features, `auto_renew` (64.8%) and `last_tx_cancel` (25.9%), had PSI of 0.0004 and 0.0001.

So the "alert" was a calendar artifact in a nearly-unused feature. **Decision (made after seeing this, and recorded as such):** keep every raw PSI status in the table, but in the component status cap an ALERT-level PSI at WARN when the feature carries under 1% of gain. **Alternatives:** (1) keep the raw rule: simple, but a page every month for a feature that cannot change predictions teaches people to ignore the monitor; (2) drop relative-time features from monitoring: loses real signals; (3) weight PSI by importance continuously: harder to explain. **Risk to note:** gain importance is one view of "matters" and a feature the model ignores today can matter after retraining; and the rule was written after I had seen the March result, so it is not a clean pre-registered test. The report prints both statuses so the cap is visible. A test pins the behaviour.

A related general point worth knowing: any feature defined as "days since a fixed date" or "count of events so far" drifts mechanically each month. `n_transactions` (PSI 0.159) is probably another case (it is not the second file: 0.153 without it).

### 5.4 Input monitoring alone would not have caught the incident

This is the main finding of the project. With labels withheld (`--no-labels`), the overall status is WARN: nothing worse than the three mechanical features. The score distribution did not move (PSI 0.007). Only comparing predictions with outcomes showed the problem.

What actually changed, from the March scores and labels:

| | Feb | Mar |
|---|---|---|
| Auto-renew share of subscribers | 88.7% | 88.2% |
| Churn among auto-renewers | 3.0% | 4.5% |
| Churn among manual renewers | 32.4% | 41.4% |

The mix of subscribers hardly moved but churn rose within each group. In ML terms: not covariate shift (P(x) stable) but a change in P(y | x). Nothing computed from inputs can detect that; it needs labels. This is why the design has a calibration/performance component that activates when labels arrive, and why the report says so when labels are missing.

Two more March facts worth recording. 9.2% of March subscribers (89,259) were not in the February cohort, and they churned at 39.8% against 29.1% predicted; they hold 40.7% of March churners. And the model ranks the February-cohort members less well (ROC-AUC 0.814) than the pooled 0.873, because the very high-churn new users are easy to rank and lift the pooled number. This does not change any headline, but it means the pooled March AUC flatters the model on repeat subscribers.

### 5.5 Calibration metrics and thresholds

**ECE (expected calibration error)**: split predictions into bins; in each, compare mean predicted and actual churn; average the absolute gaps weighted by bin size. I use 10 **equal-count** bins as primary, because scores are very skewed (most subscribers are near 2%) and equal-width bins put nearly everyone in the first bin. I also report equal-width ECE. In March both are 0.0248, because the error is systematic (every bin is under-predicted). After isotonic they are 0.0010 and 0.0009.

**Sample size matters for ECE.** My unit-test data first produced a "calibration WARN" on a perfectly stable synthetic batch of 30,000 rows (about 1% churn): ECE has sampling noise, and ECE / actual rate came out at 16% there against 7.6% at 100,000 rows. At 970,960 rows the noise is negligible. In production, small batches need either larger windows or wider thresholds.

**Thresholds** (chosen before looking at March): calibration gap |mean predicted - actual| / actual warn 10%, alert 20%; ECE / actual rate warn 10%, alert 20%; AUC drop warn 0.01, alert 0.03. Relative gaps are used because 1 percentage point is a big error at a 3% base rate and a small one at 50%. Overall status is the worst component; components needing labels show "not evaluated" until labels exist.

---

## 6. Recalibration without leakage

### 6.1 Why recalibrate, and not retrain

The ranking is intact (AUC 0.873), only the probabilities are off. A calibrator is a small monotone map from the model's score to a corrected probability; it is cheap, fast, auditable, reversible (the old version is still there), and cannot reorder users. Retraining on March would also fix it but takes longer, changes the ranking, and needs a full re-validation cycle. Recalibration is the proportionate response; retraining is for when ranking degrades.

### 6.2 The leakage protocol

Labels for a cohort exist only after its 30-day churn window closes. The risk is tuning a calibrator on labels and then reporting how well it does on the same labels: that measures memorisation, not calibration. Protocol:
1. Split March subscribers at random, stratified by churn, into half A and half B (485,480 users each).
2. Fit every calibrator on A only.
3. Report every "after" number on B.
4. Repeat for 10 other random splits to show the result is stable (standard deviation of ECE 0.0001 to 0.0003).
5. After the check passes, refit the chosen method on all March labels for deployment (more data, same recipe). That refit's own March numbers are in-sample and are never used as evidence.

**The limit of this protocol, stated plainly.** A and B come from the same month. It proves the method fixes March-like data. It cannot prove that a calibrator fitted on March will be right for April. A later cohort is not available. The transfer check below shows the danger concretely.

### 6.3 The candidates and where the prior-shift rate comes from

**(a) Prior-shift correction.** If only the base rate changed (same relationship between features and outcome, just more churners overall), Bayes' rule gives a one-line fix: add log-odds(new rate) - log-odds(old rate) to every score's log-odds. One parameter. It needs the new base rate, which in production could come from: the base rate of the most recent matured cohort; an early read of renewals inside the churn window; or a business forecast. Here the rate is estimated from the labels in half A (8.99%), not from B.

I also tried the label-free version: the EM algorithm of Saerens et al. estimates the new base rate from the scores alone. It returned 6.71% against a true 8.99%. It fails because it assumes only the base rate changed and, as section 5.4 shows, the conditional churn rate changed. A unit test shows it works when the assumption holds (it recovers 12% from 6% in simulated data), so the code is right and the March data violates the premise. Kept as a documented diagnostic, not used.

**(b) Platt scaling.** Fit a logistic regression on the model's log-odds: two parameters (slope 0.94, intercept +0.42). Fixes level and a bit of shape.

**(c) Isotonic regression.** A monotone step function fitted to the data: the most flexible, needs the most data (fine at 485k), can overfit when data is small. Stored as 300 knots.

### 6.4 Results (held-out half B)

| Method | Mean predicted (actual 8.99%) | ECE | Brier | ROC-AUC |
|---|---|---|---|---|
| none | 6.52% | 0.0247 | 0.05373 | 0.8742 |
| prior-shift | 8.10% | 0.0093 | 0.05243 | 0.8742 |
| Platt | 9.01% | 0.0048 | 0.05240 | 0.8742 |
| isotonic | 9.01% | 0.0010 | 0.05215 | 0.8742 |

Why prior-shift only gets partway: the needed correction is not a constant in log-odds. Predicted churn is too low by a factor of 2.0 in the lowest-risk decile (0.32% vs 0.63%) and 1.2 in the highest (44.6% vs 54.3%). A single offset cannot fit that.

### 6.5 Dead end found and fixed: plain isotonic creates ties

The first version of isotonic passed every check, but the in-sample PR-AUC dropped from 0.6008 to 0.5961, and held-out PR-AUC from 0.6028 to 0.5961. Looking at why: a standard isotonic fit is a step function, so 800,022 distinct raw scores collapsed into 188 distinct values, the largest tie group holding 62,982 subscribers. ROC-AUC barely notices (0.87416 to 0.87411) but anything that picks "the top 10%" suffers, which is exactly what a retention campaign does. **Fix:** blend in a vanishing share of the raw score, `(1 - 1e-6) x isotonic + 1e-6 x raw`. This makes the map strictly increasing, so every pair the raw model ordered keeps its order, and moves probabilities by less than one millionth. After the fix PR-AUC is identical to the raw model (0.6028 held out) and a test asserts it. **Lesson:** "AUC unchanged" is a weak statement of "ranking unchanged"; check a top-heavy metric too. The scoring output also keeps `raw_probability` so downstream tools can rank on it.

### 6.6 Choosing and promoting

**Selection rule** (set before running): among the three methods, admit those whose held-out ROC-AUC is within 0.001 of the uncalibrated model and whose Brier is not worse; choose the lowest held-out ECE. Isotonic won. One honest caveat: picking the best of three on half B and then reporting half B is mildly optimistic; with 485k rows and gaps this large (0.0010 vs 0.0048) it does not change the conclusion, and the 10-split table is consistent.

**Promotion rule**: the challenger replaces the champion only if all hold on held-out data: at least 50,000 users and 1,000 positives; mean predicted within 5% of actual; ECE at most half the champion's; ROC-AUC within 0.001 of the champion's; Brier not worse; and ECE better in every one of the 10 repeated splits. It is coded (`promotion_decision`) and tested, including that it blocks a calibrator that damages ranking. March passed all seven. The champion alias moved to v2 and v1 became `previous`.

**Why a coded rule.** A promotion that depends on someone eyeballing a chart is not reproducible and not auditable.

### 6.7 Transfer check: a calibrator remembers the cohort it saw

Applying the March-fitted isotonic back to the February cohort gives a mean prediction of 8.84% against a true 6.39% (the raw model was 6.38%, almost exactly right). So the calibrator has absorbed March's base rate. If April's churn goes back to 6%, v2 will over-predict. Consequences: the calibrator must be refitted as each cohort matures, and the champion must be re-monitored on every new cohort; "recalibrate once" is not a fix, it is a loop.

---

## 7. Tests

63 pytest tests in about 12 seconds. They use a synthetic data generator, because the real parquets are gitignored and tests that need them would fail for anyone who clones the repo. A tiny model is trained through the real pipeline into a throwaway MLflow store.

What is covered, per the brief: schema violations caught (missing column, wrong dtype, out of range, flags, nulls, infinities, bad labels, duplicate ids, zero rows); preprocessing identical at train and score time (subset equals subset of full, JSON round trip, pinned categories, Int64 vs float64, scorer equals direct booster path); PSI near 0 for identical or same-distribution samples and large for shifted, known-value check, nulls, ties, categorical; scoring output well formed (columns, ids, ranges, version, bad batch rejected with no output, registry load equals in-memory model, pyfunc loads by alias); recalibration improves held-out ECE for all three methods and leaves AUC unchanged; the isotonic no-ties property; monitoring flags a miniature of the March incident but stays quiet on stable data; the API.

**Tests I got wrong first (and what it taught).**
- The first synthetic data had a weak signal, so the tiny model early-stopped at 5 trees and was itself miscalibrated; a "stable batch should be OK" test failed for that reason, not because the monitor was wrong. Fixed the generator (and used a higher learning rate for test models), not the monitor.
- I assumed prior-shift would cut ECE by more than half on the synthetic data. It did not (0.0095 to 0.0056), because the synthetic model's miscalibration is not a pure shift either. On the real March data it does halve it (0.0247 to 0.0093) while still leaving the average 0.9 points short. The test now only requires prior-shift to improve ECE and to halve the average gap, and requires Platt and isotonic to halve ECE.
- My first EM test fed the algorithm already-adjusted scores, which is not what EM assumes. Rewrote it with a proper simulation: class-conditional distributions fixed, only the base rate moved.
- The first stable-batch test used 30,000 rows and tripped on ECE noise (section 5.5); raised to 80,000.

---

## 8. The API and Docker

**Decision.** FastAPI app with `/health`, `/model`, `/predict` (one subscriber, or `{"subscribers": [...]}` up to 1,000). The image holds only the exported champion bundle and five small source files: no MLflow, no scikit-learn, no training code.

**Why export instead of loading from the registry at startup.** MLflow plus its dependencies would multiply the image size and add a startup dependency on a tracking server. `pipeline.export_champion` copies exactly the files of the champion version and writes `serving.json` with the version, so `/model` can report it and the image is traceable to a registry version. The cost: the image must be rebuilt to pick up a promotion.

**Request schema generated from `pipeline/schema.py`**, so field ranges and required-ness cannot drift from the batch pipeline. Pydantic checks types and ranges; then `validate_features` runs the same integer/binary/null checks as batch scoring (a test shows a non-integer `plan_days` is caught only at that second stage).

**Detail: error messages.** A first version took `Union[BatchRequest, Subscriber]` and reported an out-of-range value as "Field required" (the error from the wrong union branch). A discriminator that routes on the presence of the `subscribers` key now makes errors point at the actual field.

**Image size.** 144 MB compressed content, 666 MB unpacked. Most of it is scipy (118 MB), pandas (84 MB), numpy (40 MB); scipy comes in because xgboost depends on it. I used `xgboost-cpu` (the CPU-only build) to avoid GPU libraries. Smaller options exist (a different base image, dropping pandas from the serving path) but were not worth the complexity for this project.

**Latency**, measured from the host through the container's forwarded port on a loaded laptop: single subscriber p50 20 ms, p95 24 ms; batch of 1000 p50 58 ms. Most of the 20 ms is HTTP and VM overhead rather than XGBoost. Predictions through the container matched the batch pipeline exactly for 200 March subscribers.

**Environment accidents worth knowing.** The first Docker build took 19 minutes because pip downloads inside the VM were slow (a 2 MB index page took 25 s); once the dependency layer was cached, rebuilds took about 11 seconds. Ports 8000 and 8765 were already used by other processes on this machine, so the test used a random free port.

---

## 9. Environment note

Installing MLflow pulled `pyarrow` down from 26.0.0 (pinned in the original `requirements.txt`) to 25.0.1. Parquet reading is unaffected (all numbers reproduce), but the two requirement files now disagree; `requirements-prod.txt` records the resolved versions.

## 10. Not done, and why

- A scheduler (cron/Airflow) and alert delivery for the monitor: out of scope; the job is a CLI with an exit code that a scheduler could use.
- Testing recalibration on a later cohort: no April data exists.
- Testing `src/build_features.py`: raw data is gone.
- Authentication, rate limiting and structured request logging on the API.
- A comparison with the file-based MLflow store.
