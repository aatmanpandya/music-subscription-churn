# Run the production pipeline end to end. Requires .venv with requirements.txt + requirements-prod.txt.
PY := .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: all clean train score monitor recalibrate promote test export image serve-docker
all: train score monitor promote

clean:            ## fresh MLflow store and outputs
	rm -rf mlflow.db mlartifacts outputs service/model_bundle

train:            ## validate, train (Feb), evaluate, log + register v1 as `champion`
	$(PY) -m pipeline.train

score:            ## batch-score the March cohort with the champion
	$(PY) -m pipeline.score --input data/features_oot.parquet --output outputs/scores/march_v1.parquet

monitor:          ## drift + calibration report for that batch (exit code 2 = ALERT)
	-$(PY) -m monitoring.run_monitoring --features data/features_oot.parquet \
	    --scores outputs/scores/march_v1.parquet --batch-name "March 2017 cohort, champion v1" \
	    --out-dir outputs/monitoring/march_v1

recalibrate:      ## fit on half of March, evaluate on the other half, register a challenger
	$(PY) -m monitoring.recalibrate

promote:          ## recalibrate and apply the promotion rule (moves `champion` if it passes)
	$(PY) -m monitoring.recalibrate --promote

test:
	$(PY) -m pytest -q

export:           ## write the champion's files to service/model_bundle for the API image
	$(PY) -m pipeline.export_champion

image: export
	docker build -f service/Dockerfile -t kkbox-churn-api .

serve-docker:
	docker run --rm -p 8000:8000 kkbox-churn-api
