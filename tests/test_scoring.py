"""Batch scoring end to end against a throwaway MLflow registry."""
import numpy as np
import pandas as pd
import pytest

from pipeline import config, registry
from pipeline import score as score_cli
from pipeline.schema import SchemaValidationError


def test_champion_alias_exists_after_training(trained):
    client = registry.configure()
    mv = registry.get_version(client, alias=config.CHAMPION_ALIAS)
    assert str(mv.version) == str(trained["version"])


def test_score_cli_output_is_well_formed(trained, train_frame, tmp_path):
    inp, out = tmp_path / "in.parquet", tmp_path / "scores.parquet"
    batch = train_frame.head(1500).drop(columns=["is_churn"])
    batch.to_parquet(inp, index=False)
    assert score_cli.main(["--input", str(inp), "--output", str(out)]) == 0
    s = pd.read_parquet(out)
    assert list(s.columns[:3]) == ["msno", "raw_probability", "churn_probability"]
    assert {"model_name", "model_version", "calibration", "scored_at"} <= set(s.columns)
    assert len(s) == len(batch) and (s.msno.to_numpy() == batch.msno.to_numpy()).all()
    assert s.churn_probability.between(0, 1).all() and s.churn_probability.notna().all()
    assert s.model_version.eq(str(trained["version"])).all()
    assert s.model_name.eq(config.MODEL_NAME).all()


def test_registry_scores_match_in_memory_model(trained, train_frame):
    scorer, _, mv = registry.load_scorer(alias=config.CHAMPION_ALIAS)
    a = scorer.predict(train_frame.head(500))["churn_probability"].to_numpy()
    b = trained["scorer"].predict(train_frame.head(500))["churn_probability"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-7)


def test_pyfunc_flavor_loads_by_alias(trained, train_frame):
    import mlflow
    mlflow.set_tracking_uri(config.tracking_uri())
    m = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
    out = m.predict(train_frame.head(200).drop(columns=["is_churn"]))
    assert list(out.columns) == ["raw_probability", "churn_probability"] and len(out) == 200


def test_bad_batch_is_rejected_not_scored(trained, train_frame, tmp_path):
    bad = train_frame.head(300).drop(columns=["is_churn"]).copy()
    bad.loc[0, "hours_30"] = 5000.0
    inp, out = tmp_path / "bad.parquet", tmp_path / "scores.parquet"
    bad.to_parquet(inp, index=False)
    assert score_cli.main(["--input", str(inp), "--output", str(out)]) == 3
    assert not out.exists()


def test_predict_raises_on_schema_violation(trained, train_frame):
    bad = train_frame.head(100).drop(columns=["plan_days"])
    with pytest.raises(SchemaValidationError):
        trained["scorer"].predict(bad)
