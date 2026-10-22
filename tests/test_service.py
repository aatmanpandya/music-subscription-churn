import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import service.app as svc
from pipeline.schema import FEATURES


@pytest.fixture(scope="module")
def client(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle")
    trained["scorer"].save(d)
    (d / "serving.json").write_text(json.dumps(
        {"model_name": "kkbox-churn", "model_version": str(trained["version"]),
         "alias": "champion", "calibration": "identity"}))
    svc.BUNDLE_DIR = d
    with TestClient(svc.app) as c:
        yield c


def _row(df, i=0):
    r = df.iloc[i][["msno"] + FEATURES].to_dict()
    return {k: (None if (isinstance(v, float) and np.isnan(v)) or v is None else v) for k, v in r.items()}


def test_health_and_model(client, trained):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/model").json()
    assert m["model_version"] == str(trained["version"]) and m["n_features"] == 27


def test_predict_single_subscriber(client, train_frame, trained):
    r = client.post("/predict", json=_row(train_frame))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 1 and 0 < body["predictions"][0]["churn_probability"] < 1
    expected = trained["scorer"].predict(train_frame.head(1))["churn_probability"].iloc[0]
    assert body["predictions"][0]["churn_probability"] == pytest.approx(expected, abs=1e-6)


def test_predict_batch_matches_batch_pipeline(client, train_frame, trained):
    rows = [_row(train_frame, i) for i in range(25)]
    r = client.post("/predict", json={"subscribers": rows})
    assert r.status_code == 200, r.text
    got = np.array([p["churn_probability"] for p in r.json()["predictions"]])
    want = trained["scorer"].predict(train_frame.head(25))["churn_probability"].to_numpy()
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert [p["msno"] for p in r.json()["predictions"]] == list(train_frame.msno.head(25))


def test_out_of_range_and_missing_fields_are_422(client, train_frame):
    bad = _row(train_frame)
    bad["hours_30"] = 5000
    assert client.post("/predict", json=bad).status_code == 422
    missing = _row(train_frame)
    del missing["active_days_30"]
    assert client.post("/predict", json=missing).status_code == 422
    extra = {**_row(train_frame), "surprise": 1}
    assert client.post("/predict", json=extra).status_code == 422
    nonint = {**_row(train_frame), "plan_days": 30.5}
    assert client.post("/predict", json=nonint).status_code == 422       # caught by pipeline.schema
    assert client.post("/predict", json={"subscribers": []}).status_code == 422
