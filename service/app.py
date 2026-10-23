"""FastAPI service for the champion churn model.

    uvicorn service.app:app --port 8000            # needs service/model_bundle (see below)

    python -m pipeline.export_champion             # writes service/model_bundle from the registry

Endpoints
    GET  /health   liveness + whether a model is loaded
    GET  /model    registry version, calibration, training provenance, feature list
    POST /predict  one subscriber (a JSON object) or a batch ({"subscribers": [...]}, <= 1000)

Input is validated twice with the SAME rules the batch pipeline uses: Pydantic
(types and ranges, generated from pipeline.schema) and then pipeline.schema's
validate_features. Preprocessing and scoring go through pipeline.scorer, the same
class training and batch scoring use.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, create_model

from pipeline.schema import CATEGORICAL, FEATURES, NUMERIC, SchemaValidationError, validate_features
from pipeline.scorer import ChurnScorer

BUNDLE_DIR = Path(os.environ.get("MODEL_BUNDLE_DIR", Path(__file__).resolve().parent / "model_bundle"))
MAX_BATCH = 1000


# ---- request schema generated from pipeline.schema so API and pipeline cannot disagree ----
def _fields():
    fields = {"msno": (Optional[str], Field(None, description="optional subscriber id, echoed back"))}
    for name, spec in {**NUMERIC, **{k: v for k, v in CATEGORICAL.items() if v}}.items():
        required = spec.max_null == 0.0
        fields[name] = (
            float if required else Optional[float],
            Field(... if required else None, ge=spec.lo, le=spec.hi,
                  description=("required; " if required else "null allowed; ") + f"range [{spec.lo}, {spec.hi}]"),
        )
    fields["gender"] = (Optional[Literal["male", "female"]], Field(None, description="male, female or null"))
    return fields


Subscriber = create_model("Subscriber", __config__=ConfigDict(extra="forbid"), **_fields())


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subscribers: Annotated[list[Subscriber], Field(min_length=1, max_length=MAX_BATCH)]


def _which(v) -> str:
    """Route a body to the batch or single schema so validation errors name the right fields."""
    if isinstance(v, dict):
        return "batch" if "subscribers" in v else "single"
    return "batch" if isinstance(v, BatchRequest) else "single"


PredictBody = Annotated[
    Union[Annotated[Subscriber, Tag("single")], Annotated[BatchRequest, Tag("batch")]],
    Discriminator(_which),
]


class Prediction(BaseModel):
    msno: Optional[str] = None
    churn_probability: float = Field(description="calibrated probability of not renewing within 30 days of expiry")
    raw_probability: float = Field(description="uncalibrated booster output")


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    calibration: str
    n: int
    predictions: list[Prediction]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer = ChurnScorer.load(BUNDLE_DIR)
    serving = json.loads((BUNDLE_DIR / "serving.json").read_text())
    state.update(scorer=scorer, serving=serving, loaded_at=time.time())
    # warm-up so the first real request does not pay one-off initialisation cost
    row = {k: 0.0 for k in FEATURES if k != "gender"} | {"gender": None}
    scorer.predict(pd.DataFrame([row]), validate=False)
    yield


app = FastAPI(title="KKBox churn model", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 with field locations and messages only. The default body echoes the offending
    input, which crashes JSON encoding (500) when the input is inf or NaN."""
    detail = [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health():
    ok = "scorer" in state
    return JSONResponse({"status": "ok" if ok else "unavailable", "model_loaded": ok},
                        status_code=200 if ok else 503)


@app.get("/model")
def model_info():
    s, sc = state["serving"], state["scorer"]
    md = sc.metadata
    return {
        **s,
        "n_features": len(FEATURES), "features": FEATURES,
        "trained_at": md.get("trained_at"), "training_data": md.get("training_data"),
        "training_base_rate": md.get("training_base_rate"),
        "calibration_method": md.get("calibration_method", s.get("calibration")),
        "calibration_fit_cohort": md.get("calibration_fit_cohort"),
        "xgboost_version": md.get("xgboost_version"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictBody):
    rows = body.subscribers if isinstance(body, BatchRequest) else [body]
    df = pd.DataFrame([r.model_dump() for r in rows])
    for c in FEATURES:                      # all-None columns arrive as object dtype
        if c != "gender":
            df[c] = pd.to_numeric(df[c]).astype("float64")
    try:
        validate_features(df, check_null_rates=False)
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=e.violations)
    scores = state["scorer"].predict(df, validate=False)
    s = state["serving"]
    return PredictResponse(
        model_name=s["model_name"], model_version=s["model_version"], calibration=s["calibration"],
        n=len(df),
        predictions=[Prediction(msno=m, churn_probability=float(p), raw_probability=float(r))
                     for m, p, r in zip(df["msno"], scores["churn_probability"], scores["raw_probability"])])
