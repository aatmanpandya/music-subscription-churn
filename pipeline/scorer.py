"""ChurnScorer: the model bundle (booster + preprocessor + calibrator + metadata).

A bundle is a directory:
    model.ubj             XGBoost booster (native format)
    preprocessing.json    pinned feature list and categorical vocabularies
    calibration.json      post-hoc calibrator (identity for a freshly trained model)
    metadata.json         training provenance
    reference_profile.json  frozen reference for the monitor (optional at serving time)

The same class is used by training (to evaluate), batch scoring, monitoring and the API,
so there is exactly one implementation of "raw rows in, probability out".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.calibration import Calibrator, Identity
from pipeline.features import Preprocessor
from pipeline.schema import validate_features


class ChurnScorer:
    def __init__(self, booster: xgb.Booster, prep: Preprocessor, calibrator: Calibrator,
                 metadata: dict | None = None):
        self.booster = booster
        self.prep = prep
        self.calibrator = calibrator
        self.metadata = metadata or {}
        self.booster.set_param({"nthread": int(os.environ.get("CHURN_NTHREAD", "4"))})

    # -- persistence -------------------------------------------------------------
    def save(self, bundle_dir: str | Path, profile: dict | None = None) -> Path:
        d = Path(bundle_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(d / "model.ubj")
        self.prep.save(d / "preprocessing.json")
        (d / "calibration.json").write_text(json.dumps(self.calibrator.to_dict()))
        (d / "metadata.json").write_text(json.dumps(self.metadata, indent=1, default=str))
        if profile is not None:
            (d / "reference_profile.json").write_text(json.dumps(profile))
        return d

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "ChurnScorer":
        d = Path(bundle_dir)
        booster = xgb.Booster()
        booster.load_model(d / "model.ubj")
        prep = Preprocessor.load(d / "preprocessing.json")
        calibrator = Calibrator.from_dict(json.loads((d / "calibration.json").read_text()))
        meta_path = d / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(booster, prep, calibrator, metadata)

    @staticmethod
    def load_profile(bundle_dir: str | Path) -> dict | None:
        p = Path(bundle_dir) / "reference_profile.json"
        return json.loads(p.read_text()) if p.exists() else None

    def with_calibrator(self, calibrator: Calibrator, extra_metadata: dict | None = None) -> "ChurnScorer":
        meta = {**self.metadata, **(extra_metadata or {})}
        return ChurnScorer(self.booster, self.prep, calibrator, meta)

    # -- scoring -----------------------------------------------------------------
    def raw_proba_from_matrix(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X, enable_categorical=True))

    def predict(self, df: pd.DataFrame, validate: bool = True, **validate_kwargs) -> pd.DataFrame:
        """Score raw feature rows. Returns raw_probability and churn_probability (calibrated)."""
        if validate:
            validate_features(df, **validate_kwargs)
        raw = self.raw_proba_from_matrix(self.prep.transform(df))
        return pd.DataFrame({
            "raw_probability": raw,
            "churn_probability": self.calibrator.transform(raw),
        }, index=df.index)
