"""Training pipeline: validate -> preprocess -> train -> evaluate -> log -> register.

    python -m pipeline.train                       # train on data/features.parquet (Feb cohort)
    python -m pipeline.train --no-register         # train + evaluate only

Reproduces the notebook's model: 23 numeric + 4 categorical features, 80/20 stratified
split (seed 42), a further 10% of the train part held for early stopping, XGBoost
(hist, categorical support, lr 0.05, subsample/colsample 0.8, min_child_weight 5),
max_depth chosen from {4, 6, 8} by validation AUC.

Feature building (raw Kaggle CSVs -> data/*.parquet) is upstream: src/build_features.py.
The raw CSVs are no longer on disk, so this pipeline starts from the feature parquets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from monitoring.profile import build_profile
from pipeline import config, registry
from pipeline.calibration import Identity
from pipeline.features import Preprocessor, load_features
from pipeline.metrics import summarize
from pipeline.schema import FEATURES, LABEL_COL, validate_features
from pipeline.scorer import ChurnScorer


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_xgb(X_tr, y_tr, X_val, y_val, depth: int, seed: int, n_jobs: int, max_trees: int = 3000,
            lr: float = 0.05) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(
        n_estimators=max_trees, learning_rate=lr, max_depth=depth,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", enable_categorical=True, eval_metric="auc",
        early_stopping_rounds=100, random_state=seed, n_jobs=n_jobs)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def train(args) -> dict:
    t0 = time.time()
    feat_path = Path(args.features)
    df = load_features(feat_path)
    validate_features(df, require_label=True, require_unique_ids=True)   # fails loudly
    print(f"schema OK: {len(df):,} rows, churn {df[LABEL_COL].mean():.2%}")

    prep = Preprocessor.fit(df)           # categories pinned from the training file
    X = prep.transform(df)
    y = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=args.seed)

    tuning = {}
    for depth in args.depths:
        m = fit_xgb(X_tr, y_tr, X_val, y_val, depth, args.seed, args.n_jobs, args.max_trees, args.learning_rate)
        tuning[depth] = (m, float(m.best_score), int(m.best_iteration))
        print(f"max_depth={depth}: validation AUC {m.best_score:.4f} at {m.best_iteration} trees "
              f"({time.time() - t0:.0f}s elapsed)")
    best_depth = max(tuning, key=lambda d: tuning[d][1])
    clf, best_val_auc, best_iter = tuning[best_depth]
    # Keep only the trees up to the early-stopping point so Booster.predict (which uses all
    # trees) matches XGBClassifier.predict_proba exactly.
    booster = clf.get_booster()[: best_iter + 1]

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "best_max_depth": best_depth,
        "best_iteration": best_iter,
        "n_trees": best_iter + 1,
        "train_rows": int(len(X_train)),
        "training_base_rate": float(y_train.mean()),
        "training_data": str(feat_path.name),
        "training_data_sha256": sha256(feat_path),
        "xgboost_version": xgb.__version__,
        "python": platform.python_version(),
        "features": FEATURES,
    }
    scorer = ChurnScorer(booster, prep, Identity(), metadata)

    # In-time evaluation on the held-out 20%.
    p_test = scorer.raw_proba_from_matrix(X_test)
    assert np.allclose(p_test, clf.predict_proba(X_test)[:, 1], atol=1e-6), "booster slice != classifier"
    test_metrics = summarize(y_test, p_test)
    print("Feb held-out test:", {k: round(v, 4) for k, v in test_metrics.items() if k != "n"})

    # Out-of-time evaluation (March), through the exact same preprocessing/scoring path.
    oot_metrics = None
    if args.oot_features and Path(args.oot_features).exists():
        oot = load_features(args.oot_features)
        oot_p = scorer.predict(oot, validate=True, require_label=True)["raw_probability"]
        oot_metrics = summarize(oot[LABEL_COL], oot_p)
        print("Mar out-of-time:  ", {k: round(v, 4) for k, v in oot_metrics.items() if k != "n"})

    importance = booster.get_score(importance_type="gain")
    profile = build_profile(
        df, prep, ref_scores=p_test, ref_labels=y_test.to_numpy(), importance=importance,
        reference_metrics=test_metrics)

    result = {"metrics": test_metrics, "oot_metrics": oot_metrics, "metadata": metadata,
              "scorer": scorer, "profile": profile, "tuning": {d: v[1:] for d, v in tuning.items()}}

    with tempfile.TemporaryDirectory() as tmp:
        bundle = scorer.save(Path(tmp) / "bundle", profile)
        if args.bundle_out:
            scorer.save(args.bundle_out, profile)
        if args.register:
            client = registry.configure()
            with mlflow.start_run(run_name=args.run_name) as run:
                mlflow.log_params({
                    "learning_rate": args.learning_rate, "subsample": 0.8, "colsample_bytree": 0.8,
                    "min_child_weight": 5, "early_stopping_rounds": 100, "tree_method": "hist",
                    "seed": args.seed, "n_jobs": args.n_jobs, "depth_grid": list(args.depths),
                    "best_max_depth": best_depth, "best_iteration": best_iter,
                    "n_features": len(FEATURES), "train_rows": len(X_train),
                    "training_data": feat_path.name,
                    "training_data_sha256": metadata["training_data_sha256"][:16],
                    "calibrator": "identity"})
                for d, (_, auc, it) in tuning.items():
                    mlflow.log_metric(f"val_auc_depth_{d}", auc)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                if oot_metrics:
                    mlflow.log_metrics({f"oot_{k}": v for k, v in oot_metrics.items()})
                mlflow.log_dict(importance, "feature_gain.json")
                mlflow.log_dict(
                    {"reference_metrics": test_metrics, "oot_metrics": oot_metrics}, "evaluation.json")
                info = registry.log_and_register(
                    bundle, pip_requirements=[
                        f"xgboost=={xgb.__version__}", f"pandas=={pd.__version__}",
                        f"numpy=={np.__version__}", "pyarrow"])
                version = info.registered_model_version
                if not registry.has_alias(client, config.CHAMPION_ALIAS):
                    registry.set_alias(client, config.CHAMPION_ALIAS, version)
                    role = "champion (first version)"
                else:
                    registry.set_alias(client, config.CHALLENGER_ALIAS, version)
                    role = "challenger (promote explicitly)"
                client.set_model_version_tag(config.MODEL_NAME, version, "calibration", "identity")
                client.set_model_version_tag(config.MODEL_NAME, version, "trained_on", feat_path.name)
                print(f"registered {config.MODEL_NAME} v{version} as {role}; run {run.info.run_id}")
                result["version"] = version
                result["run_id"] = run.info.run_id
    print(f"done in {time.time() - t0:.0f}s")
    return result


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(config.TRAIN_FEATURES))
    ap.add_argument("--oot-features", default=str(config.MARCH_FEATURES),
                    help="optional out-of-time labelled file evaluated after training")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--n-jobs", type=int, default=4, help="XGBoost threads (machine is shared)")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-trees", type=int, default=3000, help="upper bound before early stopping")
    ap.add_argument("--register", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--run-name", default="train")
    ap.add_argument("--bundle-out", default=None, help="also save the bundle directory here")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse())
