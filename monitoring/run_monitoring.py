"""Monitoring job: compare a scoring batch with the training reference.

    python -m monitoring.run_monitoring --features data/features_oot.parquet \
        --batch-name march_2017 --out-dir outputs/monitoring/march_v1

Inputs
  --features   the batch's feature parquet (also carries `is_churn` if labels have arrived)
  --scores     optional scoring output from `pipeline.score` (must match the model version);
               if omitted the batch is scored here with the alias/version given
  --labels     optional parquet with msno,is_churn when labels live in a separate file
  --no-labels  ignore any is_churn column (monitor as if labels had not arrived yet)

Exit code: 0 for OK/WARN, 2 for ALERT, 3 if the batch fails schema validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from monitoring import drift
from monitoring.report import write_report
from pipeline import config, registry
from pipeline.features import load_features
from pipeline.schema import ID_COL, LABEL_COL, SchemaValidationError, validate_features


def monitor(features: pd.DataFrame, scorer, profile: dict, *, scores: pd.DataFrame | None = None,
            labels=None) -> dict:
    """Library entry point (used by the CLI and tests)."""
    if scores is None:
        scores = scorer.predict(features, validate=False)
        raw, final = scores["raw_probability"].to_numpy(), scores["churn_probability"].to_numpy()
    else:
        raw, final = scores["raw_probability"].to_numpy(), scores["churn_probability"].to_numpy()
    return drift.run(features, raw, final, profile, scorer.prep, labels=labels)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(config.MARCH_FEATURES))
    ap.add_argument("--scores")
    ap.add_argument("--labels")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--alias", default=config.CHAMPION_ALIAS)
    ap.add_argument("--version")
    ap.add_argument("--batch-name", default="batch")
    ap.add_argument("--out-dir", default=str(config.OUTPUTS / "monitoring" / "latest"))
    args = ap.parse_args(argv)

    scorer, bundle, mv = registry.load_scorer(alias=args.alias, version=args.version)
    from pipeline.scorer import ChurnScorer
    profile = ChurnScorer.load_profile(bundle)
    df = load_features(args.features)

    try:
        validate_features(df)
    except SchemaValidationError as e:
        print("SCHEMA VALIDATION FAILED; batch not monitored/scored\n", e, file=sys.stderr)
        return 3

    labels = None
    if not args.no_labels:
        if args.labels:
            lab = pd.read_parquet(args.labels)[[ID_COL, LABEL_COL]]
            df = df.drop(columns=[LABEL_COL], errors="ignore").merge(lab, on=ID_COL, how="inner", validate="1:1")
        if LABEL_COL in df.columns:
            labels = df[LABEL_COL].to_numpy()

    scores = None
    if args.scores:
        sc = pd.read_parquet(args.scores)
        if set(sc.model_version.astype(str)) != {str(mv.version)}:
            print(f"scores were produced by model version(s) {sorted(set(sc.model_version.astype(str)))}, "
                  f"but monitoring resolved v{mv.version}; refusing to mix", file=sys.stderr)
            return 3
        sc = df[[ID_COL]].merge(sc, on=ID_COL, how="left", validate="1:1")
        if sc.churn_probability.isna().any():
            print("scores file does not cover every subscriber in the batch", file=sys.stderr)
            return 3
        scores = sc

    res = monitor(df, scorer, profile, scores=scores, labels=labels)
    meta = {
        "batch_name": args.batch_name, "features_file": Path(args.features).name,
        "model_name": config.MODEL_NAME, "model_version": mv.version, "alias": args.alias if not args.version else None,
        "calibration": scorer.calibrator.kind, "training_data": scorer.metadata.get("training_data"),
    }
    out = write_report(res, meta, args.out_dir)
    print(f"overall status: {res['overall_status']}   components: {res['components']}")
    print(f"report: {out / 'report.md'}\njson:   {out / 'report.json'}")
    return 2 if res["overall_status"] == "ALERT" else 0


if __name__ == "__main__":
    sys.exit(main())
