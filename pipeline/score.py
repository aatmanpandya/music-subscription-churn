"""Batch scoring CLI: load a registered model by alias (or version) and score a parquet.

    python -m pipeline.score --input data/features_oot.parquet \
        --output outputs/scores/march_champion.parquet                # alias `champion`
    python -m pipeline.score --input ... --output ... --version 1     # pin a version

Output columns: msno, raw_probability, churn_probability, model_name, model_version,
calibration, scored_at. The input is schema-validated first; a bad batch is rejected
(exit code 3) rather than scored.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import config, registry
from pipeline.features import load_features
from pipeline.schema import ID_COL, SchemaValidationError, validate_features


def score_frame(df: pd.DataFrame, scorer, mv) -> pd.DataFrame:
    s = scorer.predict(df, validate=True, require_unique_ids=True)
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy()})
    out["raw_probability"] = s["raw_probability"].to_numpy()
    out["churn_probability"] = s["churn_probability"].to_numpy()
    out["model_name"] = config.MODEL_NAME
    out["model_version"] = str(mv.version)
    out["calibration"] = scorer.calibrator.kind
    out["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--alias", default=config.CHAMPION_ALIAS)
    ap.add_argument("--version", help="score with this exact version instead of an alias")
    args = ap.parse_args(argv)

    scorer, _, mv = registry.load_scorer(alias=args.alias, version=args.version)
    df = load_features(args.input)
    try:
        out = score_frame(df, scorer, mv)
    except SchemaValidationError as e:
        print("SCHEMA VALIDATION FAILED; nothing scored\n", e, file=sys.stderr)
        return 3
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"scored {len(out):,} subscribers with {config.MODEL_NAME} v{mv.version} "
          f"({'version ' + args.version if args.version else 'alias ' + args.alias}, "
          f"calibration={scorer.calibrator.kind}); mean churn probability {out.churn_probability.mean():.2%}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
