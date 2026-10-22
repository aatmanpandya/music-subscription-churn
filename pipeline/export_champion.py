"""Export the registry's champion (or any alias/version) as a standalone serving bundle.

    python -m pipeline.export_champion --out service/model_bundle

The API image does not contain MLflow. Instead, the image is built from this folder:
the exact files of the registered model version plus `serving.json` recording which
registry version they came from, so /model can report it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config, registry

SERVING_FILES = ["model.ubj", "preprocessing.json", "calibration.json", "metadata.json"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.ROOT / "service" / "model_bundle"))
    ap.add_argument("--alias", default=config.CHAMPION_ALIAS)
    ap.add_argument("--version")
    args = ap.parse_args(argv)
    scorer, bundle, mv = registry.load_scorer(alias=args.alias, version=args.version)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in SERVING_FILES:
        shutil.copy(bundle / f, out / f)
    (out / "serving.json").write_text(json.dumps({
        "model_name": config.MODEL_NAME, "model_version": str(mv.version),
        "alias": None if args.version else args.alias,
        "calibration": scorer.calibrator.kind,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1))
    print(f"exported {config.MODEL_NAME} v{mv.version} ({scorer.calibrator.kind}) -> {out}")


if __name__ == "__main__":
    main()
