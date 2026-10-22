"""MLflow tracking + model registry helpers (SQLite backend, local artifact folder).

Registry vocabulary used in this project:
  * version   immutable snapshot of (booster + preprocessing + calibrator + profile)
  * alias     a movable pointer. `champion` is what production scores with,
              `challenger` is a candidate awaiting promotion, `previous` is the
              last champion (one-step rollback). Aliases replace the old
              Staging/Production "stages", which MLflow deprecated.

CLI:
    python -m pipeline.registry list
    python -m pipeline.registry promote --version 2      # move `champion` to version 2
    python -m pipeline.registry rollback                  # champion <- previous
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from pipeline import config
from pipeline.scorer import ChurnScorer

REPO_PIPELINE_DIR = str(Path(__file__).resolve().parent)


def configure() -> MlflowClient:
    mlflow.set_tracking_uri(config.tracking_uri())
    client = MlflowClient()
    if client.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        client.create_experiment(config.EXPERIMENT_NAME, artifact_location=config.artifact_root())
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    return client


def log_and_register(bundle_dir: str | Path, *, signature_input=None, pip_requirements=None):
    """Log the bundle as a pyfunc model inside the active run and register a new version."""
    from pipeline.pyfunc_model import ChurnPyfunc

    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=ChurnPyfunc(),
        artifacts={"bundle": str(bundle_dir)},
        code_paths=[REPO_PIPELINE_DIR],
        pip_requirements=pip_requirements,
        input_example=signature_input,
        registered_model_name=config.MODEL_NAME,
    )
    return info


def get_version(client: MlflowClient, *, alias: str | None = None, version: str | int | None = None):
    if version is not None:
        return client.get_model_version(config.MODEL_NAME, str(version))
    return client.get_model_version_by_alias(config.MODEL_NAME, alias or config.CHAMPION_ALIAS)


def has_alias(client: MlflowClient, alias: str) -> bool:
    try:
        client.get_model_version_by_alias(config.MODEL_NAME, alias)
        return True
    except mlflow.exceptions.MlflowException:
        return False


def load_scorer(*, alias: str | None = config.CHAMPION_ALIAS, version: str | int | None = None):
    """Resolve alias/version in the registry, download the bundle, return (scorer, bundle_dir, mv)."""
    client = configure()
    mv = get_version(client, alias=None if version else alias, version=version)
    model_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{config.MODEL_NAME}/{mv.version}")
    bundle = Path(model_dir) / "artifacts" / "bundle"
    return ChurnScorer.load(bundle), bundle, mv


def set_alias(client: MlflowClient, alias: str, version: str | int) -> None:
    client.set_registered_model_alias(config.MODEL_NAME, alias, str(version))


def promote(client: MlflowClient, version: str | int) -> None:
    """Point `champion` at `version`, remembering the old champion as `previous`."""
    if has_alias(client, config.CHAMPION_ALIAS):
        old = get_version(client, alias=config.CHAMPION_ALIAS).version
        if str(old) != str(version):
            set_alias(client, config.PREVIOUS_ALIAS, old)
    set_alias(client, config.CHAMPION_ALIAS, version)
    if has_alias(client, config.CHALLENGER_ALIAS) and \
            str(get_version(client, alias=config.CHALLENGER_ALIAS).version) == str(version):
        client.delete_registered_model_alias(config.MODEL_NAME, config.CHALLENGER_ALIAS)  # no longer pending


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("promote")
    p.add_argument("--version", required=True)
    sub.add_parser("rollback")
    args = ap.parse_args(argv)
    client = configure()
    if args.cmd == "list":
        for v in sorted(int(m.version) for m in client.search_model_versions(f"name='{config.MODEL_NAME}'")):
            mv = client.get_model_version(config.MODEL_NAME, str(v))   # search results omit aliases
            print(f"v{mv.version}  aliases={mv.aliases}  run={mv.run_id[:8]}  tags={dict(mv.tags)}")
    elif args.cmd == "promote":
        promote(client, args.version)
        print(f"champion -> v{args.version}")
    elif args.cmd == "rollback":
        prev = get_version(client, alias=config.PREVIOUS_ALIAS).version
        promote(client, prev)
        print(f"champion -> v{prev} (rollback)")


if __name__ == "__main__":
    main()
