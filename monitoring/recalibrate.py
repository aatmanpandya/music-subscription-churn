"""Recalibration without leakage, registered as a new model version.

    python -m monitoring.recalibrate --labelled data/features_oot.parquet --cohort march_2017
    python -m monitoring.recalibrate ... --promote        # apply the promotion rule

Why this exists: the champion ranks March well (AUC ~0.87) but predicts 6.5% churn where
9.0% occurred. Ranking is intact, probabilities are off, so only the probability map
needs fixing: the booster is untouched, a calibrator is stacked on its raw scores.

Leakage protocol
  * Labels for a cohort exist only after its churn window closes. The cohort is split
    at random BY SUBSCRIBER into half A (calibration) and half B (evaluation).
    Every calibrator is fitted on A only; every reported "after" number is measured on B.
  * Repeated for 10 random splits to show the result is not one lucky split.
  * The deployed calibrator is refitted on all labelled users AFTER the held-out check
    passed (more data, same recipe). Its own March numbers are then in-sample and are
    never used as evidence.
  * The limit of this protocol: A and B are the same month. It shows the method fixes
    March-like data; it cannot prove the fit transfers to April. See PRODUCTION.md.

Candidates: (a) prior-shift (log-odds offset for the new base rate), (b) Platt,
(c) isotonic. Label-free prior estimation (EM) is reported as a diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from pipeline import config, registry  # noqa: E402
from pipeline.calibration import Isotonic, Platt, PriorShift, estimate_prior_em  # noqa: E402
from pipeline.features import load_features  # noqa: E402
from pipeline.metrics import reliability_table, summarize  # noqa: E402
from pipeline.schema import LABEL_COL, validate_features  # noqa: E402

N_REPEATS = 10

# Promotion rule (all must hold on held-out half B, and ECE must improve in every repeat).
RULE = {
    "min_holdout_users": 50_000,
    "min_holdout_positives": 1_000,
    "max_calibration_gap_rel": 0.05,     # |mean pred - actual| / actual
    "max_ece_ratio": 0.5,                # ECE_new <= 0.5 * ECE_old
    "max_abs_auc_change": 0.001,
    "brier_not_worse": True,
    "ece_improves_in_all_repeats": True,
}
# Candidate admission rule for choosing which method to register.
ADMISSION = {"max_abs_auc_change": 0.001, "brier_not_worse": True}


def fit_candidates(raw_a, y_a, old_rate: float) -> dict:
    return {
        "prior_shift": PriorShift(old_rate, float(np.mean(y_a))),
        "platt": Platt.fit(raw_a, y_a),
        "isotonic": Isotonic.fit(raw_a, y_a),
    }


def evaluate(cands: dict, raw_b, y_b) -> dict:
    out = {"none (raw model)": summarize(y_b, raw_b)}
    for name, cal in cands.items():
        out[name] = summarize(y_b, cal.transform(raw_b))
    return out


def choose(results: dict) -> str:
    base = results["none (raw model)"]
    ok = {k: v for k, v in results.items() if k != "none (raw model)"
          and abs(v["roc_auc"] - base["roc_auc"]) <= ADMISSION["max_abs_auc_change"]
          and v["brier"] <= base["brier"]}
    if not ok:
        raise SystemExit("no calibrator passed the admission rule; keep the current champion")
    return min(ok, key=lambda k: ok[k]["ece_quantile10"])


def promotion_decision(results: dict, chosen: str, n_b: int, pos_b: int, repeat_ece: dict) -> dict:
    old, new = results["none (raw model)"], results[chosen]
    checks = {
        "enough_holdout_users": n_b >= RULE["min_holdout_users"],
        "enough_holdout_positives": pos_b >= RULE["min_holdout_positives"],
        "calibration_gap_within_5pct": abs(new["mean_pred"] - new["actual_rate"]) / new["actual_rate"]
        <= RULE["max_calibration_gap_rel"],
        "ece_at_most_half_of_old": new["ece_quantile10"] <= RULE["max_ece_ratio"] * old["ece_quantile10"],
        "auc_unchanged_within_0.001": abs(new["roc_auc"] - old["roc_auc"]) <= RULE["max_abs_auc_change"],
        "brier_not_worse": new["brier"] <= old["brier"],
        "ece_improves_in_every_repeat": bool(
            (np.array(repeat_ece[chosen]) < np.array(repeat_ece["none (raw model)"])).all()),
    }
    return {"checks": checks, "promote": all(checks.values())}


def plot_before_after(tables: dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=120)
    hi = 0
    style = {"none (raw model)": ("#C44E52", "o-"), "prior_shift": ("#DD8452", "s-"),
             "platt": ("#55A868", "^-"), "isotonic": ("#4C72B0", "d-")}
    for name, t in tables.items():
        c, m = style[name]
        ax.plot(t.mean_pred, t.actual, m, color=c, label=name, ms=4, lw=1.2)
        hi = max(hi, t.mean_pred.max(), t.actual.max())
    ax.plot([0, hi * 1.05], [0, hi * 1.05], "k--", lw=0.8, label="perfect")
    ax.set(xlabel="mean predicted churn", ylabel="observed churn rate", title=title)
    ax.legend(fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fmt_row(name, m):
    return (f"| {name} | {m['mean_pred']:.2%} | {m['actual_rate']:.2%} | {m['ece_quantile10']:.4f} | "
            f"{m['ece_uniform10']:.4f} | {m['brier']:.5f} | {m['log_loss']:.5f} | {m['roc_auc']:.4f} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", default=str(config.MARCH_FEATURES), help="labelled cohort (features + is_churn)")
    ap.add_argument("--cohort", default="march_2017")
    ap.add_argument("--alias", default=config.CHAMPION_ALIAS, help="model to recalibrate")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--earlier-labelled", default=str(config.TRAIN_FEATURES),
                    help="optional earlier cohort used only to show the calibrator does not transfer back")
    ap.add_argument("--out-dir", default=str(config.OUTPUTS / "recalibration"))
    ap.add_argument("--promote", action="store_true", help="move `champion` if the rule passes")
    args = ap.parse_args(argv)

    scorer, bundle_dir, parent = registry.load_scorer(alias=args.alias)
    profile = scorer.load_profile(bundle_dir)
    old_rate = scorer.metadata["training_base_rate"]
    df = load_features(args.labelled)
    validate_features(df, require_label=True, require_unique_ids=True)
    y = df[LABEL_COL].to_numpy().astype(int)
    raw = scorer.predict(df, validate=False)["raw_probability"].to_numpy()
    idx = np.arange(len(df))
    print(f"parent: v{parent.version}; cohort {args.cohort}: {len(df):,} users, churn {y.mean():.2%}, "
          f"raw mean prediction {raw.mean():.2%}, training base rate {old_rate:.2%}")

    # ---- primary split (seed 42) -------------------------------------------------
    ia, ib = train_test_split(idx, test_size=0.5, stratify=y, random_state=args.seed)
    cands = fit_candidates(raw[ia], y[ia], old_rate)
    results = evaluate(cands, raw[ib], y[ib])
    chosen = choose(results)

    # label-free diagnostic
    em_rate = estimate_prior_em(raw[ia], old_rate)

    # ---- repeated splits ---------------------------------------------------------
    rep = {k: {"ece": [], "mean_pred": [], "brier": [], "auc": []} for k in results}
    for s in range(N_REPEATS):
        a, b = train_test_split(idx, test_size=0.5, stratify=y, random_state=1000 + s)
        r = evaluate(fit_candidates(raw[a], y[a], old_rate), raw[b], y[b])
        for k, m in r.items():
            rep[k]["ece"].append(m["ece_quantile10"])
            rep[k]["mean_pred"].append(m["mean_pred"])
            rep[k]["brier"].append(m["brier"])
            rep[k]["auc"].append(m["roc_auc"])
    repeat_ece = {k: v["ece"] for k, v in rep.items()}

    decision = promotion_decision(results, chosen, len(ib), int(y[ib].sum()), repeat_ece)

    # ---- final calibrator: refit chosen recipe on all labelled users -------------
    final_cal = fit_candidates(raw, y, old_rate)[chosen]
    final_scorer = scorer.with_calibrator(final_cal, {
        "parent_version": str(parent.version), "calibration_fit_cohort": args.cohort,
        "calibration_method": chosen, "calibration_fit_rows": int(len(df)),
        "calibration_note": "validated on a held-out half, then refitted on the full cohort"})

    # ---- does the March calibrator transfer back to Feb? (limitation check) ------
    transfer = None
    if args.earlier_labelled and Path(args.earlier_labelled).exists():
        e = load_features(args.earlier_labelled)
        e_raw = scorer.predict(e, validate=False)["raw_probability"].to_numpy()
        e_cal = final_cal.transform(e_raw)
        transfer = {
            "earlier_actual_rate": float(e[LABEL_COL].mean()),
            "earlier_mean_pred_raw": float(e_raw.mean()),
            "earlier_mean_pred_recalibrated_on_march": float(e_cal.mean()),
        }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"none (raw model)": reliability_table(y[ib], raw[ib])}
    for name, cal in cands.items():
        tables[name] = reliability_table(y[ib], cal.transform(raw[ib]))
    plot_before_after(tables, out / "before_after_reliability.png",
                      f"Held-out half B of {args.cohort} (n={len(ib):,})")

    summary = {
        "cohort": args.cohort, "parent_version": str(parent.version), "n": int(len(df)),
        "n_fit_A": int(len(ia)), "n_eval_B": int(len(ib)),
        "training_base_rate": old_rate, "cohort_base_rate": float(y.mean()),
        "base_rate_estimate_from_A": float(y[ia].mean()),
        "label_free_em_estimate_of_base_rate": em_rate,
        "heldout_results_B": results, "chosen": chosen,
        "repeated_splits": {k: {m: [float(np.mean(v)), float(np.std(v))] for m, v in d.items()}
                            for k, d in rep.items()},
        "calibrators": {k: v.to_dict() if k != "isotonic" else {"kind": "isotonic", "n_knots": len(v.x)}
                        for k, v in cands.items()},
        "decision": decision, "rule": RULE, "admission": ADMISSION, "transfer_check": transfer,
    }
    (out / "recalibration.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---- register as a new version -----------------------------------------------
    client = registry.configure()
    with mlflow.start_run(run_name=f"recalibrate_{args.cohort}") as run:
        mlflow.log_params({"parent_version": parent.version, "cohort": args.cohort, "method": chosen,
                           "split_seed": args.seed, "fit_rows_heldout_protocol": len(ia),
                           "eval_rows": len(ib), "final_fit_rows": len(df),
                           "old_base_rate": round(old_rate, 5), "cohort_base_rate": round(float(y.mean()), 5)})
        for name, m in results.items():
            tag = name.split(" ")[0]
            mlflow.log_metrics({f"heldout_{tag}_{k}": v for k, v in m.items() if k != "n"})
        mlflow.log_metrics({"repeat_mean_ece_raw": float(np.mean(rep["none (raw model)"]["ece"])),
                            f"repeat_mean_ece_{chosen}": float(np.mean(rep[chosen]["ece"]))})
        mlflow.log_dict(summary, "recalibration.json")
        mlflow.log_artifact(str(out / "before_after_reliability.png"))
        with tempfile.TemporaryDirectory() as tmp:
            bundle = final_scorer.save(Path(tmp) / "bundle", profile)
            import numpy as _np, pandas as _pd, xgboost as _xgb  # noqa: E401
            info = registry.log_and_register(bundle, pip_requirements=[
                f"xgboost=={_xgb.__version__}", f"pandas=={_pd.__version__}", f"numpy=={_np.__version__}", "pyarrow"])
    new_v = info.registered_model_version
    for k, v in {"calibration": chosen, "parent_version": parent.version, "calibration_cohort": args.cohort,
                 "promotion_rule_passed": str(decision["promote"])}.items():
        client.set_model_version_tag(config.MODEL_NAME, new_v, k, str(v))
    registry.set_alias(client, config.CHALLENGER_ALIAS, new_v)
    summary["new_version"] = new_v
    promoted = False
    if args.promote and decision["promote"]:
        registry.promote(client, new_v)
        promoted = True
    summary["promoted"] = promoted
    (out / "recalibration.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---- console + markdown summary ---------------------------------------------
    hdr = ("| method | mean predicted | actual | ECE (equal-count) | ECE (equal-width) | Brier | log loss | ROC-AUC |\n"
           "|---|---|---|---|---|---|---|---|")
    md = [f"# Recalibration of v{parent.version} on {args.cohort}", "",
          f"Fit on random half A ({len(ia):,} users), evaluated on half B ({len(ib):,} users). "
          f"Training base rate {old_rate:.2%}; cohort base rate {y.mean():.2%}; estimate from A "
          f"{y[ia].mean():.2%}; label-free EM estimate {em_rate:.2%}.", "",
          "## Held-out half B (seed 42 split)", "", hdr]
    md += [fmt_row(k, v) for k, v in results.items()]
    md += ["", f"Chosen by admission rule + lowest held-out ECE: **{chosen}**. Registered as v{new_v}.", "",
           f"## Repeated splits (mean +/- sd over {N_REPEATS} random 50/50 splits, equal-count ECE)", "",
           "| method | ECE | mean predicted (actual %.2f%%) | Brier | ROC-AUC |" % (100 * y.mean()), "|---|---|---|---|---|"]
    for k, d in rep.items():
        md.append(f"| {k} | {np.mean(d['ece']):.4f} +/- {np.std(d['ece']):.4f} | "
                  f"{np.mean(d['mean_pred']):.2%} | {np.mean(d['brier']):.5f} | {np.mean(d['auc']):.4f} |")
    md += ["", "## Promotion rule", ""]
    md += [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in decision["checks"].items()]
    md += ["", f"Decision: **{'promote' if decision['promote'] else 'do not promote'}**"
           f"{' (champion moved to v' + str(new_v) + ')' if promoted else ''}."]
    if transfer:
        md += ["", "## Transfer check (limitation)", "",
               f"Applying the March-fitted calibrator to the Feb cohort: actual {transfer['earlier_actual_rate']:.2%}, "
               f"raw model {transfer['earlier_mean_pred_raw']:.2%}, recalibrated {transfer['earlier_mean_pred_recalibrated_on_march']:.2%}. "
               "A calibrator encodes the base rate of the cohort it was fitted on."]
    md += ["", "![before/after](before_after_reliability.png)"]
    (out / "recalibration.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nnew version v{new_v} (alias challenger{'; promoted to champion' if promoted else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
