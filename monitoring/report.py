"""Render the monitoring result as Markdown (+ PNG charts) and JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BADGE = {"OK": "OK", "WARN": "WARN", "ALERT": "ALERT", "NOT_EVALUATED": "n/a"}


def _pct(x, d=2):
    return f"{x:.{d}%}"


def plot_reliability(perf: dict, out: Path, title: str) -> None:
    t = pd.DataFrame(perf["reliability"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=120)
    hi = max(t.mean_pred.max(), t.actual.max()) * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, label="perfect calibration")
    ax.plot(t.mean_pred, t.actual, "o-", color="#C44E52", label="10 equal-count bins")
    ax.set(xlabel="mean predicted churn", ylabel="observed churn rate", title=title)
    ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_psi(fd: list[dict], out: Path, top: int = 15) -> None:
    t = pd.DataFrame(fd).head(top).iloc[::-1]
    colors = t.psi.map(lambda v: "#C44E52" if v >= 0.25 else "#DD8452" if v >= 0.10 else "#4C72B0")
    fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=120)
    ax.barh(t.feature, t.psi, color=colors)
    ax.axvline(0.10, color="grey", ls=":", lw=1)
    ax.axvline(0.25, color="grey", ls="--", lw=1)
    ax.set(xlabel="PSI vs reference (dotted 0.10 warn, dashed 0.25 alert)", title="Feature drift, top 15")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def to_markdown(res: dict, meta: dict) -> str:
    L = []
    a = L.append
    a(f"# Monitoring report: {meta.get('batch_name', 'batch')}")
    a("")
    a(f"**Overall status: {res['overall_status']}**")
    a("")
    a(f"- Model: `{meta.get('model_name')}` v{meta.get('model_version')} "
      f"(alias `{meta.get('alias')}`, calibration `{meta.get('calibration')}`)")
    a(f"- Batch: `{meta.get('features_file')}`, {res['n_rows']:,} subscribers")
    a(f"- Reference: training cohort `{meta.get('training_data')}` (frozen profile stored with the model)")
    a("")
    a("| Component | Status |")
    a("|---|---|")
    for k, v in res["components"].items():
        a(f"| {k.replace('_', ' ')} | {BADGE[v]} |")
    th = res["thresholds"]
    a("")
    a(f"Thresholds: PSI warn {th['psi']['warn']} / alert {th['psi']['alert']}; "
      f"calibration gap warn {th['calibration_in_the_large_rel']['warn']:.0%} / alert "
      f"{th['calibration_in_the_large_rel']['alert']:.0%} of the actual rate; "
      f"ECE/actual warn {th['ece_rel']['warn']:.0%} / alert {th['ece_rel']['alert']:.0%}; "
      f"AUC drop warn {th['auc_drop']['warn']} / alert {th['auc_drop']['alert']}.")
    a("")

    pdr = res["prediction_drift"]
    a("## Prediction drift")
    a("")
    a(f"- Score PSI vs reference: **{pdr['psi']:.3f}** ({pdr['status']})")
    a(f"- Mean raw score: reference {_pct(pdr['ref_mean_score'])} -> batch {_pct(pdr['cur_mean_score'])} "
      f"({pdr['mean_score_change_rel']:+.1%})")
    a("")

    perf = res["performance"]
    a("## Calibration and performance (labels available)")
    a("")
    if perf is None:
        a("Labels not supplied: not evaluated. Re-run with labels once the churn window has closed.")
    else:
        m, r, d = perf["metrics"], perf["reference_metrics"], perf["derived"]
        a("| | Reference (Feb held-out) | This batch |")
        a("|---|---|---|")
        a(f"| actual churn rate | {_pct(r['actual_rate'])} | {_pct(m['actual_rate'])} |")
        a(f"| mean predicted churn | {_pct(r['mean_pred'])} | {_pct(m['mean_pred'])} |")
        a(f"| predicted / actual | {r['mean_pred_over_actual']:.3f} | {m['mean_pred_over_actual']:.3f} |")
        a(f"| ROC-AUC | {r['roc_auc']:.4f} | {m['roc_auc']:.4f} |")
        a(f"| PR-AUC | {r['pr_auc']:.4f} | {m['pr_auc']:.4f} |")
        a(f"| Brier | {r['brier']:.5f} | {m['brier']:.5f} |")
        a(f"| ECE (10 equal-count bins) | {r['ece_quantile10']:.4f} | {m['ece_quantile10']:.4f} |")
        a(f"| ECE (10 equal-width bins) | {r['ece_uniform10']:.4f} | {m['ece_uniform10']:.4f} |")
        a("")
        s = perf["statuses"]
        a(f"- Calibration in the large: gap {d['calibration_gap_rel']:.1%} of actual -> **{s['calibration_in_the_large']}**")
        a(f"- ECE relative to actual rate: {d['ece_rel']:.1%} -> **{s['ece']}**")
        a(f"- AUC change vs reference: {-d['auc_drop']:+.4f} -> **{s['auc']}**")
        a("")
        a("Reliability table (equal-count bins, lowest to highest score):")
        a("")
        a("| bin | n | mean predicted | actual | gap |")
        a("|---|---|---|---|---|")
        for row in perf["reliability"]:
            a(f"| {int(row['bin']) + 1} | {int(row['n']):,} | {_pct(row['mean_pred'])} | "
              f"{_pct(row['actual'])} | {row['gap']:+.2%} |")
        a("")
        a("![reliability](reliability.png)")
    a("")

    fd = res["feature_drift"]
    a("## Feature drift (ranked by PSI)")
    a("")
    a(f"{fd['n_alert']} feature(s) at ALERT, {fd['n_warn']} at WARN, of {len(fd['all'])}.")
    a("")
    a("| # | feature | PSI | status | model importance | null rate ref -> batch | mean ref -> batch |")
    a("|---|---|---|---|---|---|---|")
    for i, r in enumerate(fd["all"], 1):
        if r["psi"] < 0.02 and i > 10:
            continue
        mean = (f"{r['ref_mean']:.3g} -> {r['cur_mean']:.3g}"
                if r.get("cur_mean") is not None and r.get("ref_mean") is not None else "")
        a(f"| {i} | {r['feature']} | {r['psi']:.3f} | {r['status']} | {r['importance_share']:.1%} | "
          f"{r['ref_null_rate']:.1%} -> {r['cur_null_rate']:.1%} | {mean} |")
    a("")
    a("![psi](psi.png)")
    a("")
    unseen = [r for r in fd["all"] if r.get("unseen_share") and r["unseen_share"] > 0]
    if unseen:
        a("Categorical codes unseen in training: " +
          ", ".join(f"{r['feature']} {r['unseen_share']:.3%}" for r in unseen))
        a("")
    return "\n".join(L)


def write_report(res: dict, meta: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"meta": meta, **res}, indent=1, default=float))
    plot_psi(res["feature_drift"]["all"], out / "psi.png")
    if res["performance"]:
        plot_reliability(res["performance"], out / "reliability.png",
                         f"Reliability, {meta.get('batch_name', 'batch')}")
    (out / "report.md").write_text(to_markdown(res, meta))
    return out
