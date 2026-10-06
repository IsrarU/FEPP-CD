#!/usr/bin/env python3
"""Recompute every table/number of the paper from the released prediction files.

Usage:  python scripts/evaluate.py [--results results] [--bootstrap 2000]

Writes:
  results/tables/*.csv          one CSV per paper table
  results/paper_numbers.json    flat dictionary of all reported values
  results/RESULTS.md            human-readable summary
No GPU, model weights or raw datasets are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fepp_cd.comm import comm_cost  # noqa: E402
from fepp_cd.metrics import (  # noqa: E402
    bootstrap_ci, calibrated_operating_point, community_sensitivity,
    compute_metrics, expected_calibration_error, load_predictions, mcnemar,
)

PRED = {
    "kaggle_iid": "predictions/kaggle/iid.csv",
    "kaggle_noniid": "predictions/kaggle/noniid.csv",
    "kaggle_flip_010": "predictions/kaggle/flip_010.csv",
    "kaggle_flip_020": "predictions/kaggle/flip_020.csv",
    "kaggle_flip_030": "predictions/kaggle/flip_030.csv",
    "jigsaw_iid": "predictions/jigsaw/iid.csv",
    "jigsaw_noniid": "predictions/jigsaw/noniid.csv",
    "hatexplain": "predictions/hatexplain/predictions.csv",
}
SUPPLEMENTARY = {"kaggle_centralized": "predictions/kaggle/centralized_supplementary.csv"}


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    res = Path(args.results)
    tab = res / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    P = {k: load_predictions(res / v) for k, v in PRED.items()}
    for k, v in SUPPLEMENTARY.items():
        if (res / v).exists():
            P[k] = load_predictions(res / v)
    # Kaggle runs share one frozen test set (same order)
    yk = P["kaggle_iid"].y_true.values
    for k in [k for k in P if k.startswith("kaggle_")]:
        assert (P[k].y_true.values == yk).all(), f"{k}: test-set order differs"

    M = {k: compute_metrics(d.y_true, d.y_pred, d.y_prob) for k, d in P.items()}
    num = {}
    for k, m in M.items():
        for f, v in m.to_dict().items():
            num[f"{k}.{f}"] = v if isinstance(v, int) else r4(v)

    # ---- bootstrap CIs ------------------------------------------------------
    ci_rows = []
    for k in ["kaggle_iid", "kaggle_noniid", "jigsaw_iid", "jigsaw_noniid", "hatexplain"]:
        d = P[k]
        ci = bootstrap_ci(d.y_true.values, d.y_pred.values, d.y_prob.values,
                          B=args.bootstrap, seed=args.seed)
        for metric, (lo, hi) in ci.items():
            num[f"{k}.ci.{metric}"] = [r4(lo), r4(hi)]
            ci_rows.append({"run": k, "metric": metric, "point": r4(getattr(M[k], metric)),
                            "ci_low": r4(lo), "ci_high": r4(hi)})
    pd.DataFrame(ci_rows).to_csv(tab / "bootstrap_ci.csv", index=False)

    # ---- Table: Kaggle IID classification report ---------------------------
    m = M["kaggle_iid"]
    pd.DataFrame([
        {"class": "Non-harmful (0)", "precision": m.precision_0, "recall": m.recall_0, "f1": m.f1_0, "support": m.TN + m.FP},
        {"class": "Harmful (1)", "precision": m.precision_1, "recall": m.recall_1, "f1": m.f1_1, "support": m.TP + m.FN},
        {"class": "Macro avg.", "precision": (m.precision_0 + m.precision_1) / 2,
         "recall": (m.recall_0 + m.recall_1) / 2, "f1": m.macro_f1, "support": m.n},
    ]).round(4).to_csv(tab / "table_kaggle_iid.csv", index=False)

    # ---- Table: IID vs non-IID ----------------------------------------------
    a, b = M["kaggle_iid"], M["kaggle_noniid"]
    rows = [("Accuracy", a.accuracy, b.accuracy), ("Macro F1", a.macro_f1, b.macro_f1),
            ("ROC-AUC", a.roc_auc, b.roc_auc), ("AUPRC", a.auprc, b.auprc),
            ("Recall (harmful)", a.recall_1, b.recall_1), ("FN", a.FN, b.FN), ("FP", a.FP, b.FP),
            ("TMR (%)", 100 * a.TMR, 100 * b.TMR), ("FAR (%)", 100 * a.FAR, 100 * b.FAR)]
    pd.DataFrame([{"metric": r[0], "iid": r[1], "noniid": r[2], "change": r[2] - r[1]} for r in rows]) \
        .round(4).to_csv(tab / "table_iid_vs_noniid.csv", index=False)
    num["kaggle.noniid_minus_iid.TMR_pp"] = r4(100 * (b.TMR - a.TMR))
    num["kaggle.noniid_minus_iid.FAR_pp"] = r4(100 * (b.FAR - a.FAR))
    mc = mcnemar(yk, P["kaggle_iid"].y_pred.values, P["kaggle_noniid"].y_pred.values)
    num.update({f"kaggle.mcnemar.{k}": (r4(v) if isinstance(v, float) else v) for k, v in mc.items()})

    # ---- Table: calibrated operating points --------------------------------
    rows = []
    for name in ["kaggle_iid", "kaggle_noniid"]:
        d = P[name]
        t05 = compute_metrics(d.y_true, (d.y_prob >= 0.5).astype(int), d.y_prob)
        rows.append({"operating_point": "tau=0.5", "model": name, "tau": 0.5,
                     "TMR_pct": 100 * t05.TMR, "FAR_pct": 100 * t05.FAR})
        for t in (0.02, 0.05):
            op = calibrated_operating_point(d.y_true.values, d.y_prob.values, t)
            rows.append({"operating_point": f"TMR<={int(t * 100)}%", "model": name, "tau": op["tau"],
                         "TMR_pct": 100 * op["TMR"], "FAR_pct": 100 * op["FAR"]})
            num[f"{name}.calib.tmr{int(t * 100)}.FAR_pct"] = r4(100 * op["FAR"])
            num[f"{name}.calib.tmr{int(t * 100)}.TMR_pct"] = r4(100 * op["TMR"])
            num[f"{name}.calib.tmr{int(t * 100)}.tau"] = float(op["tau"])
    pd.DataFrame(rows).round(6).to_csv(tab / "table_calibrated_operating_points.csv", index=False)

    # ---- Table: ECE ------------------------------------------------------------
    rows = []
    for name in P:
        e = expected_calibration_error(P[name].y_true.values, P[name].y_prob.values, 10)
        num[f"{name}.ece"] = r4(e)
        rows.append({"run": name, "ece_10bins": e})
    pd.DataFrame(rows).round(4).to_csv(tab / "table_ece.csv", index=False)

    # ---- Jigsaw partition & convergence (copied logs) -------------------------
    jp = pd.read_csv(res / "logs/jigsaw_noniid_client_distribution.csv")
    jp.to_csv(tab / "table_jigsaw_partition.csv", index=False)
    num["jigsaw.dominant_clients_share_pct"] = r4(jp.loc[jp.client.isin([0, 3]), "share_pct"].sum())
    pd.read_csv(res / "logs/jigsaw_noniid_round_log.csv").round(4).to_csv(tab / "table_jigsaw_convergence.csv", index=False)
    pd.read_csv(res / "logs/kaggle_noniid_client_distribution.csv").to_csv(tab / "table_kaggle_partition.csv", index=False)

    # ---- SecAgg (utility identical by construction) + communication --------
    cc = comm_cost(10, 5, 66_000_000, 4, 0.25).iloc[-1]
    num["comm.fedavg_gb"] = r4(cc.fedavg_gb)
    num["comm.secagg_gb"] = r4(cc.secagg_gb)
    pd.DataFrame([
        {"method": "FedAvg", "accuracy": a.accuracy, "macro_f1": a.macro_f1, "comm_gb": cc.fedavg_gb},
        {"method": "FedAvg + SecAgg (Prop. 1)", "accuracy": a.accuracy, "macro_f1": a.macro_f1, "comm_gb": cc.secagg_gb},
    ]).round(4).to_csv(tab / "table_secagg.csv", index=False)

    # ---- Label flipping --------------------------------------------------------
    rows = []
    for g, name in [(0, "kaggle_iid"), (10, "kaggle_flip_010"), (20, "kaggle_flip_020"), (30, "kaggle_flip_030")]:
        x = M[name]
        rows.append({"flip_pct": g, "accuracy": x.accuracy, "macro_f1": x.macro_f1, "roc_auc": x.roc_auc,
                     "auprc": x.auprc, "FN": x.FN, "FP": x.FP, "TMR_pct": 100 * x.TMR, "FAR_pct": 100 * x.FAR})
    pd.DataFrame(rows).round(4).to_csv(tab / "table_label_flipping.csv", index=False)
    f30 = M["kaggle_flip_030"]
    num["flip30.acc_drop_pp"] = r4(100 * (a.accuracy - f30.accuracy))
    num["flip30.tmr_relative_increase_pct"] = r4(100 * (f30.TMR / a.TMR - 1))
    num["flip30.auc_drop"] = r4(a.roc_auc - f30.roc_auc)

    # ---- HateXplain: rationale alignment + community sensitivity ------------
    ra = pd.read_csv(res / "xai/rationale_alignment.csv")
    for c in ("precision", "recall", "f1"):
        num[f"hatexplain.rationale.{c}"] = r4(ra[c].mean())
    num["hatexplain.rationale.n"] = int(len(ra))
    comm_df = community_sensitivity(P["hatexplain"], top_k=12)
    comm_df.round(4).to_csv(tab / "table_community_sensitivity.csv", index=False)
    num["hatexplain.community.macro_f1_min"] = r4(comm_df.macro_f1.min())
    num["hatexplain.community.macro_f1_max"] = r4(comm_df.macro_f1.max())
    num["hatexplain.community.macro_f1_mean"] = r4(comm_df.macro_f1.mean())
    num["hatexplain.community.macro_f1_range"] = r4(comm_df.macro_f1.max() - comm_df.macro_f1.min())

    # ---- Cross-dataset summary -------------------------------------------------
    rows = []
    for label, name in [("Kaggle CB | IID FL", "kaggle_iid"), ("Kaggle CB | IID FL + SecAgg", "kaggle_iid"),
                        ("Kaggle CB | Non-IID FL", "kaggle_noniid"), ("Kaggle CB | Flip 30%", "kaggle_flip_030"),
                        ("Jigsaw | Non-IID FL", "jigsaw_noniid"), ("Jigsaw | IID FL", "jigsaw_iid"),
                        ("HateXplain | IID FL (external)", "hatexplain")]:
        x = M[name]
        rows.append({"dataset_condition": label, "n": x.n, "accuracy": x.accuracy, "macro_f1": x.macro_f1,
                     "roc_auc": x.roc_auc, "auprc": x.auprc, "TMR_pct": 100 * x.TMR, "FAR_pct": 100 * x.FAR})
    pd.DataFrame(rows).round(4).to_csv(tab / "table_cross_dataset_summary.csv", index=False)
    num["macro_f1_drop.kaggle"] = r4(M["kaggle_iid"].macro_f1 - M["kaggle_noniid"].macro_f1)
    num["macro_f1_drop.jigsaw"] = r4(M["jigsaw_iid"].macro_f1 - M["jigsaw_noniid"].macro_f1)

    if "kaggle_centralized" in M:
        x = M["kaggle_centralized"]
        pd.DataFrame([{"run": "Kaggle centralized (supplementary, not in paper)", **{k: r4(v) if isinstance(v, float) else v
                       for k, v in x.to_dict().items()}}]).to_csv(tab / "supplementary_centralized_baseline.csv", index=False)

    (res / "paper_numbers.json").write_text(json.dumps(num, indent=1, sort_keys=True))
    write_markdown(res, tab)
    print(f"Wrote {len(num)} values to {res / 'paper_numbers.json'} and tables to {tab}")


def write_markdown(res: Path, tab: Path) -> None:
    parts = ["# Recomputed results\n",
             "Generated by `scripts/evaluate.py` from the released prediction files. "
             "Do not edit by hand.\n"]
    for f in sorted(tab.glob("*.csv")):
        df = pd.read_csv(f)
        parts.append(f"\n## {f.stem}\n\n" + df.to_markdown(index=False) + "\n")
    (res / "RESULTS.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
