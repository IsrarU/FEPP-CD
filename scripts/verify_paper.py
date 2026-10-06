#!/usr/bin/env python3
"""Check that the numbers printed in the paper match the recomputed values.

Run ``python scripts/evaluate.py`` first; this script reads
``results/paper_numbers.json`` and compares it with the values as they are
printed in the manuscript (rounded the same way). Exit code 1 on any mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (claim in paper, key in paper_numbers.json, printed value, scale, decimals)
# scale converts the stored value to the printed unit (e.g. 100 for %).
CLAIMS = [
    # Kaggle IID (Table: Kaggle IID; abstract)
    ("Kaggle IID accuracy", "kaggle_iid.accuracy", 0.9286, 1, 4),
    ("Kaggle IID macro F1", "kaggle_iid.macro_f1", 0.8500, 1, 4),
    ("Kaggle IID ROC-AUC", "kaggle_iid.roc_auc", 0.9605, 1, 4),
    ("Kaggle IID AUPRC", "kaggle_iid.auprc", 0.9928, 1, 4),
    ("Kaggle IID TMR %", "kaggle_iid.TMR", 3.62, 100, 2),
    ("Kaggle IID FAR %", "kaggle_iid.FAR", 28.25, 100, 2),
    ("Kaggle IID FN", "kaggle_iid.FN", 273, 1, 0),
    ("Kaggle IID FP", "kaggle_iid.FP", 354, 1, 0),
    ("Kaggle IID precision 0", "kaggle_iid.precision_0", 0.7671, 1, 4),
    ("Kaggle IID recall 0", "kaggle_iid.recall_0", 0.7175, 1, 4),
    ("Kaggle IID F1 0", "kaggle_iid.f1_0", 0.7414, 1, 4),
    ("Kaggle IID precision 1", "kaggle_iid.precision_1", 0.9535, 1, 4),
    ("Kaggle IID recall 1", "kaggle_iid.recall_1", 0.9638, 1, 4),
    ("Kaggle IID F1 1", "kaggle_iid.f1_1", 0.9586, 1, 4),
    # Kaggle non-IID
    ("Kaggle non-IID accuracy", "kaggle_noniid.accuracy", 0.9245, 1, 4),
    ("Kaggle non-IID macro F1", "kaggle_noniid.macro_f1", 0.8233, 1, 4),
    ("Kaggle non-IID ROC-AUC", "kaggle_noniid.roc_auc", 0.9592, 1, 4),
    ("Kaggle non-IID AUPRC", "kaggle_noniid.auprc", 0.9925, 1, 4),
    ("Kaggle non-IID recall 1", "kaggle_noniid.recall_1", 0.9806, 1, 4),
    ("Kaggle non-IID TMR %", "kaggle_noniid.TMR", 1.94, 100, 2),
    ("Kaggle non-IID FAR %", "kaggle_noniid.FAR", 41.26, 100, 2),
    ("Kaggle non-IID FN", "kaggle_noniid.FN", 146, 1, 0),
    ("Kaggle non-IID FP", "kaggle_noniid.FP", 517, 1, 0),
    ("TMR change (pp)", "kaggle.noniid_minus_iid.TMR_pp", -1.69, 1, 2),
    ("FAR change (pp)", "kaggle.noniid_minus_iid.FAR_pp", 13.01, 1, 2),
    ("McNemar b", "kaggle.mcnemar.b", 242, 1, 0),
    ("McNemar c", "kaggle.mcnemar.c", 206, 1, 0),
    ("McNemar chi2", "kaggle.mcnemar.chi2", 2.734, 1, 3),
    ("McNemar p", "kaggle.mcnemar.p_value", 0.0982, 1, 4),
    # Calibrated operating points
    ("IID FAR @ TMR<=2%", "kaggle_iid.calib.tmr2.FAR_pct", 39.82, 1, 2),
    ("non-IID FAR @ TMR<=2%", "kaggle_noniid.calib.tmr2.FAR_pct", 40.78, 1, 2),
    ("IID FAR @ TMR<=5%", "kaggle_iid.calib.tmr5.FAR_pct", 22.11, 1, 2),
    ("non-IID FAR @ TMR<=5%", "kaggle_noniid.calib.tmr5.FAR_pct", 23.62, 1, 2),
    ("matched TMR (2%)", "kaggle_iid.calib.tmr2.TMR_pct", 1.99, 1, 2),
    ("matched TMR (5%)", "kaggle_iid.calib.tmr5.TMR_pct", 4.99, 1, 2),
    # ECE
    ("ECE Kaggle IID", "kaggle_iid.ece", 0.069, 1, 3),
    ("ECE Jigsaw non-IID", "jigsaw_noniid.ece", 0.038, 1, 3),
    ("ECE HateXplain", "hatexplain.ece", 0.219, 1, 3),
    # Jigsaw
    ("Jigsaw IID accuracy", "jigsaw_iid.accuracy", 0.9625, 1, 4),
    ("Jigsaw IID macro F1", "jigsaw_iid.macro_f1", 0.9010, 1, 4),
    ("Jigsaw IID ROC-AUC", "jigsaw_iid.roc_auc", 0.9799, 1, 4),
    ("Jigsaw IID AUPRC", "jigsaw_iid.auprc", 0.9107, 1, 4),
    ("Jigsaw IID TN", "jigsaw_iid.TN", 27906, 1, 0),
    ("Jigsaw IID FP", "jigsaw_iid.FP", 734, 1, 0),
    ("Jigsaw IID FN", "jigsaw_iid.FN", 462, 1, 0),
    ("Jigsaw IID TP", "jigsaw_iid.TP", 2780, 1, 0),
    ("Jigsaw IID TMR %", "jigsaw_iid.TMR", 14.25, 100, 2),
    ("Jigsaw IID FAR %", "jigsaw_iid.FAR", 2.56, 100, 2),
    ("Jigsaw non-IID accuracy", "jigsaw_noniid.accuracy", 0.9606, 1, 4),
    ("Jigsaw non-IID macro F1", "jigsaw_noniid.macro_f1", 0.8732, 1, 4),
    ("Jigsaw non-IID ROC-AUC", "jigsaw_noniid.roc_auc", 0.9706, 1, 4),
    ("Jigsaw non-IID AUPRC", "jigsaw_noniid.auprc", 0.8925, 1, 4),
    ("Jigsaw non-IID TMR %", "jigsaw_noniid.TMR", 35.87, 100, 2),
    ("Jigsaw non-IID FAR %", "jigsaw_noniid.FAR", 0.33, 100, 2),
    ("Jigsaw non-IID TN", "jigsaw_noniid.TN", 28546, 1, 0),
    ("Jigsaw non-IID FP", "jigsaw_noniid.FP", 94, 1, 0),
    ("Jigsaw non-IID FN", "jigsaw_noniid.FN", 1163, 1, 0),
    ("Jigsaw non-IID TP", "jigsaw_noniid.TP", 2079, 1, 0),
    ("Jigsaw dominant-client share %", "jigsaw.dominant_clients_share_pct", 77, 1, 0),
    # Communication
    ("FedAvg GB", "comm.fedavg_gb", 13.2, 1, 1),
    ("SecAgg GB", "comm.secagg_gb", 16.5, 1, 1),
    # Label flipping
    ("Flip10 accuracy", "kaggle_flip_010.accuracy", 0.9247, 1, 4),
    ("Flip20 accuracy", "kaggle_flip_020.accuracy", 0.9211, 1, 4),
    ("Flip30 accuracy", "kaggle_flip_030.accuracy", 0.9205, 1, 4),
    ("Flip10 macro F1", "kaggle_flip_010.macro_f1", 0.8378, 1, 4),
    ("Flip20 macro F1", "kaggle_flip_020.macro_f1", 0.8401, 1, 4),
    ("Flip30 macro F1", "kaggle_flip_030.macro_f1", 0.8416, 1, 4),
    ("Flip10 ROC-AUC", "kaggle_flip_010.roc_auc", 0.9552, 1, 4),
    ("Flip20 ROC-AUC", "kaggle_flip_020.roc_auc", 0.9520, 1, 4),
    ("Flip30 ROC-AUC", "kaggle_flip_030.roc_auc", 0.9505, 1, 4),
    ("Flip10 AUPRC", "kaggle_flip_010.auprc", 0.9917, 1, 4),
    ("Flip20 AUPRC", "kaggle_flip_020.auprc", 0.9912, 1, 4),
    ("Flip30 AUPRC", "kaggle_flip_030.auprc", 0.9908, 1, 4),
    ("Flip10 TMR %", "kaggle_flip_010.TMR", 3.40, 100, 2),
    ("Flip20 TMR %", "kaggle_flip_020.TMR", 4.77, 100, 2),
    ("Flip30 TMR %", "kaggle_flip_030.TMR", 5.18, 100, 2),
    ("Flip10 FAR %", "kaggle_flip_010.FAR", 32.40, 100, 2),
    ("Flip20 FAR %", "kaggle_flip_020.FAR", 26.66, 100, 2),
    ("Flip30 FAR %", "kaggle_flip_030.FAR", 24.66, 100, 2),
    ("Flip30 accuracy drop (pp)", "flip30.acc_drop_pp", 0.82, 1, 2),
    ("Flip30 relative TMR increase %", "flip30.tmr_relative_increase_pct", 42.9, 1, 1),
    # HateXplain
    ("HateXplain accuracy", "hatexplain.accuracy", 0.7607, 1, 4),
    ("HateXplain macro F1", "hatexplain.macro_f1", 0.7497, 1, 4),
    ("HateXplain ROC-AUC", "hatexplain.roc_auc", 0.8283, 1, 4),
    ("HateXplain AUPRC", "hatexplain.auprc", 0.8583, 1, 4),
    ("HateXplain TMR %", "hatexplain.TMR", 18.47, 100, 2),
    ("HateXplain FAR %", "hatexplain.FAR", 31.96, 100, 2),
    ("HateXplain TN", "hatexplain.TN", 1109, 1, 0),
    ("HateXplain FP", "hatexplain.FP", 521, 1, 0),
    ("HateXplain FN", "hatexplain.FN", 443, 1, 0),
    ("HateXplain TP", "hatexplain.TP", 1956, 1, 0),
    ("Rationale precision", "hatexplain.rationale.precision", 0.5681, 1, 4),
    ("Rationale recall", "hatexplain.rationale.recall", 0.5686, 1, 4),
    ("Rationale F1", "hatexplain.rationale.f1", 0.5683, 1, 4),
    ("Community macro F1 min", "hatexplain.community.macro_f1_min", 0.5485, 1, 4),
    ("Community macro F1 max", "hatexplain.community.macro_f1_max", 0.7316, 1, 4),
    ("Community macro F1 mean", "hatexplain.community.macro_f1_mean", 0.6686, 1, 4),
    ("Community macro F1 range", "hatexplain.community.macro_f1_range", 0.18, 1, 2),
    # Non-IID macro-F1 drop
    ("Kaggle macro-F1 drop", "macro_f1_drop.kaggle", 0.0268, 1, 4),
    ("Jigsaw macro-F1 drop", "macro_f1_drop.jigsaw", 0.0278, 1, 4),
]

CI_CLAIMS = [
    # (run, metric, printed [lo, hi], scale, decimals)
    ("kaggle_iid", "accuracy", (0.9235, 0.9343), 1, 4),
    ("kaggle_iid", "macro_f1", (0.8392, 0.8615), 1, 4),
    ("kaggle_iid", "roc_auc", (0.9560, 0.9649), 1, 4),
    ("kaggle_iid", "auprc", (0.9917, 0.9938), 1, 4),
    ("kaggle_iid", "TMR", (3.18, 4.05), 100, 2),
    ("kaggle_iid", "FAR", (25.73, 30.75), 100, 2),
    ("kaggle_noniid", "roc_auc", (0.9544, 0.9634), 1, 4),
    ("kaggle_noniid", "TMR", (1.63, 2.25), 100, 2),
    ("kaggle_noniid", "FAR", (38.60, 43.90), 100, 2),
    ("jigsaw_noniid", "accuracy", (0.9584, 0.9627), 1, 4),
    ("jigsaw_noniid", "TMR", (34.18, 37.50), 100, 2),
    ("jigsaw_noniid", "FAR", (0.26, 0.40), 100, 2),
    ("hatexplain", "accuracy", (0.7476, 0.7744), 1, 4),
    ("hatexplain", "macro_f1", (0.7355, 0.7638), 1, 4),
    ("hatexplain", "roc_auc", (0.8155, 0.8417), 1, 4),
    ("hatexplain", "auprc", (0.8424, 0.8739), 1, 4),
    ("hatexplain", "TMR", (16.85, 20.02), 100, 2),
    ("hatexplain", "FAR", (29.68, 34.22), 100, 2),
]


def main() -> int:
    path = ROOT / "results" / "paper_numbers.json"
    if not path.exists():
        print("results/paper_numbers.json not found - run scripts/evaluate.py first")
        return 2
    num = json.loads(path.read_text())
    bad = 0
    for name, key, printed, scale, dec in CLAIMS:
        got = round(num[key] * scale, dec)
        ok = abs(got - printed) < 10 ** (-dec) / 2 + 1e-9
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] {name:34s} paper={printed:<10} recomputed={got}")
    for run, metric, (lo, hi), scale, dec in CI_CLAIMS:
        glo, ghi = (round(v * scale, dec) for v in num[f"{run}.ci.{metric}"])
        ok = abs(glo - lo) < 1e-9 + 10 ** (-dec) / 2 and abs(ghi - hi) < 1e-9 + 10 ** (-dec) / 2
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] CI {run}.{metric:9s}      paper=[{lo}, {hi}] recomputed=[{glo}, {ghi}]")
    total = len(CLAIMS) + len(CI_CLAIMS)
    print(f"\n{total - bad}/{total} printed values reproduced.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
