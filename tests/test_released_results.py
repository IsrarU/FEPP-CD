"""Spot checks of headline numbers against the released prediction files."""
from pathlib import Path

import pytest

from fepp_cd.metrics import compute_metrics, load_predictions

R = Path(__file__).resolve().parents[1] / "results" / "predictions"

CASES = [
    ("kaggle/iid.csv", 0.9286, 0.8500, 0.9605, 273, 354),
    ("kaggle/noniid.csv", 0.9245, 0.8233, 0.9592, 146, 517),
    ("kaggle/flip_030.csv", 0.9205, 0.8416, 0.9505, 390, 309),
    ("jigsaw/iid.csv", 0.9625, 0.9010, 0.9799, 462, 734),
    ("jigsaw/noniid.csv", 0.9606, 0.8732, 0.9706, 1163, 94),
    ("hatexplain/predictions.csv", 0.7607, 0.7497, 0.8283, 443, 521),
]


@pytest.mark.parametrize("rel,acc,mf1,auc,fn,fp", CASES)
def test_headline(rel, acc, mf1, auc, fn, fp):
    d = load_predictions(R / rel)
    m = compute_metrics(d.y_true, d.y_pred, d.y_prob)
    assert round(m.accuracy, 4) == acc and round(m.macro_f1, 4) == mf1
    assert round(m.roc_auc, 4) == auc and (m.FN, m.FP) == (fn, fp)
    assert ((d.y_prob >= 0.5).astype(int) == d.y_pred).all()
