"""Evaluation metrics used in the paper.

Every number in the paper's results section is produced by the functions in
this module from the released per-example prediction files
(``y_true``, ``y_pred``, ``y_prob``). Positive class (1) = harmful.

Definitions follow Section 3.4 of the paper:

* TMR (toxic miss rate)   = FN / (TP + FN)
* FAR (false alarm rate)  = FP / (TN + FP)
* Calibrated operating point  tau*(t) = max{tau : TMR(tau) <= t}
* ECE over M equal-width bins
* Percentile bootstrap CIs (B = 2000 resamples, seed 42 per prediction file)
* Continuity-corrected McNemar test for paired comparisons
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

BOOTSTRAP_SEED = 42
BOOTSTRAP_B = 2000


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------
@dataclass
class MetricSet:
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float
    auprc: float
    precision_0: float
    recall_0: float
    f1_0: float
    precision_1: float
    recall_1: float
    f1_1: float
    TN: int
    FP: int
    FN: int
    TP: int
    TMR: float
    FAR: float

    def to_dict(self) -> Dict:
        return asdict(self)


def counts(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def tmr_far(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = counts(y_true, y_pred)
    tmr = fn / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tn + fp) if (tn + fp) else float("nan")
    return tmr, far


def compute_metrics(y_true, y_pred, y_prob) -> MetricSet:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = counts(y_true, y_pred)
    tmr, far = tmr_far(y_true, y_pred)
    return MetricSet(
        n=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        weighted_f1=f1_score(y_true, y_pred, average="weighted"),
        roc_auc=roc_auc_score(y_true, y_prob),
        auprc=average_precision_score(y_true, y_prob),
        precision_0=p[0], recall_0=r[0], f1_0=f[0],
        precision_1=p[1], recall_1=r[1], f1_1=f[1],
        TN=tn, FP=fp, FN=fn, TP=tp, TMR=tmr, FAR=far,
    )


def safety_utility_score(acc: float, tmr: float, far: float,
                         lambda1: float = 1.0, lambda2: float = 1.0) -> float:
    """SUS = Acc - lambda1 * TMR - lambda2 * FAR (Eq. 'safety' in the paper)."""
    return acc - lambda1 * tmr - lambda2 * far


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, y_prob, B: int = BOOTSTRAP_B,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> Dict[str, tuple]:
    """Percentile bootstrap CIs over the frozen test set.

    A fresh ``numpy.random.default_rng(seed)`` is created for every call so
    that each prediction file has a reproducible, order-independent interval.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {k: [] for k in ("accuracy", "macro_f1", "roc_auc", "auprc", "TMR", "FAR")}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_prob[idx]
        stats["accuracy"].append(accuracy_score(yt, yp))
        stats["macro_f1"].append(f1_score(yt, yp, average="macro"))
        stats["roc_auc"].append(roc_auc_score(yt, ys))
        stats["auprc"].append(average_precision_score(yt, ys))
        tmr, far = tmr_far(yt, yp)
        stats["TMR"].append(tmr)
        stats["FAR"].append(far)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: tuple(np.percentile(v, [lo, hi])) for k, v in stats.items()}


def mcnemar(y_true, pred_a, pred_b) -> Dict[str, float]:
    """Continuity-corrected McNemar test on per-example correctness."""
    y_true = np.asarray(y_true)
    ca = np.asarray(pred_a) == y_true
    cb = np.asarray(pred_b) == y_true
    b = int(np.sum(ca & ~cb))   # A correct, B wrong
    c = int(np.sum(~ca & cb))   # A wrong, B correct
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return {"b": b, "c": c, "chi2": stat, "p_value": float(1 - chi2.cdf(stat, 1))}


# --------------------------------------------------------------------------
# Threshold (operating-point) calibration
# --------------------------------------------------------------------------
def tmr_far_at(y_true, y_prob, tau: float):
    y_pred = (np.asarray(y_prob) >= tau).astype(int)
    return tmr_far(np.asarray(y_true).astype(int), y_pred)


def calibrated_operating_point(y_true, y_prob, target_tmr: float) -> Dict[str, float]:
    """tau*(t) = max{tau : TMR(tau) <= t}, prediction = 1[p >= tau].

    TMR(tau) = #{positives with p < tau} / #positives is non-decreasing in
    tau, so the most permissive threshold meeting the target is the largest
    candidate score whose miss count does not exceed floor(t * N_pos).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = np.sort(y_prob[y_true == 1])
    allowed = int(np.floor(target_tmr * len(pos)))
    cand = np.unique(pos)
    misses = np.searchsorted(pos, cand, side="left")   # #pos strictly below candidate
    ok = cand[misses <= allowed]
    tau = float(ok.max())
    tmr, far = tmr_far_at(y_true, y_prob, tau)
    return {"target_TMR": target_tmr, "tau": tau, "TMR": tmr, "FAR": far}


def operating_curve(y_true, y_prob, n_points: int = 400) -> pd.DataFrame:
    """FAR-TMR curve over thresholds (used for the calibration figure)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    qs = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_points)))
    rows = [(t, *tmr_far_at(y_true, y_prob, t)) for t in qs]
    return pd.DataFrame(rows, columns=["tau", "TMR", "FAR"])


# --------------------------------------------------------------------------
# Probability calibration
# --------------------------------------------------------------------------
def reliability_bins(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin m = (edges[m], edges[m+1]]; the first bin also includes 0
    ids = np.clip(np.searchsorted(edges, y_prob, side="left") - 1, 0, n_bins - 1)
    rows = []
    for m in range(n_bins):
        mask = ids == m
        if mask.sum() == 0:
            rows.append((m, edges[m], edges[m + 1], 0, np.nan, np.nan))
            continue
        rows.append((m, edges[m], edges[m + 1], int(mask.sum()),
                     float(y_true[mask].mean()), float(y_prob[mask].mean())))
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "count", "acc", "conf"])


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    rb = reliability_bins(y_true, y_prob, n_bins)
    rb = rb[rb["count"] > 0]
    n = rb["count"].sum()
    return float(np.sum(rb["count"] / n * np.abs(rb["acc"] - rb["conf"])))


# --------------------------------------------------------------------------
# Community (target-group) sensitivity for HateXplain
# --------------------------------------------------------------------------
def community_sensitivity(df: pd.DataFrame, target_col: str = "targets",
                          min_n: Optional[int] = None, top_k: Optional[int] = 12,
                          unspecified_label: str = "unspecified") -> pd.DataFrame:
    """Per-community metrics. Posts with an empty target list form the
    ``unspecified`` group; a post with several targets counts in each."""
    import ast

    groups: Dict[str, list] = {}
    for i, t in enumerate(df[target_col]):
        tl = ast.literal_eval(t) if isinstance(t, str) else list(t)
        tl = [g for g in tl if g and g != "None"] or [unspecified_label]
        for g in tl:
            groups.setdefault(g, []).append(i)
    rows = []
    for g, idx in groups.items():
        s = df.iloc[idx]
        tn, fp, fn, tp = counts(s["y_true"].values, s["y_pred"].values)
        rows.append({
            "community": g, "n": len(idx),
            "macro_f1": f1_score(s["y_true"], s["y_pred"], average="macro"),
            "miss_rate": 100 * fn / (fn + tp) if (fn + tp) else np.nan,
            "false_alarm": 100 * fp / (fp + tn) if (fp + tn) else np.nan,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    if min_n is not None:
        out = out[out["n"] >= min_n]
    if top_k is not None:
        out = out.head(top_k)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Rationale alignment
# --------------------------------------------------------------------------
def rationale_prf(model_tokens: Iterable[int], human_tokens: Iterable[int]):
    """Token-set precision/recall/F1 between model and human rationales."""
    rm, rh = set(model_tokens), set(human_tokens)
    if not rm or not rh:
        return 0.0, 0.0, 0.0
    inter = len(rm & rh)
    p, r = inter / len(rm), inter / len(rh)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# --------------------------------------------------------------------------
# I/O helper
# --------------------------------------------------------------------------
def load_predictions(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"y_true", "y_pred", "y_prob"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df
