#!/usr/bin/env python3
"""Regenerate the data-driven figures of the paper from the released files.

Usage:  python scripts/make_figures.py [--results results] [--fmt pdf]
Output: results/figures/<name>.<fmt>  (file names match figs/ in the paper source)

The framework diagram (mainfig.png) and the qualitative state-of-the-art
matrix are not data-driven and are not regenerated here.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fepp_cd.comm import comm_cost  # noqa: E402
from fepp_cd.metrics import (  # noqa: E402
    community_sensitivity, compute_metrics, operating_curve, reliability_bins,
)

# Fixed categorical order (validated: CVD-safe adjacent pairs, see README)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 5, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def load(res, rel):
    return pd.read_csv(res / rel)


def save(fig, out, name, fmt):
    fig.savefig(out / f"{name}.{fmt}")
    plt.close(fig)


def cm_panel(ax, d, title):
    cm = confusion_matrix(d.y_true, d.y_pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["Non-harm", "Harm"])
    ax.set_yticks([0, 1], ["Non-harm", "Harm"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--fmt", default="pdf")
    a = ap.parse_args()
    res = Path(a.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    f = a.fmt

    P = {
        "Kaggle IID": load(res, "predictions/kaggle/iid.csv"),
        "Kaggle non-IID": load(res, "predictions/kaggle/noniid.csv"),
        "Jigsaw IID": load(res, "predictions/jigsaw/iid.csv"),
        "Jigsaw non-IID": load(res, "predictions/jigsaw/noniid.csv"),
        "HateXplain (external)": load(res, "predictions/hatexplain/predictions.csv"),
    }
    M = {k: compute_metrics(d.y_true, d.y_pred, d.y_prob) for k, d in P.items()}

    # cm_iid --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 3))
    cm_panel(ax, P["Kaggle IID"], "Kaggle IID")
    save(fig, out, "cm_iid", f)

    # iid_vs_noniid --------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    a_, b_ = M["Kaggle IID"], M["Kaggle non-IID"]
    x = np.arange(4)
    for i, (m, lab) in enumerate([(a_, "IID"), (b_, "Non-IID")]):
        axs[0].bar(x + (i - 0.5) * 0.38, [m.accuracy, m.macro_f1, m.roc_auc, m.auprc], 0.36,
                   color=C[i], label=lab, edgecolor="white", linewidth=1)
        axs[1].bar(np.arange(2) + (i - 0.5) * 0.38, [100 * m.TMR, 100 * m.FAR], 0.36,
                   color=C[i], label=lab, edgecolor="white", linewidth=1)
    axs[0].set_xticks(x, ["Accuracy", "Macro F1", "ROC-AUC", "AUPRC"])
    axs[0].set_ylim(0.75, 1.0)
    axs[0].set_title("Ranking and accuracy metrics")
    axs[1].set_xticks([0, 1], ["TMR (%)", "FAR (%)"])
    axs[1].set_title(r"Safety metrics at $\tau=0.5$")
    for ax in axs:
        ax.legend(frameon=False)
    save(fig, out, "iid_vs_noniid", f)

    # calibration (FAR-TMR operating curves) -------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for i, k in enumerate(["Kaggle IID", "Kaggle non-IID"]):
        d = P[k]
        oc = operating_curve(d.y_true.values, d.y_prob.values, 600)
        ax.plot(100 * oc.TMR, 100 * oc.FAR, color=C[i], label=k.replace("Kaggle ", ""))
        m = M[k]
        ax.plot(100 * m.TMR, 100 * m.FAR, "o", color=C[i], markersize=8, markeredgecolor="white")
    for t in (2, 5):
        ax.axvline(t, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, 12)
    ax.set_xlabel("Toxic miss rate, TMR (%)")
    ax.set_ylabel("False alarm rate, FAR (%)")
    ax.legend(frameon=False, title=r"markers: $\tau=0.5$", title_fontsize=8)
    save(fig, out, "calibration", f)

    # reliability ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    for i, k in enumerate(["Kaggle IID", "Jigsaw non-IID", "HateXplain (external)"]):
        rb = reliability_bins(P[k].y_true, P[k].y_prob, 10).dropna()
        ax.plot(rb.conf, rb.acc, "o-", color=C[i], label=k)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed harmful frequency")
    ax.legend(frameon=False)
    save(fig, out, "reliability", f)

    # score_dist -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    d = P["Kaggle IID"]
    bins = np.linspace(0, 1, 41)
    for i, (lab, yv) in enumerate([("Non-harmful", 0), ("Harmful", 1)]):
        ax.hist(d.y_prob[d.y_true == yv], bins=bins, color=C[i], alpha=0.75, label=lab)
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted harmful probability")
    ax.set_ylabel("Count (log)")
    ax.legend(frameon=False)
    save(fig, out, "score_dist", f)

    # partition_composition ------------------------------------------------------------
    kp = load(res, "logs/kaggle_noniid_client_distribution.csv")
    jp = load(res, "logs/jigsaw_noniid_client_distribution.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, cl, pct, n, title in [
        (axs[0], kp.client, 100 * kp.n_cb / kp.n_total, kp.n_total, "Kaggle (non-IID)"),
        (axs[1], jp.client, jp.toxic_pct, jp.n_total, "Jigsaw (non-IID)"),
    ]:
        bars = ax.bar(cl, pct, color=C[0], edgecolor="white")
        for bar, nn in zip(bars, n):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"n={nn:,}",
                    ha="center", fontsize=7, color=INK)
        ax.set_xticks(cl, [f"C{c}" for c in cl])
        ax.set_title(title)
        ax.set_ylim(0, 110)
    axs[0].set_ylabel("Harmful share of client data (%)")
    save(fig, out, "partition_composition", f)

    # jigsaw_convergence (two panels instead of a dual axis) ---------------------
    rl = load(res, "logs/jigsaw_noniid_round_log.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    axs[0].plot(rl["round"], rl.accuracy, "o-", color=C[0], label="Accuracy")
    axs[0].plot(rl["round"], rl.macro_f1, "s-", color=C[1], label="Macro F1")
    axs[0].set_xlabel("Communication round")
    axs[0].set_ylabel("Score")
    axs[0].legend(frameon=False)
    axs[1].plot(rl["round"], rl.avg_loss, "o-", color=C[2])
    axs[1].set_xlabel("Communication round")
    axs[1].set_ylabel("Average local loss")
    for ax in axs:
        ax.set_xticks(rl["round"])
    save(fig, out, "jigsaw_convergence", f)

    # comm -------------------------------------------------------------------------
    cc = comm_cost(10, 5, 66_000_000, 4, 0.25)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(cc["round"], cc.fedavg_gb, "o-", color=C[0], label="FedAvg")
    ax.plot(cc["round"], cc.secagg_gb, "s--", color=C[1], label=r"FedAvg + SecAgg ($\rho=0.25$)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative upload (GB)")
    ax.legend(frameon=False)
    save(fig, out, "comm", f)

    # label flipping ---------------------------------------------------------------------
    flips = {0: P["Kaggle IID"], 10: load(res, "predictions/kaggle/flip_010.csv"),
             20: load(res, "predictions/kaggle/flip_020.csv"), 30: load(res, "predictions/kaggle/flip_030.csv")}
    FM = {g: compute_metrics(d.y_true, d.y_pred, d.y_prob) for g, d in flips.items()}
    g = list(FM)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    axs[0].plot(g, [FM[x].accuracy for x in g], "o-", color=C[0])
    axs[0].set_ylabel("Accuracy")
    axs[1].plot(g, [100 * FM[x].TMR for x in g], "o-", color=C[1])
    axs[1].set_ylabel("Toxic miss rate (%)")
    for ax in axs:
        ax.set_xlabel("Flip ratio (%)")
        ax.set_xticks(g)
    save(fig, out, "labelflip", f)

    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.6))
    for ax, (lab, fn, col) in zip(axs, [("ROC-AUC", lambda m: m.roc_auc, C[0]),
                                        ("AUPRC", lambda m: m.auprc, C[1]),
                                        ("Accuracy", lambda m: m.accuracy, C[2])]):
        ax.plot(g, [fn(FM[x]) for x in g], "o-", color=col)
        ax.set_title(lab)
        ax.set_xlabel("Flip ratio (%)")
        ax.set_xticks(g)
    save(fig, out, "flip_ranking", f)

    # rationale ------------------------------------------------------------------------
    ra = load(res, "xai/rationale_alignment.csv")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.hist(ra.f1, bins=np.linspace(0, 1, 11), color=C[0], edgecolor="white")
    ax.axvline(ra.f1.mean(), color=INK, linestyle="--", linewidth=1, label=f"mean F1 = {ra.f1.mean():.3f}")
    ax.set_xlabel("Rationale alignment F1")
    ax.set_ylabel("Posts")
    ax.legend(frameon=False)
    save(fig, out, "rationale", f)

    # cm_multi ----------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.9))
    for j, (ax, k) in enumerate(zip(axs, ["Kaggle non-IID", "HateXplain (external)", "Jigsaw non-IID"])):
        cm_panel(ax, P[k], k)
        if j:
            ax.set_ylabel("")
    fig.tight_layout()
    save(fig, out, "cm_multi", f)

    # community -------------------------------------------------------------------------
    cs = community_sensitivity(P["HateXplain (external)"], top_k=12).sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.barh(cs.community, cs.macro_f1, color=C[0], edgecolor="white")
    ax.axvline(cs.macro_f1.mean(), color=INK, linestyle="--", linewidth=1,
               label=f"mean = {cs.macro_f1.mean():.3f}")
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel("Macro F1")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "community", f)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sc = ax.scatter(cs.false_alarm, cs.miss_rate, s=cs.n / 3, c=cs.macro_f1, cmap="Blues",
                    edgecolor=INK, linewidth=0.5, vmin=0.45, vmax=0.78)
    for _, r in cs.iterrows():
        ax.annotate(r.community, (r.false_alarm, r.miss_rate), fontsize=7, xytext=(4, 3),
                    textcoords="offset points", color=INK)
    fig.colorbar(sc, ax=ax, label="Macro F1")
    ax.set_xlabel("False alarm rate (%)")
    ax.set_ylabel("Miss rate (%)")
    save(fig, out, "community_scatter", f)

    # global_token -------------------------------------------------------------------------
    gt = load(res, "xai/global_token_importance.csv").sort_values("mean_attribution")
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.barh(gt.token, gt.mean_attribution, color=C[0], edgecolor="white")
    ax.set_xlabel("Mean IG attribution")
    save(fig, out, "global_token", f)

    # cross_dataset_performance ------------------------------------------------------------
    keys = ["Kaggle IID", "Kaggle non-IID", "Jigsaw IID", "Jigsaw non-IID", "HateXplain (external)"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.5, 3))
    for i, (lab, fn) in enumerate([("Accuracy", lambda m: m.accuracy), ("Macro F1", lambda m: m.macro_f1),
                                   ("ROC-AUC", lambda m: m.roc_auc)]):
        ax.bar(x + (i - 1) * 0.26, [fn(M[k]) for k in keys], 0.24, color=C[i], label=lab,
               edgecolor="white", linewidth=1)
    ax.set_xticks(x, [k.replace(" (external)", "") for k in keys])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, out, "cross_dataset_performance", f)

    # noniid_f1_degradation -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for i, (lab, ks) in enumerate([("IID", ["Kaggle IID", "Jigsaw IID"]),
                                   ("Non-IID", ["Kaggle non-IID", "Jigsaw non-IID"])]):
        ax.bar(np.arange(2) + (i - 0.5) * 0.38, [M[k].macro_f1 for k in ks], 0.36, color=C[i],
               label=lab, edgecolor="white", linewidth=1)
    for j, (k1, k2) in enumerate([("Kaggle IID", "Kaggle non-IID"), ("Jigsaw IID", "Jigsaw non-IID")]):
        ax.text(j, M[k1].macro_f1 + 0.006, f"drop = {M[k1].macro_f1 - M[k2].macro_f1:.4f}",
                ha="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["Kaggle", "Jigsaw"])
    ax.set_ylim(0.78, 0.93)
    ax.set_ylabel("Macro F1")
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, "noniid_f1_degradation", f)

    # ROC & PR -------------------------------------------------------------------------------
    sel = ["Kaggle IID", "HateXplain (external)", "Jigsaw non-IID"]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for i, k in enumerate(sel):
        fpr, tpr, _ = roc_curve(P[k].y_true, P[k].y_prob)
        ax.plot(fpr, tpr, color=C[i], label=f"{k}  AUC={M[k].roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, out, "roc_curves", f)

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for i, k in enumerate(sel):
        pr, rc, _ = precision_recall_curve(P[k].y_true, P[k].y_prob)
        ax.plot(rc, pr, color=C[i], label=f"{k}  AP={M[k].auprc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, out, "pr_curves", f)

    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
