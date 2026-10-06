#!/usr/bin/env python3
"""Integrated Gradients explanations for a trained FEPP-CD model.

  # HateXplain rationale alignment (100 posts with human rationales, k = |R_h|)
  python scripts/explain.py --mode rationale --model runs/hatexplain/model.pt \
      --data_path data/raw/hatexplain/dataset.json --out results/xai/rationale_alignment_rerun.csv

  # Global token importance on the Kaggle test set
  python scripts/explain.py --mode global --dataset kaggle --model runs/kaggle_iid/model.pt \
      --data_path data/raw/kaggle/cyberbullying_tweets.csv --n_texts 500
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from fepp_cd.data import LOADERS, frozen_split  # noqa: E402
from fepp_cd.explain import IGExplainer, global_importance, rationale_alignment  # noqa: E402
from fepp_cd.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rationale", "global"], required=True)
    ap.add_argument("--dataset", default="hatexplain")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model", required=True, help="state_dict saved by train.py --save_model")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_texts", type=int, default=500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, tok = build_model(a.model_name)
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    ex = IGExplainer(model, tok, n_steps=a.steps, device=a.device)
    _, test = frozen_split(LOADERS[a.dataset](a.data_path), 0.2, a.seed)
    if a.mode == "rationale":
        df = rationale_alignment(ex, test, n=a.n, seed=a.seed)
        print(df[["precision", "recall", "f1"]].mean())
        out = a.out or "results/xai/rationale_alignment_rerun.csv"
    else:
        df = global_importance(ex, test["text"].sample(min(a.n_texts, len(test)), random_state=a.seed))
        print(df.head(20))
        out = a.out or "results/xai/global_token_importance_rerun.csv"
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
