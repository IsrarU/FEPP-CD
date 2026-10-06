#!/usr/bin/env bash
# Re-run every experiment of the paper (GPU recommended; ~T4 class).
# Raw datasets must be placed as described in docs/DATA.md.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train.py --config configs/kaggle_iid.yaml --save_model
python scripts/train.py --config configs/kaggle_iid_secagg.yaml
python scripts/train.py --config configs/kaggle_noniid.yaml
for r in 010 020 030; do
  python scripts/train.py --config configs/kaggle_flip_${r}.yaml
done
python scripts/train.py --config configs/jigsaw_iid.yaml
python scripts/train.py --config configs/jigsaw_noniid.yaml
python scripts/train.py --config configs/hatexplain.yaml --save_model
python scripts/explain.py --mode rationale --model runs/hatexplain/model.pt \
  --data_path data/raw/hatexplain/dataset.json
python scripts/explain.py --mode global --dataset kaggle --model runs/kaggle_iid/model.pt \
  --data_path data/raw/kaggle/cyberbullying_tweets.csv

echo "New predictions are in runs/*/predictions.csv."
echo "To analyse them, copy them into results/predictions/ (same names) and run:"
echo "  python scripts/evaluate.py && python scripts/make_figures.py"
