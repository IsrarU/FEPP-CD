#!/usr/bin/env python3
"""Train a federated DistilBERT model (FedAvg, optional SecAgg / label flipping).

Examples:
  python scripts/train.py --config configs/kaggle_iid.yaml
  python scripts/train.py --config configs/kaggle_noniid.yaml
  python scripts/train.py --config configs/kaggle_iid.yaml --secagg --output_dir runs/kaggle_iid_secagg
  python scripts/train.py --config configs/smoke.yaml          # 1-minute CPU smoke test
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fepp_cd.federated import load_config, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir")
    ap.add_argument("--data_path")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--flip_ratio", type=float)
    ap.add_argument("--secagg", action="store_true", default=None)
    ap.add_argument("--save_model", action="store_true", default=None)
    ap.add_argument("--device")
    args = vars(ap.parse_args())
    cfg = load_config(args.pop("config"), args)
    run(cfg)


if __name__ == "__main__":
    main()
