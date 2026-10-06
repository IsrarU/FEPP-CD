"""Federated DistilBERT training with optional Secure Aggregation (Algorithm 1).

Each round t:
  1. the server broadcasts w^t to all K clients;
  2. client k fine-tunes on D_k for E local epochs (AdamW, grad-clip 1.0,
     class-weighted BCE) and forms Delta w_k = w_k^{t+1} - w^t;
  3. the server computes w^{t+1} = w^t + sum_k (n_k / n) Delta w_k (FedAvg),
     either from plain updates or from SecAgg-masked updates;
  4. the global model is evaluated on the frozen test set.

Outputs (in ``output_dir``): predictions.csv (y_true, y_pred, y_prob),
round_log.csv, client_distribution.csv, metrics.json, classification_report.csv,
communication.json, run_config.yaml and optionally model.pt.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from . import secagg
from .attacks import flip_labels
from .comm import comm_cost
from .data import LOADERS, frozen_split, positive_class_weight
from .metrics import compute_metrics
from .model import build_model, count_parameters
from .partition import describe_partition, dirichlet_partition, iid_partition

DEFAULTS: Dict = {
    "dataset": "kaggle",
    "data_path": "data/raw/kaggle/cyberbullying_tweets.csv",
    "model_name": "distilbert-base-uncased",
    "max_len": 128,
    "dropout": 0.30,
    "lr": 2e-5,
    "batch_size": 32,
    "eval_batch_size": 128,
    "local_epochs": 3,
    "num_clients": 5,
    "rounds": 10,
    "partition": "iid",          # iid | dirichlet
    "alpha": 0.5,
    "grad_clip": 1.0,
    "seed": 42,
    "test_size": 0.2,
    "threshold": 0.5,
    "pos_weight": "auto",        # auto -> n_neg / n_pos of the training set
    "secagg": False,
    "secagg_frac_bits": 24,
    "flip_ratio": 0.0,
    "flip_client": 0,
    "comm_rho": 0.25,
    "max_train_samples": None,   # for smoke tests only
    "max_test_samples": None,
    "save_model": False,
    "output_dir": "runs/kaggle_iid",
    "device": "auto",
}


# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels, tokenizer, max_len: int):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.enc["input_ids"][i], self.enc["attention_mask"][i], self.labels[i])


def _float_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if torch.is_floating_point(v)}


def local_train(model, loader, epochs, lr, clip, pos_weight, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    total, batches = 0.0, 0
    for _ in range(epochs):
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += float(loss.item())
            batches += 1
    return total / max(epochs, 1), total / max(batches, 1)


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device))
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def fedavg_aggregate(global_state, client_states, weights):
    new = {}
    for k, g in global_state.items():
        delta = sum(w * (cs[k] - g) for cs, w in zip(client_states, weights))
        new[k] = g + delta
    return new


def secagg_aggregate(global_state, client_states, weights, seeds, round_idx, frac_bits):
    """Same result as FedAvg, but the server only sees masked uint32 vectors."""
    new = {}
    k_clients = len(client_states)
    for stream, (k, g) in enumerate(global_state.items()):
        g_np = g.cpu().double().numpy()
        masked = []
        for cid, (cs, w) in enumerate(zip(client_states, weights)):
            upd = w * (cs[k].cpu().double().numpy() - g_np)          # (n_k/n) * Delta w_k
            enc = secagg.encode(upd.ravel(), frac_bits)
            masked.append(secagg.mask_update(cid, enc, seeds, k_clients, round_idx, stream))
        agg = secagg.decode(secagg.aggregate_masked(masked), frac_bits).reshape(g_np.shape)
        new[k] = (torch.from_numpy(g_np + agg)).to(g.dtype).to(g.device)
    return new


# --------------------------------------------------------------------------
def load_config(path: str | os.PathLike | None, overrides: Dict | None = None) -> Dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _load_dataset(cfg):
    if cfg["dataset"] == "synthetic":
        from .synthetic import make_synthetic
        return make_synthetic(n=cfg.get("synthetic_n", 600), seed=cfg["seed"])
    return LOADERS[cfg["dataset"]](cfg["data_path"])


def run(cfg: Dict) -> Dict:
    t0 = time.time()
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]

    df = _load_dataset(cfg)
    train_df, test_df = frozen_split(df, cfg["test_size"], cfg["seed"])
    if cfg["max_train_samples"]:
        train_df = train_df.sample(n=min(cfg["max_train_samples"], len(train_df)),
                                   random_state=cfg["seed"]).reset_index(drop=True)
    if cfg["max_test_samples"]:
        test_df = test_df.sample(n=min(cfg["max_test_samples"], len(test_df)),
                                 random_state=cfg["seed"]).reset_index(drop=True)
    print(f"[data] {cfg['dataset']}: total={len(df)} train={len(train_df)} test={len(test_df)} "
          f"test_pos={int(test_df.label.sum())}")

    y_train = train_df["label"].values
    if cfg["partition"] == "dirichlet":
        parts = dirichlet_partition(y_train, cfg["num_clients"], cfg["alpha"], cfg["seed"])
    else:
        parts = iid_partition(len(train_df), cfg["num_clients"], cfg["seed"])
    dist = describe_partition(parts, y_train)
    dist.to_csv(out / "client_distribution.csv", index=False)
    print(dist.to_string(index=False))

    pos_weight = positive_class_weight(y_train) if cfg["pos_weight"] == "auto" else float(cfg["pos_weight"])

    model, tok = build_model(cfg["model_name"], cfg["dropout"])
    model.to(device)
    d = count_parameters(model)

    client_loaders, client_sizes = [], []
    for k, idx in enumerate(parts):
        labels = y_train[idx]
        if cfg["flip_ratio"] > 0 and k == cfg["flip_client"]:
            labels, flipped = flip_labels(labels, cfg["flip_ratio"], cfg["seed"])
            print(f"[attack] client {k}: flipped {len(flipped)} harmful labels (gamma={cfg['flip_ratio']})")
        ds = TextDataset(train_df["text"].values[idx], labels, tok, cfg["max_len"])
        g = torch.Generator().manual_seed(cfg["seed"] + k)
        client_loaders.append(DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=g))
        client_sizes.append(len(idx))
    n = sum(client_sizes)
    weights = [nk / n for nk in client_sizes]

    test_loader = DataLoader(TextDataset(test_df["text"].values, test_df["label"].values, tok, cfg["max_len"]),
                             batch_size=cfg["eval_batch_size"], shuffle=False)
    y_test = test_df["label"].values

    seeds = secagg.setup_pairwise_seeds(cfg["num_clients"], seed=cfg["seed"]) if cfg["secagg"] else None

    log = []
    for r in range(1, cfg["rounds"] + 1):
        global_state = _float_state(model)
        client_states, losses, batch_losses = [], [], []
        for k, loader in enumerate(client_loaders):
            local = copy.deepcopy(model)
            ep_loss, b_loss = local_train(local, loader, cfg["local_epochs"], cfg["lr"],
                                          cfg["grad_clip"], pos_weight, device)
            client_states.append(_float_state(local))
            losses.append(ep_loss)
            batch_losses.append(b_loss)
            del local
        if cfg["secagg"]:
            new_state = secagg_aggregate(global_state, client_states, weights, seeds, r,
                                         cfg["secagg_frac_bits"])
        else:
            new_state = fedavg_aggregate(global_state, client_states, weights)
        model.load_state_dict(new_state, strict=False)

        prob = predict_proba(model, test_loader, device)
        m = compute_metrics(y_test, (prob >= cfg["threshold"]).astype(int), prob)
        log.append({"round": r, "accuracy": m.accuracy, "macro_f1": m.macro_f1,
                    "avg_loss": float(np.mean(losses)), "mean_batch_loss": float(np.mean(batch_losses))})
        print(f"[round {r}] acc={m.accuracy:.4f} macroF1={m.macro_f1:.4f} "
              f"loss={np.mean(losses):.3f} TMR={m.TMR:.4f} FAR={m.FAR:.4f}")
    pd.DataFrame(log).to_csv(out / "round_log.csv", index=False)

    y_pred = (prob >= cfg["threshold"]).astype(int)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_prob": prob}).to_csv(out / "predictions.csv", index=False)
    rep = classification_report(y_test, y_pred, target_names=["Non-harmful (0)", "Harmful (1)"],
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out / "classification_report.csv")
    final = compute_metrics(y_test, y_pred, prob).to_dict()
    final.update({"parameters": d, "pos_weight": pos_weight, "seconds": time.time() - t0})
    (out / "metrics.json").write_text(json.dumps(final, indent=2, default=float))

    cc = comm_cost(cfg["rounds"], cfg["num_clients"], d, 4, cfg["comm_rho"]).iloc[-1]
    comm = {"parameters": d, "fedavg_gb_analytical": float(cc.fedavg_gb),
            "secagg_gb_analytical": float(cc.secagg_gb), "rho_assumed": cfg["comm_rho"],
            "secagg_payload_bytes_per_client_round": secagg.payload_bytes(d) if cfg["secagg"] else None}
    (out / "communication.json").write_text(json.dumps(comm, indent=2))
    if cfg["save_model"]:
        torch.save(model.state_dict(), out / "model.pt")
    print(f"[done] {out} ({time.time() - t0:.1f}s)")
    return final
