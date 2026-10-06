"""Client partitioning: IID and Dirichlet non-IID (label + quantity skew).

For each class c with N_c samples, pi_c ~ Dirichlet_K(alpha) and client k
receives ~pi_{c,k} * N_c samples of class c. Split points are taken at
floor(cumsum(pi_c) * N_c), so every training sample is assigned to exactly
one client (the per-client counts are the floor-based allocation of the
paper's Eq. (dirichlet) with the remainder absorbed by the split points).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def iid_partition(n: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(part) for part in np.array_split(idx, num_clients)]


def dirichlet_partition(labels, num_clients: int, alpha: float = 0.5,
                        seed: int = 42, min_size: int = 1,
                        max_tries: int = 100) -> List[np.ndarray]:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        buckets: List[list] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            pi = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(pi) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                buckets[k].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            return [np.sort(np.array(b, dtype=int)) for b in buckets]
    raise RuntimeError("Could not draw a Dirichlet partition with non-empty clients")


def describe_partition(parts: List[np.ndarray], labels) -> pd.DataFrame:
    labels = np.asarray(labels)
    total = sum(len(p) for p in parts)
    rows = []
    for k, p in enumerate(parts):
        y = labels[p]
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        rows.append({
            "client": k, "n_total": len(p), "n_neg": n0, "n_pos": n1,
            "pos_pct": round(100 * n1 / max(len(p), 1), 2),
            "share_pct": round(100 * len(p) / max(total, 1), 2),
            "pos_to_neg_ratio": round(n1 / n0, 2) if n0 else float("inf"),
        })
    return pd.DataFrame(rows)
