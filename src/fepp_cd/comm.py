"""Analytical communication accounting (paper Eq. comm).

C_FedAvg = T * K * d * 4 bytes (float32 full-model uploads)
C_SecAgg = (1 + rho) * C_FedAvg, with rho = 0.25 the masking-and-key-agreement
overhead *assumed* in the paper (an accounting assumption, not a measurement).
"""

from __future__ import annotations

import pandas as pd

DISTILBERT_PARAMS = 66_000_000  # d ~ 66M as used in the paper


def comm_cost(rounds: int = 10, clients: int = 5, params: int = DISTILBERT_PARAMS,
              bytes_per_param: int = 4, rho: float = 0.25) -> pd.DataFrame:
    rows = []
    for t in range(1, rounds + 1):
        fedavg = t * clients * params * bytes_per_param
        rows.append({"round": t, "fedavg_gb": fedavg / 1e9,
                     "secagg_gb": (1 + rho) * fedavg / 1e9})
    return pd.DataFrame(rows)
