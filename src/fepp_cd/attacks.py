"""Label-flipping poisoning (paper Section 3.3).

A single malicious client (client 0 by default) flips a fraction gamma of
its *harmful* labels to non-harmful before local training:

    N_flip = floor(gamma * |{i in D_0 : y_i = 1}|)
"""

from __future__ import annotations

import numpy as np


def flip_labels(labels, gamma: float, seed: int = 42):
    """Return (new_labels, flipped_indices). Only positives are flipped 1 -> 0."""
    labels = np.asarray(labels).copy()
    if gamma <= 0:
        return labels, np.array([], dtype=int)
    pos = np.where(labels == 1)[0]
    n_flip = int(np.floor(gamma * len(pos)))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(pos, size=n_flip, replace=False))
    labels[chosen] = 0
    return labels, chosen
