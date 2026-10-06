"""Tiny synthetic corpus for offline smoke tests (not used for any reported result)."""

from __future__ import annotations

import numpy as np
import pandas as pd

_HARM = ["you are stupid", "shut up loser", "i hate you", "go away idiot", "you are ugly and dumb",
         "nobody likes you loser", "kill yourself idiot", "fat ugly bully"]
_SAFE = ["thanks my friend", "have a nice day", "great game today", "i love this school",
         "good job everyone", "see you at school", "what a funny joke lol", "nice to meet you"]


def make_synthetic(n: int = 600, pos_frac: float = 0.6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(int)
    text = [(rng.choice(_HARM) if yi else rng.choice(_SAFE)) + f" {rng.choice(['a', 'the', 'all'])}" for yi in y]
    return pd.DataFrame({"text": text, "label": y})
