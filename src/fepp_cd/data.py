"""Dataset loading, cleaning, binary label mapping, and the frozen split.

Raw datasets are NOT redistributed in this repository (see docs/DATA.md).
Expected raw files (default locations, configurable in the YAML configs):

* Kaggle Cyberbullying Classification: data/raw/kaggle/cyberbullying_tweets.csv
  (columns ``tweet_text``, ``cyberbullying_type``)
* Jigsaw Toxic Comment Classification: data/raw/jigsaw/train.csv
  (``comment_text`` + six toxicity columns)
* HateXplain: data/raw/hatexplain/dataset.json

Binary mapping (positive = harmful):
* Kaggle:     any cyberbullying type -> 1, ``not_cyberbullying`` -> 0
* Jigsaw:     any of the six toxicity labels -> 1, otherwise 0
* HateXplain: majority label ``hatespeech`` or ``offensive`` -> 1, ``normal`` -> 0
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_TEXT = re.compile(r"[^a-z0-9\s']")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = s.replace("#", " ")
    s = _NON_TEXT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _finalize(df: pd.DataFrame, text_col: str, dedup: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df[text_col].map(clean_text)
    df = df[df["text"].str.len() > 0]
    if dedup:
        df = df.drop_duplicates(subset="text")
    return df.reset_index(drop=True)


def load_kaggle(path: str | Path, dedup: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)
    df["category"] = df["cyberbullying_type"]
    return _finalize(df, "tweet_text", dedup)[["text", "label", "category"]]


def load_jigsaw(path: str | Path, dedup: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = (df[JIGSAW_LABELS].sum(axis=1) > 0).astype(int)
    return _finalize(df, "comment_text", dedup)[["text", "label"]]


def load_hatexplain(path: str | Path) -> pd.DataFrame:
    """Majority-vote labels; posts without a majority are dropped.

    ``targets`` keeps communities named by at least two annotators;
    ``rationale_pos`` holds word positions marked by a majority of the
    rationale annotators (harmful posts only).
    """
    data = json.loads(Path(path).read_text())
    rows = []
    for pid, post in data.items():
        ann = post["annotators"]
        votes = Counter(a["label"] for a in ann)
        label, cnt = votes.most_common(1)[0]
        if cnt < 2:
            continue
        tgt = Counter(t for a in ann for t in a.get("target", []) if t and t != "None")
        targets = sorted(t for t, c in tgt.items() if c >= 2)
        tokens = post["post_tokens"]
        rat = post.get("rationales", [])
        if rat:
            mean = np.mean(np.array(rat, dtype=float), axis=0)
            rationale_pos = [int(i) for i in np.where(mean >= 0.5)[0]]
        else:
            rationale_pos = []
        rows.append({
            "post_id": pid,
            "text": " ".join(tokens),
            "tokens": tokens,
            "label_orig": label,
            "label": int(label in ("hatespeech", "offensive")),
            "targets": targets,
            "rationale_pos": rationale_pos,
            "has_rationale": int(len(rationale_pos) > 0),
            "n_annotators": len(ann),
            "label_agree": cnt / len(ann),
        })
    return pd.DataFrame(rows)


LOADERS = {"kaggle": load_kaggle, "jigsaw": load_jigsaw, "hatexplain": load_hatexplain}


def frozen_split(df: pd.DataFrame, test_size: float = 0.2,
                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split with a fixed seed (the 'frozen' test set)."""
    tr, te = train_test_split(df, test_size=test_size, stratify=df["label"],
                              random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def positive_class_weight(labels) -> float:
    """omega+ = n_negative / n_positive (paper Eq. loss)."""
    labels = np.asarray(labels)
    return float((labels == 0).sum() / max((labels == 1).sum(), 1))
