"""Integrated Gradients token attribution and HateXplain rationale alignment.

* IG is computed on the embedding layer with Captum's
  ``LayerIntegratedGradients`` (S = 50 steps). The baseline keeps [CLS]/[SEP]
  and replaces every other token by [PAD].
* Token score = L2 norm of the attribution vector over the embedding
  dimension, normalised by the maximum score in the post; special tokens are
  removed and WordPiece pieces are merged into words by summation.
* Rationale alignment: for each post with a human rationale R_h, the model
  rationale R_m is the set of the k = |R_h| highest-scoring words; token-set
  precision, recall and F1 are then averaged over posts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import rationale_prf


class IGExplainer:
    def __init__(self, model, tokenizer, max_len: int = 128, n_steps: int = 50,
                 device: str = "cpu", internal_batch_size: int = 16):
        from captum.attr import LayerIntegratedGradients

        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.max_len = max_len
        self.n_steps = n_steps
        self.device = device
        self.ibs = internal_batch_size

        def forward(input_ids, attention_mask):
            return torch.sigmoid(self.model(input_ids=input_ids, attention_mask=attention_mask))

        self.lig = LayerIntegratedGradients(forward, self.model.embeddings)

    def word_scores(self, words: Sequence[str]) -> np.ndarray:
        """Normalised IG score per input word (length = len(words), truncated words get 0)."""
        enc = self.tok(list(words), is_split_into_words=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        special = torch.tensor(self.tok.get_special_tokens_mask(
            ids[0].tolist(), already_has_special_tokens=True), dtype=torch.bool, device=self.device)
        base = torch.where(special, ids, torch.full_like(ids, self.tok.pad_token_id))
        attr = self.lig.attribute(inputs=ids, baselines=base, additional_forward_args=(mask,),
                                  n_steps=self.n_steps, internal_batch_size=self.ibs)
        tok_scores = attr[0].norm(dim=-1).detach().cpu().numpy()
        tok_scores[special.cpu().numpy()] = 0.0
        if tok_scores.max() > 0:
            tok_scores = tok_scores / tok_scores.max()
        out = np.zeros(len(words))
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                out[wid] += tok_scores[pos]      # WordPiece merge (sum)
        return out


def rationale_alignment(explainer: IGExplainer, df: pd.DataFrame, n: int = 100,
                        seed: int = 42) -> pd.DataFrame:
    """df needs columns ``tokens`` (list of words) and ``rationale_pos`` (list of ints)."""
    cand = df[df["rationale_pos"].map(len) > 0]
    cand = cand.sample(n=min(n, len(cand)), random_state=seed)
    rows = []
    for _, r in cand.iterrows():
        scores = explainer.word_scores(r["tokens"])
        k = len(r["rationale_pos"])
        top = np.argsort(-scores)[:k]
        p, rc, f = rationale_prf(top.tolist(), r["rationale_pos"])
        rows.append({"post_id": r.get("post_id"), "k": k, "f1": f, "precision": p, "recall": rc})
    return pd.DataFrame(rows)


def global_importance(explainer: IGExplainer, texts: Sequence[str], top_k: int = 20) -> pd.DataFrame:
    acc: Dict[str, List[float]] = defaultdict(list)
    for t in texts:
        words = str(t).split()
        if not words:
            continue
        for w, s in zip(words, explainer.word_scores(words)):
            acc[w].append(float(s))
    rows = [(w, float(np.mean(v)), len(v)) for w, v in acc.items()]
    df = pd.DataFrame(rows, columns=["token", "mean_attribution", "count"])
    return df.sort_values("mean_attribution", ascending=False).head(top_k).reset_index(drop=True)
