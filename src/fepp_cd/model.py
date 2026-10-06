"""DistilBERT binary classifier: p = sigmoid(W_c Dropout(h_[CLS]) + b_c)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, DistilBertConfig, DistilBertModel


class DistilBertClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.30):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.dim, 1)

    @property
    def embeddings(self):
        return self.encoder.embeddings

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           inputs_embeds=inputs_embeds)
        h = out.last_hidden_state[:, 0]          # [CLS] representation
        return self.head(self.dropout(h)).squeeze(-1)  # logits


def build_model(name: str = "distilbert-base-uncased", dropout: float = 0.30,
                tiny_vocab_size: int | None = None):
    """Returns (model, tokenizer).

    ``name='tiny-random'`` builds a 2-layer randomly initialised DistilBERT
    with a locally generated tokenizer; it is used only for smoke tests / CI
    where the Hugging Face Hub is not reachable.
    """
    if name == "tiny-random":
        from .tiny_tokenizer import build_tiny_tokenizer
        tok = build_tiny_tokenizer()
        cfg = DistilBertConfig(vocab_size=tok.vocab_size, dim=64, hidden_dim=128,
                               n_layers=2, n_heads=2, max_position_embeddings=128)
        return DistilBertClassifier(DistilBertModel(cfg), dropout), tok
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name)
    return DistilBertClassifier(enc, dropout), tok


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
