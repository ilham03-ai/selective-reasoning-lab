from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        pad_token_id: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(sequences))
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.dropout(hidden[-1])
