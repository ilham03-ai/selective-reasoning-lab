from __future__ import annotations

import torch
from torch import nn

from config import EnvironmentConfig, ModelConfig
from models.encoder import ObservationEncoder


class SelectiveDecisionModel(nn.Module):
    def __init__(self, env_config: EnvironmentConfig, model_config: ModelConfig):
        super().__init__()
        self.num_states = env_config.num_states
        self.pad_token_id = env_config.num_observations
        self.encoder = ObservationEncoder(
            num_tokens=env_config.num_observations + 1,
            embedding_dim=model_config.embedding_dim,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
            pad_token_id=self.pad_token_id,
        )
        self.state_head = nn.Linear(model_config.hidden_dim, env_config.num_states)
        decision_input_dim = model_config.hidden_dim + env_config.num_states + 2
        self.decision_head = nn.Sequential(
            nn.Linear(decision_input_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim, 3),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(sequences, lengths)
        state_logits = self.state_head(hidden)
        state_probs = torch.softmax(state_logits, dim=-1)
        topk = torch.topk(state_probs, k=min(2, self.num_states), dim=-1).values
        max_prob = topk[:, :1]
        margin = max_prob if self.num_states == 1 else max_prob - topk[:, 1:2]
        decision_features = torch.cat([hidden, state_probs, max_prob, margin], dim=-1)
        decision_logits = self.decision_head(decision_features)
        return {
            "hidden": hidden,
            "state_logits": state_logits,
            "state_probs": state_probs,
            "decision_logits": decision_logits,
        }
