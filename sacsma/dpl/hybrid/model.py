"""The hybrid LSTM (ported from neuralhyd-ca ``SingleLSTM``).

Entity-aware, many-to-one: a 365-day window of [basin forcing + SAC-SMA sim
+ static embedding] -> the final hidden state -> a linear head ->
streamflow (Softplus, >= 0).  The physics sim enters ONLY as an input channel.

The net emits a NORMALIZED prediction (the trainer scales the target by each
basin's cal-window std); denormalization back to mm/day lives in the trainer/
evaluator so the model stays a plain sequence regressor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridLSTM(nn.Module):
    def __init__(self, n_dynamic: int, n_static: int, *,
                 hidden: int, static_embed: int, dropout: float):
        """The statics enter through one linear map; the head is linear."""
        super().__init__()
        self.static_encoder = nn.Linear(n_static, static_embed)
        self.lstm = nn.LSTM(n_dynamic + static_embed, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softplus(),                     # streamflow is non-negative
        )

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        tw = x_dyn.shape[1]
        e = self.static_encoder(x_static).unsqueeze(1).expand(-1, tw, -1)
        _, (h, _) = self.lstm(torch.cat([x_dyn, e], dim=-1))
        h = self.dropout(h.squeeze(0))
        return self.head(h).squeeze(-1)        # (B,) normalized prediction
