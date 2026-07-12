"""The hybrid LSTM (ported from neuralhyd-ca ``SingleLSTM``).

Entity-aware, many-to-one: a 365-day window of [basin forcing + SAC-SMA sim
(+ static embedding)] -> the final hidden state -> a small MLP head.  The head
is the only variant-dependent piece: ``feature`` predicts streamflow directly
(Softplus, >= 0); ``residual`` predicts the signed SAC-SMA error (linear).

The net emits a NORMALIZED prediction (the trainer scales the target by each
basin's cal-window std); denormalization back to mm/day lives in the trainer/
evaluator so the model stays a plain sequence regressor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridLSTM(nn.Module):
    def __init__(self, n_dynamic: int, n_static: int = 0, *,
                 variant: str = "residual", hidden: int = 128,
                 static_embed: int = 16, dropout: float = 0.15):
        super().__init__()
        if variant not in ("feature", "residual"):
            raise ValueError(f"variant {variant!r}")
        self.variant = variant
        self.n_static = n_static
        if n_static > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(n_static, static_embed), nn.ReLU(),
                nn.Linear(static_embed, static_embed), nn.ReLU(),
            )
            in_size = n_dynamic + static_embed
        else:
            self.static_encoder = None
            in_size = n_dynamic
        self.lstm = nn.LSTM(in_size, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        head: list[nn.Module] = [nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)]
        if variant == "feature":
            head.append(nn.Softplus())        # streamflow is non-negative
        self.head = nn.Sequential(*head)

    def forward(self, x_dyn: torch.Tensor,
                x_static: torch.Tensor | None = None) -> torch.Tensor:
        tw = x_dyn.shape[1]
        if self.static_encoder is not None:
            e = self.static_encoder(x_static).unsqueeze(1).expand(-1, tw, -1)
            x = torch.cat([x_dyn, e], dim=-1)
        else:
            x = x_dyn
        _, (h, _) = self.lstm(x)
        h = self.dropout(h.squeeze(0))
        return self.head(h).squeeze(-1)        # (B,) normalized prediction
