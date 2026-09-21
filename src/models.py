"""
models.py — PyTorch Model Definitions for ECG Forecasting
==========================================================
Configurable LSTM and GRU architectures with proper weight
initialisation, dropout regularisation, and optional
bidirectional processing.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMForecaster(nn.Module):
    """Multi-step time-series forecaster using stacked LSTM layers.

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step (default 8).
    hidden_dim : int
        Hidden state dimensionality of each LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers (ignored when
        ``num_layers == 1``).
    bidirectional : bool
        If ``True``, use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.model_name = self._make_name(num_layers, bidirectional)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input, 1)

        self._init_weights()

    # ── helpers ──────────────────────────────
    @staticmethod
    def _make_name(n_layers: int, bidir: bool) -> str:
        prefix = "BiLSTM" if bidir else "LSTM"
        return f"{prefix}-{n_layers} layer{'s' if n_layers > 1 else ''}"

    def _init_weights(self):
        """Xavier uniform initialisation for all parameters."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    # ── forward ─────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape ``(batch, seq_len, input_dim)``

        Returns
        -------
        Tensor, shape ``(batch, seq_len, 1)``
        """
        lstm_out, _ = self.lstm(x)          # (B, T, H) or (B, T, 2H)
        predictions = self.fc(lstm_out)     # (B, T, 1)
        return predictions


class GRUForecaster(nn.Module):
    """Multi-step time-series forecaster using stacked GRU layers.

    Same API as :class:`LSTMForecaster` but uses GRU cells, which
    have fewer parameters and often train faster on small datasets.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.model_name = self._make_name(num_layers, bidirectional)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input, 1)

        self._init_weights()

    @staticmethod
    def _make_name(n_layers: int, bidir: bool) -> str:
        prefix = "BiGRU" if bidir else "GRU"
        return f"{prefix}-{n_layers} layer{'s' if n_layers > 1 else ''}"

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.fc(gru_out)


# ──────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────
_REGISTRY = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
}


def model_factory(
    arch: str = "lstm",
    input_dim: int = 8,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> nn.Module:
    """Instantiate a forecasting model by name.

    Parameters
    ----------
    arch : str
        ``'lstm'`` or ``'gru'``.
    **kwargs
        Forwarded to the model constructor.

    Returns
    -------
    nn.Module
    """
    cls = _REGISTRY.get(arch.lower())
    if cls is None:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(_REGISTRY)}")
    return cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    """Return a concise summary string."""
    lines = [
        f"Model : {getattr(model, 'model_name', model.__class__.__name__)}",
        f"Params: {count_parameters(model):,}",
        "",
    ]
    for name, param in model.named_parameters():
        lines.append(f"  {name:40s}  {str(list(param.shape)):>20s}")
    return "\n".join(lines)
