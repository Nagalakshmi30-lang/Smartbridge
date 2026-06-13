import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Lightweight temporal attention
    Works well on CPU (no heavy matrix ops)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (B, T, 2H)
        scores = self.attn(x)              # (B, T, 1)
        weights = F.softmax(scores, dim=1) # temporal attention
        context = (x * weights).sum(dim=1) # (B, 2H)
        return context, weights


class BridgeBiGRU(nn.Module):
    """
    SMART BRIDGE – BiGRU + Attention + CTC-ready output
    """

    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.attention = Attention(hidden_size)

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        x: (B, T, F)
        returns: (B, T, C) → required for CTC
        """

        gru_out, _ = self.gru(x)        # (B, T, 2H)

        # Frame-wise classification (CTC needs this)
        logits = self.classifier(gru_out)

        return logits
