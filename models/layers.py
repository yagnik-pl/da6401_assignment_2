"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Inverted dropout implemented without torch.nn.Dropout."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must lie in [0, 1], got {p}.")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)

        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(dtype=x.dtype)
        return x * mask / keep_prob
