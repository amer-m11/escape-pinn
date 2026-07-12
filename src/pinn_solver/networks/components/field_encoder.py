"""DeepONet-style field encoder for the slowness-conditioned PINN."""

from __future__ import annotations

import torch
from torch import nn


class FieldEncoder(nn.Module):
    """``(..., n_probe_points) -> (..., latent_dim)`` tanh-MLP branch net.

    Input is the per-point probe-grid slowness *perturbation* ``n(probe)-n0``
    (≈0 in the constant medium). Output is the conditioning latent.
    """

    def __init__(
        self,
        n_probe_points: int,
        latent_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
    ):
        super().__init__()
        self.n_probe_points = int(n_probe_points)
        self.latent_dim = int(latent_dim)
        layers: list[nn.Module] = []
        d = self.n_probe_points
        for _ in range(max(1, depth)):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, self.latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)
