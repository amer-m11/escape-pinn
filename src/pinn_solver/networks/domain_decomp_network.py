"""Domain-decomposition escape network (FBPINN-style, 3D spherical)."""

import torch
from torch import nn

from .base_network import EscapeNet3D
from .fc_mlp_network import FCMLPNet3D


class DomainDecompNet3D(EscapeNet3D):
    """Partition of unity over ``n_sub³`` spatial :class:`FCMLPNet3D` subnets."""

    def __init__(
        self,
        physical_size: tuple[float, float, float],
        n_sub: int,
        overlap: float = 0.25,
        arch_name: str = "fc-ddm-3d",
        **branch_kwargs,
    ):
        super().__init__()
        if n_sub < 2:
            raise ValueError("n_sub must be >= 2 (or None to disable)")
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.arch_name = arch_name
        self._n_sub = int(n_sub)
        self._overlap = float(overlap)
        L = torch.tensor(physical_size, dtype=torch.float64)
        spacing = L / n_sub  # (3,)
        # Subdomain centres on the n_sub³ grid + the cos²-bump half-widths.
        c1 = (torch.arange(n_sub) + 0.5) * spacing[0]
        c2 = (torch.arange(n_sub) + 0.5) * spacing[1]
        c3 = (torch.arange(n_sub) + 0.5) * spacing[2]
        C1, C2, C3 = torch.meshgrid(c1, c2, c3, indexing="ij")
        centers = torch.stack([C1.reshape(-1), C2.reshape(-1), C3.reshape(-1)], dim=1)
        self.register_buffer("centers", centers.to(torch.float32))  # (K, 3)
        self.register_buffer(
            "half_width", ((0.5 + overlap) * spacing).to(torch.float32)
        )  # (3,)
        branch_kwargs.pop("arch_name", None)
        self.subnets = nn.ModuleList(
            FCMLPNet3D(
                physical_size=physical_size,
                arch_name=f"{arch_name}-sub{i}",
                **branch_kwargs,
            )
            for i in range(centers.shape[0])
        )

    def _weights(self, x1, x2, x3):
        """Partition of unity weights ``W`` with trailing subdomain axis K."""
        x = torch.stack([x1, x2, x3], dim=-1)  # (..., 3)
        # t = (x − c)/h, distance from center c in half-width units h
        t = (x.unsqueeze(-2) - self.centers) / self.half_width  # (..., K, 3)
        inside = t.abs() < 1.0
        bump = torch.where(
            inside, torch.cos(0.5 * torch.pi * t) ** 2, torch.zeros_like(t)
        )  # smooth cos² bump, zero outside the half-width
        omega = bump.prod(dim=-1)  # (..., K)
        return omega / omega.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    def forward(self, x1, x2, x3, phi, theta):
        W = self._weights(x1, x2, x3)  # (..., K)
        outs = [s(x1, x2, x3, phi, theta) for s in self.subnets]
        mixed = tuple(
            sum(W[..., k] * outs[k][c] for k in range(len(self.subnets)))
            for c in range(self.N_OUTPUTS)
        )
        return mixed

    def arch_details(self) -> dict:
        info = self.subnets[0].arch_details()
        info.update(
            arch_name=self.arch_name,
            domain_decomp=self._n_sub,
            domain_decomp_overlap=self._overlap,
            n_subdomains=len(self.subnets),
            n_parameters=sum(p.numel() for p in self.parameters()),
        )
        return info
