"""Selector-head escape network (3D spherical + Cartesian)."""

import torch
from torch import nn

from .base_network import EscapeNet3D, EscapeNetCartesian3D
from .components import build_trunk
from .fc_mlp_network import FCMLPNet3D, FCMLPNetCartesian3D


class _GatedSelectorMixin:
    """Shared learned-gate blend for the spherical/Cartesian selector wrappers."""

    def _init_gate(
        self,
        n_branches: int,
        gate_hidden_layers: int = 2,
        gate_hidden_neurons: int = 32,
        slope_init: float = 4.0,
    ) -> None:
        """Initialize the gate sub-network and slope parameter."""
        if n_branches < 2:
            raise ValueError("selector head needs n_branches >= 2")
        self._n_branches = int(n_branches)
        # Gate sub-net: (x1,x2,x3,d1,d2,d3) -> K logits/scores.
        self.gate = build_trunk(
            "tanh", 6, gate_hidden_layers, gate_hidden_neurons, self._n_branches
        )
        # Learnable slope s = exp(log_s). As s grows the gate sharpens toward a step.
        # Do not learn s directly, as it can become negative and flip the selection.
        self.log_s = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(slope_init))))
        )

    def _mix_gated(self, x1, x2, x3, d1, d2, d3, outs) -> tuple:
        """Infer the gate weights and return the weighted sum of branch outputs."""
        coords = torch.stack([x1, x2, x3, d1, d2, d3], dim=-1)  # (...,6) or (6,)
        logits = self.gate(coords)  # (...,K)
        w = torch.softmax(torch.exp(self.log_s) * logits, dim=-1)
        return tuple(
            sum(w[..., k] * outs[k][c] for k in range(self._n_branches))
            for c in range(self.N_OUTPUTS)
        )

    def _selector_details(self) -> dict:
        """Return a dict of selector architecture details."""
        info = self.branches[0].arch_details()
        info.update(
            arch_name=self.arch_name,
            selector_branches=self._n_branches,
            selector_slope=float(torch.exp(self.log_s).item()),
            n_parameters=sum(p.numel() for p in self.parameters()),
        )
        return info


class SelectorNet3D(_GatedSelectorMixin, EscapeNet3D):
    """Learned-gate blend of ``n_branches`` :class:`FCMLPNet3D` branches (9 outputs)."""

    def __init__(
        self,
        physical_size,
        n_branches: int = 2,
        arch_name: str = "fc-selector-3d",
        **branch_kwargs,
    ):
        super().__init__()
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.arch_name = arch_name
        branch_kwargs.pop("arch_name", None)
        self._init_gate(n_branches)
        self.branches = nn.ModuleList(
            FCMLPNet3D(
                physical_size=physical_size,
                arch_name=f"{arch_name}-branch{i}",
                **branch_kwargs,
            )
            for i in range(self._n_branches)
        )

    def forward(self, x1, x2, x3, phi, theta, cond=None):
        sin_p = torch.sin(phi)
        d1, d2, d3 = sin_p * torch.cos(theta), sin_p * torch.sin(theta), torch.cos(phi)
        outs = [b(x1, x2, x3, phi, theta) for b in self.branches]
        return self._mix_gated(x1, x2, x3, d1, d2, d3, outs)

    def arch_details(self) -> dict:
        return self._selector_details()


class SelectorNetCartesian3D(_GatedSelectorMixin, EscapeNetCartesian3D):
    """Learned-gate blend of ``n_branches`` :class:`FCMLPNetCartesian3D` branches (8 outputs)."""

    def __init__(
        self,
        physical_size,
        n_branches: int = 2,
        arch_name: str = "fc-selector-cart-3d",
        **branch_kwargs,
    ):
        super().__init__()
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.arch_name = arch_name
        branch_kwargs.pop("arch_name", None)
        self._init_gate(n_branches)
        self.branches = nn.ModuleList(
            FCMLPNetCartesian3D(
                physical_size=physical_size,
                arch_name=f"{arch_name}-branch{i}",
                **branch_kwargs,
            )
            for i in range(self._n_branches)
        )

    def forward(self, x1, x2, x3, d1, d2, d3, cond=None):
        outs = [b(x1, x2, x3, d1, d2, d3) for b in self.branches]
        return self._mix_gated(x1, x2, x3, d1, d2, d3, outs)

    def arch_details(self) -> dict:
        return self._selector_details()
