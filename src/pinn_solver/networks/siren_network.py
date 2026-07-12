"""SIREN architecture (Sitzmann et al. 2020) for the escape equations.

Sine activations + ω₀-scaled init give the network a high-frequency
representation basis while pure tanh has a smoothing effect.
"""

import math

import torch
import torch.nn as nn

from .base_network import (
    EscapeNet2D,
    EscapeNet3D,
    EscapeNetCartesian3D,
)


class Sine(nn.Module):
    """``y = sin(ω₀ · x)`` with ω₀ stored as a plain Python float (not trainable).

    Keeping ω₀ as a scalar (not a buffer) avoids broadcasting surprises
    under ``torch.func.vmap`` and keeps the module stateless.
    """

    def __init__(self, omega_0: float):
        super().__init__()
        self.omega_0 = float(omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


def _siren_init_(layers: list[nn.Linear], omega_0: float) -> None:
    """Apply the canonical SIREN init in-place.

    First Linear: weight ~ uniform(-1/d_in, 1/d_in).
    Hidden Linears: weight ~ uniform(-sqrt(6/d_in)/ω₀, +sqrt(6/d_in)/ω₀).
    Final Linear is left at the PyTorch default (paper convention).
    All biases are zeroed.
    """
    if not layers:
        return
    if len(layers) == 1:
        nn.init.zeros_(layers[0].bias)
        return

    first, *hidden_then_last = layers
    hidden = hidden_then_last[:-1]
    last = hidden_then_last[-1]

    d_in = first.in_features
    with torch.no_grad():
        first.weight.uniform_(-1.0 / d_in, 1.0 / d_in)
        nn.init.zeros_(first.bias)

    for layer in hidden:
        d_in = layer.in_features
        bound = math.sqrt(6.0 / d_in) / omega_0
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)
            nn.init.zeros_(layer.bias)

    nn.init.zeros_(last.bias)


def _build_siren_mlp(
    input_dim: int,
    hidden_layers: int,
    hidden_neurons: int,
    output_dim: int,
    omega_0: float,
) -> nn.Sequential:
    """``Linear -> Sine`` repeating block followed by a final Linear readout."""
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_neurons), Sine(omega_0)]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_neurons, hidden_neurons), Sine(omega_0)]
    layers.append(nn.Linear(hidden_neurons, output_dim))
    net = nn.Sequential(*layers)
    linear_layers = [m for m in net if isinstance(m, nn.Linear)]
    _siren_init_(linear_layers, omega_0)
    return net


class SIRENNet2D(EscapeNet2D):
    """Fully-connected SIREN on a 2D physical domain.

    Input: ``(x1, x2, theta)`` -> 4-D ``(x1_norm, x2_norm, cos θ, sin θ)``.
    Output: 6-D ``(û, σ̂, ŷ₁, ŷ₂, cos θ̂, sin θ̂)``.
    """

    def __init__(
        self,
        physical_size: tuple[float, float],
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        omega_0: float = 30.0,
        arch_name: str = "siren",
        unit_norm_output: bool = False,
        factored_eikonal: bool = False,
    ):
        super().__init__()
        self.L_x1, self.L_x2 = physical_size
        self.unit_norm_output = unit_norm_output
        self.factored_eikonal = factored_eikonal
        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self._omega_0 = float(omega_0)
        self.arch_name = arch_name
        self.net = _build_siren_mlp(
            input_dim=4,
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            output_dim=self.N_OUTPUTS,
            omega_0=self._omega_0,
        )

    def arch_details(self) -> dict:
        return {
            "arch_name": self.arch_name,
            "hidden_layers": self._hidden_layers,
            "hidden_neurons": self._hidden_neurons,
            "n_outputs": self.N_OUTPUTS,
            "input_dim": 4,
            "activation": "sin",
            "omega_0": self._omega_0,
            "unit_norm_output": bool(self.unit_norm_output),
            "factored_eikonal": bool(self.factored_eikonal),
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, theta: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x1_norm = 2.0 * x1 / self.L_x1 - 1.0
        x2_norm = 2.0 * x2 / self.L_x2 - 1.0
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        inp = torch.stack([x1_norm, x2_norm, cos_theta, sin_theta], dim=-1)
        out = self.net(inp)
        if self.factored_eikonal:
            u, sigma = self.factored_uv(out[..., 0], out[..., 1], x1, x2, theta)
        else:
            u, sigma = out[..., 0], out[..., 1]
        return self.apply_unit_norm_if_enabled(u, sigma, out)


class SIRENNet3D(EscapeNet3D):
    """Fully-connected SIREN on a 3D physical domain.

    Input: ``(x1, x2, x3, phi, theta)`` -> 7-D
    ``(x1_norm, x2_norm, x3_norm, cos θ, sin θ, cos φ, sin φ)``.
    Output: 8-D ``(û, σ̂, ŷ₁, ŷ₂, ŷ₃, cos θ̂, sin θ̂, cos φ̂, sin φ̂)``.
    """

    def __init__(
        self,
        physical_size: tuple[float, float, float],
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        omega_0: float = 30.0,
        arch_name: str = "siren-3d",
        unit_norm_output: bool = False,
        factored_eikonal: bool = False,
    ):
        super().__init__()
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.unit_norm_output = unit_norm_output
        self.factored_eikonal = factored_eikonal
        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self._omega_0 = float(omega_0)
        self.arch_name = arch_name
        self.net = _build_siren_mlp(
            input_dim=7,
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            output_dim=self.N_OUTPUTS,
            omega_0=self._omega_0,
        )

    def arch_details(self) -> dict:
        return {
            "arch_name": self.arch_name,
            "hidden_layers": self._hidden_layers,
            "hidden_neurons": self._hidden_neurons,
            "n_outputs": self.N_OUTPUTS,
            "input_dim": 7,
            "activation": "sin",
            "omega_0": self._omega_0,
            "unit_norm_output": bool(self.unit_norm_output),
            "factored_eikonal": bool(self.factored_eikonal),
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        phi: torch.Tensor,
        theta: torch.Tensor,
        cond=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x1_norm = 2.0 * x1 / self.L_x1 - 1.0
        x2_norm = 2.0 * x2 / self.L_x2 - 1.0
        x3_norm = 2.0 * x3 / self.L_x3 - 1.0
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        inp = torch.stack(
            [x1_norm, x2_norm, x3_norm, cos_theta, sin_theta, cos_phi, sin_phi],
            dim=-1,
        )
        out = self.net(inp)
        if self.factored_eikonal:
            u, sigma = self.factored_uv(
                out[..., 0], out[..., 1], x1, x2, x3, phi, theta
            )
        else:
            u, sigma = out[..., 0], out[..., 1]
        return self.apply_unit_norm(u, sigma, out)


class SIRENNetCartesian3D(EscapeNetCartesian3D):
    """Fully-connected SIREN on a 3D domain with Cartesian momentum.

    Input: ``(x1, x2, x3, d1, d2, d3)`` -> 6-D
    ``(x1_norm, x2_norm, x3_norm, d1, d2, d3)``.
    Output: 8-D ``(û, σ̂, ŷ₁, ŷ₂, ŷ₃, d̂₁, d̂₂, d̂₃)``.
    """

    def __init__(
        self,
        physical_size: tuple[float, float, float],
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        omega_0: float = 30.0,
        arch_name: str = "siren-cart-3d",
        unit_norm_output: bool = False,
        factored_eikonal: bool = False,
    ):
        super().__init__()
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.unit_norm_output = unit_norm_output
        self.factored_eikonal = factored_eikonal
        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self._omega_0 = float(omega_0)
        self.arch_name = arch_name
        self.net = _build_siren_mlp(
            input_dim=6,
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            output_dim=self.N_OUTPUTS,
            omega_0=self._omega_0,
        )

    def arch_details(self) -> dict:
        return {
            "arch_name": self.arch_name,
            "hidden_layers": self._hidden_layers,
            "hidden_neurons": self._hidden_neurons,
            "n_outputs": self.N_OUTPUTS,
            "input_dim": 6,
            "activation": "sin",
            "omega_0": self._omega_0,
            "unit_norm_output": bool(self.unit_norm_output),
            "factored_eikonal": bool(self.factored_eikonal),
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        d3: torch.Tensor,
        cond=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x1_norm = 2.0 * x1 / self.L_x1 - 1.0
        x2_norm = 2.0 * x2 / self.L_x2 - 1.0
        x3_norm = 2.0 * x3 / self.L_x3 - 1.0

        inp = torch.stack([x1_norm, x2_norm, x3_norm, d1, d2, d3], dim=-1)
        out = self.net(inp)
        if self.factored_eikonal:
            u, sigma = self.factored_uv(
                out[..., 0], out[..., 1], x1, x2, x3, d1, d2, d3
            )
        else:
            u, sigma = out[..., 0], out[..., 1]
        return self.apply_unit_norm(u, sigma, out)
