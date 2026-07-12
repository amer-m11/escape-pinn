"""Pluggable MLP trunks for the escape-equation networks.

A *trunk* maps an already-built input feature vector ``(..., D_in)`` to the
raw output channels ``(..., D_out)``. Every trunk here uses only standard
``nn`` ops (no complex tensors, no in-place control flow on the batch dim).

Families (selected by ``cfg.net.trunk``):

* ``"tanh"``: plain ``Linear -> tanh`` MLP (the baseline).
* ``"adaptive_tanh"``: ``tanh(aₗ·x)`` with a learnable per-layer slope ``aₗ``
(Jagtap et al. 2020).
* ``"pirate"``: PirateNet-style (Wang et al. 2024): two-encoder
  multiplicative gating inside gated **residual** blocks with a learnable
  per-block scale ``αₗ`` initialized at 0 (identity at init -> trains deep
  stably). The default high-accuracy trunk.
* ``"siren"``: sine-activated MLP with the ω₀-scaled init (Sitzmann 2020).

``build_trunk`` returns an ``nn.Module``. ``output_dim`` linear readouts are
left at PyTorch defaults (the final layer must not be saturated at init).
"""

import math

import torch
import torch.nn as nn

TRUNK_KINDS = ("tanh", "adaptive_tanh", "pirate", "siren")


class _AdaptiveTanh(nn.Module):
    """``tanh(a·x)`` with a learnable scalar slope ``a`` (init 1.0)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a * x)


def _build_tanh_mlp(
    input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_neurons), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
    layers.append(nn.Linear(hidden_neurons, output_dim))
    return nn.Sequential(*layers)


def _build_adaptive_tanh_mlp(
    input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_neurons), _AdaptiveTanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_neurons, hidden_neurons), _AdaptiveTanh()]
    layers.append(nn.Linear(hidden_neurons, output_dim))
    return nn.Sequential(*layers)


class _PirateBlock(nn.Module):
    """Gated residual block with a learnable scale α (init 0 -> identity)."""

    def __init__(self, hidden_neurons: int) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.f = nn.Linear(hidden_neurons, hidden_neurons)
        self.g = nn.Linear(hidden_neurons, hidden_neurons)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        z = self.act(self.f(h))
        gated = (1.0 - z) * u + z * v
        delta = self.act(self.g(gated))
        return h + self.alpha * delta


class _PirateNet(nn.Module):
    """PirateNet-style trunk: modified-MLP gating in α-scaled residual blocks."""

    def __init__(
        self, input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int
    ) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.U = nn.Linear(input_dim, hidden_neurons)
        self.V = nn.Linear(input_dim, hidden_neurons)
        self.in_layer = nn.Linear(input_dim, hidden_neurons)
        self.blocks = nn.ModuleList(
            _PirateBlock(hidden_neurons) for _ in range(max(1, hidden_layers - 1))
        )
        self.out = nn.Linear(hidden_neurons, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.U(x))
        v = self.act(self.V(x))
        h = self.act(self.in_layer(x))
        for block in self.blocks:
            h = block(h, u, v)
        return self.out(h)


class _PerChannelHeads(nn.Module):
    """Shared trunk body -> ``output_dim`` independent ``Linear(hidden,1)`` heads.

    The body is any trunk built to emit ``hidden_neurons`` features. A final
    ``tanh`` then feeds one small head per output channel, so channels of
    differing regularity (û kinked vs cos/sin oscillatory) get dedicated
    readouts instead of sharing a single last layer.
    """

    def __init__(self, body: nn.Module, hidden_neurons: int, output_dim: int):
        super().__init__()
        self.body = body
        self.act = nn.Tanh()
        self.heads = nn.ModuleList(
            nn.Linear(hidden_neurons, 1) for _ in range(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.body(x))
        return torch.cat([head(h) for head in self.heads], dim=-1)


class _Sine(nn.Module):
    """``sin(ω₀·x)`` activation (SIREN). ``ω₀`` sets the trunk's frequency band."""

    def __init__(self, w0: float) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def _siren_linear(in_dim: int, out_dim: int, w0: float, first: bool) -> nn.Linear:
    lin = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        if first:
            lin.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
        else:
            b = math.sqrt(6.0 / in_dim) / w0
            lin.weight.uniform_(-b, b)
    return lin


def _build_siren_mlp(
    input_dim, hidden_layers, hidden_neurons, out_dim, w0: float = 30.0
):
    """SIREN sine-activated MLP with the standard ω₀-scaled init."""
    layers: list[nn.Module] = [
        _siren_linear(input_dim, hidden_neurons, w0, True),
        _Sine(w0),
    ]
    for _ in range(hidden_layers - 1):
        layers += [_siren_linear(hidden_neurons, hidden_neurons, w0, False), _Sine(w0)]
    layers.append(_siren_linear(hidden_neurons, out_dim, w0, False))
    return nn.Sequential(*layers)


def _build_body(kind, input_dim, hidden_layers, hidden_neurons, out_dim):
    if kind == "tanh":
        return _build_tanh_mlp(input_dim, hidden_layers, hidden_neurons, out_dim)
    if kind == "adaptive_tanh":
        return _build_adaptive_tanh_mlp(
            input_dim, hidden_layers, hidden_neurons, out_dim
        )
    if kind == "pirate":
        return _PirateNet(input_dim, hidden_layers, hidden_neurons, out_dim)
    if kind == "siren":
        return _build_siren_mlp(input_dim, hidden_layers, hidden_neurons, out_dim)
    raise ValueError(f"unknown trunk kind: {kind!r} (one of {TRUNK_KINDS})")


def build_trunk(
    kind: str,
    input_dim: int,
    hidden_layers: int,
    hidden_neurons: int,
    output_dim: int,
    per_channel_heads: bool = False,
) -> nn.Module:
    """Construct a trunk module ``(..., input_dim) -> (..., output_dim)``.

    ``per_channel_heads`` wraps the trunk body (emitting ``hidden_neurons``
    features) with one independent linear head per output channel.
    """
    if per_channel_heads:
        body = _build_body(
            kind, input_dim, hidden_layers, hidden_neurons, hidden_neurons
        )
        return _PerChannelHeads(body, hidden_neurons, output_dim)
    return _build_body(kind, input_dim, hidden_layers, hidden_neurons, output_dim)
