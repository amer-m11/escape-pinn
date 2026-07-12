"""Tanh-MLP networks for the Eikonal escape equations (2D and 3D variants).

Both variants optionally apply a random-Fourier-feature spatial encoding
lifting (x₁, ...) to a high-frequency basis y(v) = [sin(2π Bv), cos(2π Bv)]
before the tanh MLP. Angle inputs are passed through unchanged as (cos, sin)
embeddings.
"""

import math

import torch

from .base_network import (
    EscapeNet2D,
    EscapeNet3D,
    EscapeNetCartesian3D,
)
from .components import build_trunk
from .components.escape_background import _calculate_straight_exit_time
from .components.field_encoder import FieldEncoder


def _exit_time_feature_vector(
    x1, x2, x3, d1, d2, d3, L, with_sign: bool = False
) -> torch.Tensor:
    """The exit-time trunk input block, shared by the spherical & Cartesian nets.

    Builds the 3 per-axis straight-ray exit times + their min (the n=1 escape
    time), normalized by the box diagonal and clamped to ``[0, 1]`` (+∞  maps to 1).

    ``with_sign=True``: also append the per-axis ``sign(dᵢ)``.
    """

    diag = math.sqrt(L[0] ** 2 + L[1] ** 2 + L[2] ** 2)
    t1 = _calculate_straight_exit_time(x1, L[0], d1)
    t2 = _calculate_straight_exit_time(x2, L[1], d2)
    t3 = _calculate_straight_exit_time(x3, L[2], d3)
    tm = torch.minimum(torch.minimum(t1, t2), t3)
    times = torch.stack([t1, t2, t3, tm], dim=-1)
    base = (times / diag).clamp(0.0, 1.0)
    if not with_sign:
        return base
    signs = torch.stack([torch.sign(d1), torch.sign(d2), torch.sign(d3)], dim=-1)
    return torch.cat([base, signs], dim=-1)


class FCMLPNet2D(EscapeNet2D):
    """Fully-connected tanh MLP on a 2D physical domain.

    Inputs: ``(x1, x2, theta)`` -> 4-D ``(x1_norm, x2_norm, cos θ, sin θ)``.
    Outputs: 6-D ``(û, σ̂, ŷ₁, ŷ₂, cos θ̂, sin θ̂)``.
    """

    def __init__(
        self,
        physical_size: tuple[float, float],
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        arch_name: str = "fc-tanh",
        fourier_n_features: int | None = None,
        fourier_sigma: float | None = None,
        trunk: str = "tanh",
        unit_norm_output: bool = False,
        factored_eikonal: bool = False,
        per_channel_heads: bool = False,
    ):
        """
        Parameters
        ----------
        physical_size
            ``(L_x1, L_x2)`` domain extents used for input normalization.
        hidden_layers
            Number of hidden layers (default 4).
        hidden_neurons
            Neurons per hidden layer (default 64).
        arch_name
            Short human-readable identifier string.
        fourier_n_features, fourier_sigma
            When both are set, a random-Fourier-feature spatial encoding
            with ``fourier_n_features`` directions and Gaussian B-matrix
            scale ``fourier_sigma`` is prepended.
        """
        super().__init__()
        self.L_x1, self.L_x2 = physical_size
        self._trunk = trunk
        self.unit_norm_output = unit_norm_output
        self.factored_eikonal = factored_eikonal
        self._per_channel_heads = per_channel_heads
        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self.arch_name = arch_name

        if (fourier_n_features is None) != (fourier_sigma is None):
            raise ValueError(
                "fourier_n_features and fourier_sigma must be set together "
                "or both left None"
            )
        self._fourier_n_features = fourier_n_features
        self._fourier_sigma = fourier_sigma

        if fourier_n_features is not None:
            assert fourier_sigma is not None
            b_matrix = torch.randn(fourier_n_features, 2) * fourier_sigma
            self.register_buffer("fourier_B", b_matrix, persistent=True)
            # (sin+cos)·n_feat spatial + (cos θ, sin θ) = 2·n_feat + 2
            input_dim = 2 * fourier_n_features + 2
        else:
            self.fourier_B = None
            input_dim = 4

        self._input_dim = input_dim
        self.net = build_trunk(
            self._trunk,
            input_dim,
            hidden_layers,
            hidden_neurons,
            self.N_OUTPUTS,
            per_channel_heads=per_channel_heads,
        )

    def arch_details(self) -> dict:
        info = {
            "arch_name": self.arch_name,
            "hidden_layers": self._hidden_layers,
            "hidden_neurons": self._hidden_neurons,
            "n_outputs": self.N_OUTPUTS,
            "input_dim": self._input_dim,
            "activation": "tanh",
            "trunk": str(self._trunk),
            "unit_norm_output": bool(self.unit_norm_output),
            "factored_eikonal": bool(self.factored_eikonal),
            "per_channel_heads": bool(self._per_channel_heads),
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }
        if self._fourier_n_features is not None:
            assert self._fourier_sigma is not None
            info["fourier_features"] = {
                "n_features": int(self._fourier_n_features),
                "sigma": float(self._fourier_sigma),
            }
        return info

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

        if self.fourier_B is not None:
            spatial = torch.stack([x1_norm, x2_norm], dim=-1)
            projection = 2.0 * math.pi * (spatial @ self.fourier_B.T)
            inp = torch.cat(
                [
                    torch.sin(projection),
                    torch.cos(projection),
                    cos_theta.unsqueeze(-1),
                    sin_theta.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            inp = torch.stack([x1_norm, x2_norm, cos_theta, sin_theta], dim=-1)

        out = self.net(inp)
        if self.factored_eikonal:
            u, sigma = self.factored_uv(out[..., 0], out[..., 1], x1, x2, theta)
        else:
            u, sigma = out[..., 0], out[..., 1]
        return self.apply_unit_norm_if_enabled(u, sigma, out)


class FCMLPNet3D(EscapeNet3D):
    """Fully-connected tanh MLP on a 3D physical domain.

    Inputs: ``(x1, x2, x3, phi, theta)`` -> 7-D
    ``(x1_norm, x2_norm, x3_norm, cos θ, sin θ, cos φ, sin φ)``. Outputs: 9-D
    ``(û, σ̂, ŷ₁, ŷ₂, ŷ₃, cos θ̂, sin θ̂, cos φ̂, sin φ̂)``.
    """

    def __init__(
        self,
        physical_size: tuple[float, float, float],
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        arch_name: str = "fc-tanh-3d",
        fourier_n_features: int | None = None,
        fourier_sigma: float | None = None,
        fourier_angular: bool = False,
        trunk: str = "tanh",
        unit_norm_output: bool = False,
        factored_eikonal: bool = False,
        per_channel_heads: bool = False,
        exit_time_features: bool = False,
        exit_time_sign: bool = False,
        conditioning_dim: int = 0,
        field_encoder: bool = False,
        encoder_latent_dim: int = 16,
        encoder_hidden: int = 64,
        encoder_depth: int = 2,
    ):
        super().__init__()
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.conditioning_dim = int(conditioning_dim)
        # field encoder: when on, the raw conditioning vector (a probe-grid
        # slowness field of width ``conditioning_dim``) is mapped to a latent before
        # the trunk. The trunk input then widens by ``latent_dim`` (not the grid).
        self._field_encoder = None
        if field_encoder and self.conditioning_dim > 0:

            self._field_encoder = FieldEncoder(
                self.conditioning_dim, encoder_latent_dim, encoder_hidden, encoder_depth
            )
        self._trunk = trunk
        self.unit_norm_output = unit_norm_output
        self.factored_eikonal = factored_eikonal
        self._per_channel_heads = per_channel_heads
        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self.arch_name = arch_name
        self.exit_time_features = exit_time_features
        self.exit_time_sign = bool(exit_time_sign)

        if (fourier_n_features is None) != (fourier_sigma is None):
            raise ValueError(
                "fourier_n_features and fourier_sigma must be set together "
                "or both left None"
            )
        self._fourier_n_features = fourier_n_features
        self._fourier_sigma = fourier_sigma
        self._fourier_angular = bool(fourier_angular)

        if fourier_n_features is not None:
            assert fourier_sigma is not None
            b_matrix = torch.randn(fourier_n_features, 3) * fourier_sigma
            self.register_buffer("fourier_B", b_matrix, persistent=True)
            if self._fourier_angular:
                # Lift the (cos θ, sin θ, cos φ, sin φ) embedding
                b_angular = torch.randn(fourier_n_features, 4) * fourier_sigma
                self.register_buffer("fourier_B_angular", b_angular, persistent=True)
                input_dim = 4 * fourier_n_features
            else:
                # Conventional: spatial FFs + raw angular embedding (4).
                self.fourier_B_angular = None
                input_dim = 2 * fourier_n_features + 4
        else:
            self.fourier_B = None
            self.fourier_B_angular = None
            input_dim = 7
        if exit_time_features:
            # 3 per axis straight ray exit times + their min, diagonal-normalized.
            input_dim += 4 + (3 if exit_time_sign else 0)

        if getattr(self, "conditioning_dim", 0):
            input_dim += (
                self._field_encoder.latent_dim
                if getattr(self, "_field_encoder", None) is not None
                else self.conditioning_dim
            )
        self._input_dim = input_dim
        self.net = build_trunk(
            self._trunk,
            input_dim,
            hidden_layers,
            hidden_neurons,
            self.N_OUTPUTS,
            per_channel_heads=per_channel_heads,
        )

    def arch_details(self) -> dict:
        info = {
            "arch_name": self.arch_name,
            "hidden_layers": self._hidden_layers,
            "hidden_neurons": self._hidden_neurons,
            "n_outputs": self.N_OUTPUTS,
            "input_dim": self._input_dim,
            "activation": "tanh",
            "trunk": str(self._trunk),
            "unit_norm_output": bool(self.unit_norm_output),
            "factored_eikonal": bool(self.factored_eikonal),
            "per_channel_heads": bool(self._per_channel_heads),
            "exit_time_features": bool(self.exit_time_features),
            "exit_time_sign": bool(self.exit_time_sign),
            "conditioning_dim": int(self.conditioning_dim),
            "field_encoder": self._field_encoder is not None,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }
        if self._fourier_n_features is not None:
            assert self._fourier_sigma is not None
            info["fourier_features"] = {
                "n_features": int(self._fourier_n_features),
                "sigma": float(self._fourier_sigma),
                "angular": bool(self._fourier_angular),
            }
        return info

    def _exit_time_inputs(self, x1, x2, x3, phi, theta) -> torch.Tensor:
        """Exit-time trunk inputs along the spherical direction ``d = (sinφ cosθ,
        sinφ sinθ, cosφ)``. Delegates to :func:`_exit_time_feature_vector`."""
        sin_p = torch.sin(phi)
        d1 = sin_p * torch.cos(theta)
        d2 = sin_p * torch.sin(theta)
        d3 = torch.cos(phi)
        return _exit_time_feature_vector(
            x1,
            x2,
            x3,
            d1,
            d2,
            d3,
            (self.L_x1, self.L_x2, self.L_x3),
            with_sign=self.exit_time_sign,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        phi: torch.Tensor,
        theta: torch.Tensor,
        cond: torch.Tensor | None = None,
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

        if self.fourier_B is not None:
            spatial = torch.stack([x1_norm, x2_norm, x3_norm], dim=-1)
            proj_s = 2.0 * math.pi * (spatial @ self.fourier_B.T)
            sin_s, cos_s = torch.sin(proj_s), torch.cos(proj_s)
            if self._fourier_angular:
                # Project the (cos,sin) embedding 4-vector.
                angular = torch.stack([cos_theta, sin_theta, cos_phi, sin_phi], dim=-1)
                assert self.fourier_B_angular is not None
                proj_a = 2.0 * math.pi * (angular @ self.fourier_B_angular.T)
                inp = torch.cat(
                    [sin_s, cos_s, torch.sin(proj_a), torch.cos(proj_a)],
                    dim=-1,
                )
            else:
                # Spatial FFs + raw angular embedding (conventional).
                inp = torch.cat(
                    [
                        sin_s,
                        cos_s,
                        cos_theta.unsqueeze(-1),
                        sin_theta.unsqueeze(-1),
                        cos_phi.unsqueeze(-1),
                        sin_phi.unsqueeze(-1),
                    ],
                    dim=-1,
                )
        else:
            inp = torch.stack(
                [x1_norm, x2_norm, x3_norm, cos_theta, sin_theta, cos_phi, sin_phi],
                dim=-1,
            )

        if self.exit_time_features:
            inp = torch.cat(
                [inp, self._exit_time_inputs(x1, x2, x3, phi, theta)], dim=-1
            )

        if cond is not None:
            c = self._field_encoder(cond) if self._field_encoder is not None else cond
            inp = torch.cat([inp, c], dim=-1)

        out = self.net(inp)
        ru, rs = out[..., 0], out[..., 1]
        if self.factored_eikonal:
            u, sigma = self.factored_uv(ru, rs, x1, x2, x3, phi, theta)
        else:
            u, sigma = ru, rs
        return self.apply_unit_norm(u, sigma, out)


class FCMLPNetCartesian3D(EscapeNetCartesian3D):
    """Fully-connected tanh MLP on a 3D domain with Cartesian momentum.

    Inputs: ``(x1, x2, x3, d1, d2, d3)`` -> 6-D
    ``(x1_norm, x2_norm, x3_norm, d1, d2, d3)``. Outputs:
    8-D ``(û, σ̂, ŷ₁, ŷ₂, ŷ₃, d̂₁, d̂₂, d̂₃)``.
    """

    def __init__(
        self,
        physical_size: tuple[float, float, float],
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        arch_name: str = "fc-tanh-cart-3d",
        fourier_n_features: int | None = None,
        fourier_sigma: float | None = None,
        trunk: str = "tanh",
        unit_norm_output: bool = False,
        factored_eikonal: bool = False,
        per_channel_heads: bool = False,
        exit_time_features: bool = False,
        exit_time_sign: bool = False,
        conditioning_dim: int = 0,
        field_encoder: bool = False,
        encoder_latent_dim: int = 16,
        encoder_hidden: int = 64,
        encoder_depth: int = 2,
    ):
        super().__init__()
        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.conditioning_dim = int(conditioning_dim)
        # field encoder: when on, the raw conditioning vector (a probe-grid
        # slowness field of width ``conditioning_dim``) is mapped to a latent before
        # the trunk. The trunk input then widens by ``latent_dim`` (not the grid).
        self._field_encoder = None
        if field_encoder and self.conditioning_dim > 0:
            from .components.field_encoder import FieldEncoder

            self._field_encoder = FieldEncoder(
                self.conditioning_dim, encoder_latent_dim, encoder_hidden, encoder_depth
            )
        self._trunk = trunk
        self.unit_norm_output = unit_norm_output
        self.factored_eikonal = factored_eikonal
        self._per_channel_heads = per_channel_heads
        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self.arch_name = arch_name
        self.exit_time_features = exit_time_features
        self.exit_time_sign = bool(exit_time_sign)

        if (fourier_n_features is None) != (fourier_sigma is None):
            raise ValueError(
                "fourier_n_features and fourier_sigma must be set together "
                "or both left None"
            )
        self._fourier_n_features = fourier_n_features
        self._fourier_sigma = fourier_sigma

        if fourier_n_features is not None:
            assert fourier_sigma is not None
            b_matrix = torch.randn(fourier_n_features, 3) * fourier_sigma
            self.register_buffer("fourier_B", b_matrix, persistent=True)
            # (sin+cos)·n_feat spatial + (d1, d2, d3) = 2·n_feat + 3
            input_dim = 2 * fourier_n_features + 3
        else:
            self.fourier_B = None
            input_dim = 6
        if exit_time_features:
            # 3 per-axis exit times + their min, diagonal-normalized.
            input_dim += 4 + (3 if exit_time_sign else 0)

        if getattr(self, "conditioning_dim", 0):
            input_dim += (
                self._field_encoder.latent_dim
                if getattr(self, "_field_encoder", None) is not None
                else self.conditioning_dim
            )
        self._input_dim = input_dim
        self.net = build_trunk(
            self._trunk,
            input_dim,
            hidden_layers,
            hidden_neurons,
            self.N_OUTPUTS,
            per_channel_heads=per_channel_heads,
        )

    def arch_details(self) -> dict:
        info = {
            "arch_name": self.arch_name,
            "hidden_layers": self._hidden_layers,
            "hidden_neurons": self._hidden_neurons,
            "n_outputs": self.N_OUTPUTS,
            "input_dim": self._input_dim,
            "activation": "tanh",
            "trunk": str(self._trunk),
            "unit_norm_output": bool(self.unit_norm_output),
            "factored_eikonal": bool(self.factored_eikonal),
            "per_channel_heads": bool(self._per_channel_heads),
            "exit_time_features": bool(self.exit_time_features),
            "exit_time_sign": bool(self.exit_time_sign),
            "conditioning_dim": int(self.conditioning_dim),
            "field_encoder": self._field_encoder is not None,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }
        if self._fourier_n_features is not None:
            assert self._fourier_sigma is not None
            info["fourier_features"] = {
                "n_features": int(self._fourier_n_features),
                "sigma": float(self._fourier_sigma),
            }
        return info

    def _exit_time_inputs(self, x1, x2, x3, d1, d2, d3) -> torch.Tensor:
        """Exit-time trunk inputs along the Cartesian direction ``d``. Delegates to
        :func:`_exit_time_feature_vector`."""
        return _exit_time_feature_vector(
            x1,
            x2,
            x3,
            d1,
            d2,
            d3,
            (self.L_x1, self.L_x2, self.L_x3),
            with_sign=self.exit_time_sign,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        d3: torch.Tensor,
        cond: torch.Tensor | None = None,
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

        if self.fourier_B is not None:
            spatial = torch.stack([x1_norm, x2_norm, x3_norm], dim=-1)
            proj_s = 2.0 * math.pi * (spatial @ self.fourier_B.T)
            sin_s, cos_s = torch.sin(proj_s), torch.cos(proj_s)
            inp = torch.cat(
                [
                    sin_s,
                    cos_s,
                    d1.unsqueeze(-1),
                    d2.unsqueeze(-1),
                    d3.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            inp = torch.stack([x1_norm, x2_norm, x3_norm, d1, d2, d3], dim=-1)

        if self.exit_time_features:
            inp = torch.cat(
                [inp, self._exit_time_inputs(x1, x2, x3, d1, d2, d3)], dim=-1
            )

        if cond is not None:
            c = self._field_encoder(cond) if self._field_encoder is not None else cond
            inp = torch.cat([inp, c], dim=-1)

        out = self.net(inp)
        ru, rs = out[..., 0], out[..., 1]
        if self.factored_eikonal:
            u, sigma = self.factored_uv(ru, rs, x1, x2, x3, d1, d2, d3)
        else:
            u, sigma = ru, rs
        return self.apply_unit_norm(u, sigma, out)
