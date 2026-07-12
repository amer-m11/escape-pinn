"""Base classes for escape equation networks.

Each dimensionality/representation variant (2D, 3D spherical, 3D Cartesian) has
its own abstract base type, which concrete subclasses implement.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .components.escape_background import (
    background_escape_2d,
    background_escape_3d,
    background_escape_cartesian_3d,
)


class EscapeNet(nn.Module, ABC):
    """Abstract base type for every escape equation network.

    Concrete subclasses bind to a specific physical dim (2D or 3D) through
    the corresponding :class:`EscapeNet2D` / :class:`EscapeNet3D` subclass.
    ``N_OUTPUTS`` is set by those subclasses.

    TODO: find a better way to structure this
    """

    N_OUTPUTS: int

    @abstractmethod
    def arch_details(self) -> dict:
        """Return a JSON-serializable dict describing the network
        architecture.

        At minimum the dict must include:

        * ``n_parameters``: Total trainable scalar count.
        * ``arch_name``: Short human-readable identifier string.

        Architecture specific extras are free form (e.g. ``hidden_layers``,
        ``hidden_neurons`` for an MLP).
        """


class EscapeNet2D(EscapeNet):
    """Escape equation network on a 2D physical domain (3D phase space).

    Solves the five escape quantities

        û    - escape travel time
        σ̂    - parametric escape distance
        ŷ1   - exit x1-position
        ŷ2   - exit x2-position
        θ̂    - exit angle, embedded internally as (cos θ̂, sin θ̂)

    Concrete subclasses' ``forward(x1, x2, theta)`` returns the six
    components in canonical order:

        û, σ̂, ŷ₁, ŷ₂, cos θ̂, sin θ̂
    """

    N_OUTPUTS: int = 6

    L_x1: float
    L_x2: float
    unit_norm_output: bool = False
    factored_eikonal: bool = False

    def factored_uv(self, raw_u, raw_sigma, x1, x2, theta):
        """û, σ̂ for the factored-Eikonal ansatz: analytic straight ray
        distance-to-∂D background + the raw network correction."""

        bg = background_escape_2d(x1, x2, theta, (self.L_x1, self.L_x2))
        return bg + raw_u, bg + raw_sigma

    def apply_unit_norm_if_enabled(self, u, sigma, out):
        """Final 6-tuple, applying ``unit_norm_output`` to (cos θ̂, sin θ̂).
        no-op if ``unit_norm_output=False``."""
        c_th, s_th = out[..., 4], out[..., 5]
        if self.unit_norm_output:
            r = torch.sqrt(c_th * c_th + s_th * s_th).clamp_min(1e-12)
            c_th, s_th = c_th / r, s_th / r
        return (u, sigma, out[..., 2], out[..., 3], c_th, s_th)

    @abstractmethod
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Evaluate the network at one or many phase-space points.

        Must be shape-polymorphic:
        - 1-D batched tensors -> six 1-D tensors of the same length
        - 0-D scalars -> six 0-D scalars.

        Parameters
        ----------
        x1, x2, theta : torch.Tensor
            Phase-space coordinates. x1 and x2 will be normalized to
            [-1, 1] internally so spatial inputs have the same magnitude
            regardless of domain size.
        """

    def predict_theta_exit(
        self, x1: torch.Tensor, x2: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Recover the scalar exit angle θ̂ ∈ [0, 2π) by applying ``atan2``
        to the (cos, sin) pair and shifting negative angles by 2π.
        """
        _, _, _, _, c, s = self(x1, x2, theta)
        angle = torch.atan2(s, c)
        return torch.where(angle < 0, angle + 2 * math.pi, angle)


class EscapeNet3D(EscapeNet):
    """Escape equation network on a 3D physical domain (5D phase space).

    Solves the seven escape quantities (û, σ̂, ŷ₁, ŷ₂, ŷ₃, θ̂, φ̂). Concrete
    subclasses emit nine output channels because each exit angle is
    embedded as a (cos, sin) pair.

    ``forward(x1, x2, x3, phi, theta)`` returns the nine components in
    canonical order:

        û, σ̂, ŷ₁, ŷ₂, ŷ₃, cos θ̂, sin θ̂, cos φ̂, sin φ̂

    The φ-embedding mirrors θ. φ ∈ [0, π] is non-periodic, the
    (cos, sin) pair keeps the network output smooth across the pole.
    """

    N_OUTPUTS: int = 9

    # Populated by concrete subclasses.
    L_x1: float
    L_x2: float
    L_x3: float
    unit_norm_output: bool = False
    factored_eikonal: bool = False

    def factored_uv(self, raw_u, raw_sigma, x1, x2, x3, phi, theta):
        """û, σ̂ for the factored-Eikonal ansatz."""

        bg = background_escape_3d(
            x1, x2, x3, phi, theta, (self.L_x1, self.L_x2, self.L_x3)
        )
        return bg + raw_u, bg + raw_sigma

    def apply_unit_norm(self, u, sigma, out):
        """Final 9-tuple, applying ``unit_norm_output`` to the (cos,sin) pairs of θ
        (channels 5,6) and φ̂ (channels 7,8). No-op if ``unit_norm_output=False``."""
        cth, sth, cph, sph = out[..., 5], out[..., 6], out[..., 7], out[..., 8]
        if self.unit_norm_output:
            rt = torch.sqrt(cth * cth + sth * sth).clamp_min(1e-12)
            cth, sth = cth / rt, sth / rt
            rp = torch.sqrt(cph * cph + sph * sph).clamp_min(1e-12)
            cph, sph = cph / rp, sph / rp
        return (u, sigma, out[..., 2], out[..., 3], out[..., 4], cth, sth, cph, sph)

    @abstractmethod
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
        """Evaluate the network at one or many 5D phase-space points.

        Same shape-polymorphism contract as :meth:`EscapeNet2D.forward`:
        - 1-D batched tensors -> nine 1-D tensors of the same length
        - 0-D scalars -> nine 0-D scalars (vmap+jacrev path)

        ``cond`` (slowness conditioning, default ``None``) is concatenated to the
        trunk inputs when ``conditioning_dim > 0``.
        """

    def predict_theta_exit(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        phi: torch.Tensor,
        theta: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recover θ̂ ∈ [0, 2π) from the network's (cos θ̂, sin θ̂) output."""
        _, _, _, _, _, c, s, _, _ = self(x1, x2, x3, phi, theta, cond)
        angle = torch.atan2(s, c)
        return torch.where(angle < 0, angle + 2 * math.pi, angle)

    def predict_phi_exit(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        phi: torch.Tensor,
        theta: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recover φ̂ ∈ [0, π] from the network's (cos φ̂, sin φ̂) output.

        ``atan2`` returns the angle in [-π, π]. The φ image is [0, π] so
        we take the absolute value of the result.
        """
        _, _, _, _, _, _, _, c, s = self(x1, x2, x3, phi, theta, cond)
        return torch.atan2(s.abs(), c)


class EscapeNetCartesian3D(EscapeNet):
    """Escape equation network on a 3D domain with Cartesian momentum
    (6D phase space).

    The ray direction is a unit 3-vector ``d = (d₁, d₂, d₃)`` rather than polar
    angles, so there is no ``1/sin φ`` term and no pole band.

    ``forward(x1, x2, x3, d1, d2, d3)`` returns eight outputs in
    canonical order:

        û, σ̂, ŷ₁, ŷ₂, ŷ₃, d̂₁, d̂₂, d̂₃

    The exit direction ``d̂`` is a free 3-vector. The optional sphere
    regularizer ``mean((|d̂|²-1)²)`` pins its magnitude during training.
    """

    N_OUTPUTS: int = 8

    L_x1: float
    L_x2: float
    L_x3: float
    unit_norm_output: bool = False
    factored_eikonal: bool = False

    def factored_uv(self, raw_u, raw_sigma, x1, x2, x3, d1, d2, d3):
        """û, σ̂ for the factored-Eikonal ansatz (Cartesian direction)."""

        bg = background_escape_cartesian_3d(
            x1, x2, x3, d1, d2, d3, (self.L_x1, self.L_x2, self.L_x3)
        )
        return bg + raw_u, bg + raw_sigma

    def apply_unit_norm(self, u, sigma, out):
        """Final 8-tuple, applying ``unit_norm_output`` to d̂ (channels 5,6,7).
        no-op if ``unit_norm_output=False``."""
        e1, e2, e3 = out[..., 5], out[..., 6], out[..., 7]
        if self.unit_norm_output:
            r = torch.sqrt(e1 * e1 + e2 * e2 + e3 * e3).clamp_min(1e-12)
            e1, e2, e3 = e1 / r, e2 / r, e3 / r
        return (u, sigma, out[..., 2], out[..., 3], out[..., 4], e1, e2, e3)

    @abstractmethod
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
        """Evaluate the network at one or many 6-coordinate phase points.

        Same shape-polymorphism contract as the other variants:
        - 1-D batched tensors -> eight 1-D tensors of the same length
        - 0-D scalars -> eight 0-D scalars (the vmap+jacrev path).

        ``cond`` (slowness conditioning, default ``None``) is concatenated to the
        trunk inputs when ``conditioning_dim > 0``.
        """

    def spherical_view(self) -> "_SphericalView":
        """Wrap as a 9-output spherical-signature module for diagnostics.

        The entire 3D visualization + metrics surface is written against the
        spherical ``forward(x1,x2,x3,φ,θ) -> 9-tuple`` contract. This
        adapter converts ``(φ,θ)`` inputs to a unit vector ``d``, calls the
        Cartesian network, and maps the exit direction ``d̂`` back to the
        spherical ``(cos θ̂, sin θ̂, cos φ̂, sin φ̂)`` embedding. So, every
        existing 3D diagnostic works on a Cartesian model unchanged. The
        training path never uses this wrapper.
        """
        return _SphericalView(self)


class _SphericalView(nn.Module):
    """Spherical signature adapter around an :class:`EscapeNetCartesian3D`.

    Presents ``N_OUTPUTS = 9`` and a ``forward(x1,x2,x3,phi,theta)`` that
    returns ``(û, σ̂, ŷ₁, ŷ₂, ŷ₃, cos θ̂, sin θ̂, cos φ̂, sin φ̂)``, so the
    dim-dispatching diagnostic helpers route to their 3D-spherical branches.
    TODO: This is a hack, refactor.
    """

    N_OUTPUTS: int = 9

    def __init__(self, model: "EscapeNetCartesian3D"):
        super().__init__()
        self.model = model
        self.L_x1 = model.L_x1
        self.L_x2 = model.L_x2
        self.L_x3 = model.L_x3

    def forward(self, x1, x2, x3, phi, theta, cond=None):
        sin_ph = torch.sin(phi)
        d1 = sin_ph * torch.cos(theta)
        d2 = sin_ph * torch.sin(theta)
        d3 = torch.cos(phi)
        u, sigma, y1, y2, y3, e1, e2, e3 = (
            self.model(x1, x2, x3, d1, d2, d3)
            if cond is None
            else self.model(x1, x2, x3, d1, d2, d3, cond)
        )
        # Map exit direction d̂ -> spherical (cos θ̂, sin θ̂, cos φ̂, sin φ̂).
        rho = torch.sqrt(e1 * e1 + e2 * e2) + 1e-12
        norm = torch.sqrt(e1 * e1 + e2 * e2 + e3 * e3) + 1e-12
        cos_th = e1 / rho
        sin_th = e2 / rho
        cos_ph = e3 / norm
        sin_ph_hat = rho / norm
        return (u, sigma, y1, y2, y3, cos_th, sin_th, cos_ph, sin_ph_hat)

    def predict_theta_exit(self, x1, x2, x3, phi, theta, cond=None):
        _, _, _, _, _, c, s, _, _ = self(x1, x2, x3, phi, theta, cond)
        angle = torch.atan2(s, c)
        return torch.where(angle < 0, angle + 2 * math.pi, angle)

    def predict_phi_exit(self, x1, x2, x3, phi, theta, cond=None):
        _, _, _, _, _, _, _, c, s = self(x1, x2, x3, phi, theta, cond)
        return torch.atan2(s.abs(), c)
