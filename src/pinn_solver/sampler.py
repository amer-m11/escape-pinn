"""Collocation point sampling for the PINN solver.

Dispatches on :attr:`PhysicalSize.dim` (and, in 3D, the momentum
parametrization) to produce a 2D 3-coordinate set, a 3D-spherical
5-coordinate set, or a 3D-Cartesian 6-coordinate set:
    - The 3D-spherical sampler keeps φ strictly inside the safe band
    ``(δ, π - δ)`` to avoid the ``1/sin φ`` singularity in the
    angular-drift term of the PDE residual.
    - The Cartesian sampler draws unit direction vectors uniformly
    on the whole sphere S².

In the generalized PINN, each collocation point also draws a per-point
medium descriptor and a per-point conditioning vector for the network.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .configuration import PhysicalSize, SampleMode


@dataclass
class Samples2D:
    """Sampled collocation points in the 2D reduced phase space (x1, x2, θ)."""

    x1: torch.Tensor
    x2: torch.Tensor
    theta: torch.Tensor


@dataclass
class Samples3D:
    """Sampled collocation points in the 3D reduced phase space (x1, x2, x3, φ, θ).

    ``phi`` lies strictly inside ``(δ, π - δ)`` (guaranteed by the sampler)
    so the PDE residual's ``1/sin φ`` factor stays bounded.
    """

    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    phi: torch.Tensor
    theta: torch.Tensor
    cond: torch.Tensor | None = None
    """Per point slowness conditioning vector ``(N, cond_dim)`` fed to the network
    (normalized descriptor or probe grid). ``None`` for a single fixed medium."""
    params: torch.Tensor | None = None
    """Per point physical lens descriptor ``(N, param_dim)`` used to compute the 
    slowness for computing the PDE residuals ``n(x; params)``. ``None`` for a 
    single fixed medium."""


@dataclass
class Samples3DCartesian:
    """Collocation points in the 3D Cartesian-momentum phase space.

    ``(x1, x2, x3, d1, d2, d3)`` with ``(d1, d2, d3)`` a unit direction
    vector drawn uniformly on the sphere S².
    """

    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    d1: torch.Tensor
    d2: torch.Tensor
    d3: torch.Tensor
    cond: torch.Tensor | None = None
    """Per-point slowness-conditioning vector ``(N, cond_dim)`` fed to the network
    (normalized descriptor or probe grid). ``None`` for a single fixed medium."""
    params: torch.Tensor | None = None
    """Per-point physical lens descriptor ``(N, param_dim)`` used to compute the 
    slowness for computing the PDE residuals ``n(x; params)``. ``None`` for a 
    single fixed medium."""


Samples = Samples2D
"""Back-compat alias for existing 2D call sites."""


# Default half-cell offset matching the mesh solver's ``φ_m = (m + ½)π/N_φ``
# convention for N_φ = 4. Configurable per call.
_DEFAULT_PHI_DELTA = math.pi / 8


def _draw_conditioning(
    family, n: int, generator, device, size, medium_scale: float = 1.0
):
    """``(params, cond)`` for a conditioned family, else ``(None, None)``.

    Each collocation point draws an independent medium (per-point conditioning).
    - ``params`` is the physical descriptor (drives the PDE slowness n).
    - ``cond`` is the network input. Can be either
        1. the normalized descriptor (i.e. ``params``)
        2. the probe-grid slowness (encoder mode).
    Neither carries autograd grad as they are inputs, never differentiated by
    the PDE residual's Jacobian since they are not needed for the residuals.

    ``medium_scale`` (<1) scales down the drawn amplitudes of the lens before
    deriving ``cond``.
    """
    if family is None:
        return None, None
    params = family.sample_params(n, generator=generator, device=device)
    if medium_scale != 1.0:
        params = family.scale_contrast(params, medium_scale)
    cond = family.cond_for(size.as_tuple(), params)
    return params, cond


class Sampler:
    """Container of collocation point sampling strategies.

    The single entry point :meth:`sample_interior` dispatches on
    ``size.dim`` and returns either :class:`Samples2D` or :class:`Samples3D`.
    """

    @staticmethod
    def sample_interior(
        n: int,
        size: PhysicalSize,
        device: torch.device,
        *,
        mode: SampleMode = "random",
        generator: torch.Generator | None = None,
        phi_delta: float = _DEFAULT_PHI_DELTA,
        parametrization: Literal["spherical", "cartesian"] = "spherical",
        family=None,
        medium_scale: float = 1.0,
    ) -> Samples2D | Samples3D | Samples3DCartesian:
        """Sample ``n`` interior collocation points.

        Output tensors carry ``requires_grad=True``.

        Parameters
        ----------
        n : int
            Number of collocation points.
        size : PhysicalSize
            Domain extents. ``size.dim`` decides whether a 2D or 3D sample
            set is returned.
        device : torch.device
            Device for the returned tensors.
        mode : SampleMode
            ``"random"`` (default): uniform-random i.i.d. draws.
            ``"grid"`` deterministic lattice (not implemented yet).
        generator : torch.Generator | None
            Optional source of randomness for ``mode="random"``. When
            ``None`` the default torch RNG is used.
        phi_delta : float
            Safe width for φ in 3D-spherical representation. Samples lie strictly
            inside ``(phi_delta, π - phi_delta)``. Ignored in 2D and in the
            Cartesian path.
        parametrization : str
            3D momentum parametrization.
            - ``"spherical"`` (default) => :class:`Samples3D`
            - ``"cartesian"`` => :class:`Samples3DCartesian`
        medium_scale : float
            Scale factor for the drawn lens amplitudes. 1.0 (default) means no
            scaling. <1.0 reduces the contrast of the drawn lenses. Ignored for
            a single fixed medium. This can be used to successively increase the
            contrast of the training lenses over multiple phases of training.
        """
        if mode == "grid":
            raise NotImplementedError(
                "Sampler.sample_interior(mode='grid') is reserved for a fixed "
                "deterministic lattice; not implemented yet."
            )
        if mode != "random":
            raise ValueError(f"Unsupported interior sample mode: {mode!r}")

        if size.dim == 3:
            if parametrization == "cartesian":
                return Sampler._sample_random_cartesian_3d(
                    n,
                    size,
                    device,
                    generator=generator,
                    family=family,
                    medium_scale=medium_scale,
                )
            return Sampler._sample_random_3d(
                n,
                size,
                device,
                generator=generator,
                phi_delta=phi_delta,
                family=family,
                medium_scale=medium_scale,
            )
        return Sampler._sample_random_2d(n, size, device, generator=generator)

    # ------------------------------------------------------------------
    # Internal: dim-specific samplers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_random_2d(
        n: int,
        size: PhysicalSize,
        device: torch.device,
        *,
        generator: torch.Generator | None,
    ) -> Samples2D:
        x1 = torch.rand(n, generator=generator, device=device) * size.L_x1
        x2 = torch.rand(n, generator=generator, device=device) * size.L_x2
        theta = torch.rand(n, generator=generator, device=device) * 2 * torch.pi
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        theta.requires_grad_(True)
        return Samples2D(x1=x1, x2=x2, theta=theta)

    @staticmethod
    def _sample_random_3d(
        n: int,
        size: PhysicalSize,
        device: torch.device,
        *,
        generator: torch.Generator | None,
        phi_delta: float,
        family=None,
        medium_scale: float = 1.0,
    ) -> Samples3D:
        assert size.L_x3 is not None, "size.dim==3 implies L_x3 is set"
        if not 0.0 < phi_delta < math.pi / 2:
            raise ValueError(f"phi_delta must lie in (0, π/2); got {phi_delta!r}")
        x1 = torch.rand(n, generator=generator, device=device) * size.L_x1
        x2 = torch.rand(n, generator=generator, device=device) * size.L_x2
        x3 = torch.rand(n, generator=generator, device=device) * size.L_x3
        # φ ∈ Uniform(δ, π − δ).
        phi = phi_delta + torch.rand(n, generator=generator, device=device) * (
            math.pi - 2 * phi_delta
        )
        theta = torch.rand(n, generator=generator, device=device) * 2 * torch.pi
        for t in (x1, x2, x3, phi, theta):
            t.requires_grad_(True)
        params, cond = _draw_conditioning(
            family, n, generator, device, size, medium_scale
        )
        return Samples3D(
            x1=x1, x2=x2, x3=x3, phi=phi, theta=theta, cond=cond, params=params
        )

    @staticmethod
    def _sample_random_cartesian_3d(
        n: int,
        size: PhysicalSize,
        device: torch.device,
        *,
        generator: torch.Generator | None,
        family=None,
        medium_scale: float = 1.0,
    ) -> Samples3DCartesian:
        """Spatial uniform in the cuboid, direction uniform on the sphere S².
        The direction vector is normalized to unit length.

        Uniformity on S² comes from normalizing an i.i.d. standard-Gaussian
        3-vector (Muller's method): the Gaussian is rotationally symmetric,
        so the normalized vector is uniform over directions.
        """
        assert size.L_x3 is not None, "size.dim==3 implies L_x3 is set"
        L = (size.L_x1, size.L_x2, size.L_x3)

        def _unit(m):
            g = torch.randn(m, 3, generator=generator, device=device)
            return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)

        x = torch.rand(n, 3, generator=generator, device=device) * torch.tensor(
            L, device=device
        )
        d = _unit(n)

        x1, x2, x3 = x[:, 0].clone(), x[:, 1].clone(), x[:, 2].clone()
        d1, d2, d3 = d[:, 0].clone(), d[:, 1].clone(), d[:, 2].clone()
        for t in (x1, x2, x3, d1, d2, d3):
            t.requires_grad_(True)
        params, cond = _draw_conditioning(
            family, n, generator, device, size, medium_scale
        )
        return Samples3DCartesian(
            x1=x1, x2=x2, x3=x3, d1=d1, d2=d2, d3=d3, cond=cond, params=params
        )
