"""Boundary condition sampling and loss for the 3D escape equations with
spherical coordinates (5D phase space).

On the spatial boundary ∂D of the cuboid [0, L_x1] x [0, L_x2] x [0, L_x3],
for every outward-pointing ray direction (φ, θ) all escape quantities are
known:

    û = 0,  σ̂ = 0,  ŷ = (x₁, x₂, x₃),  θ̂ = θ,  φ̂ = φ

Outward-pointing means the characteristic velocity
``(sin φ cos θ, sin φ sin θ, cos φ)`` points away from the interior at
the given face.

Sampling respects face area and counts the edges and corners only once.
φ stays strictly inside ``(δ, π - δ)`` with ``δ = π / (2 N_φ)`` (mirrors
the mesh solver's half-cell offset).

The loss is scale-normalized: ``yᵢ`` residuals are divided by ``L_xᵢ²`` so
all nine BC components live on comparable magnitudes regardless of cuboid
aspect ratio.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..networks import EscapeNet3D


@dataclass
class BoundaryLoss3D:
    """Boundary loss with per-component breakdown.

    Same conventions as :class:`BoundaryLoss2D`: ``total`` carries the
    autograd graph. The nine per-component fields are detached for
    logging and diagnostics.
    """

    total: torch.Tensor
    u: torch.Tensor
    sigma: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor
    y3: torch.Tensor
    cos_theta: torch.Tensor
    sin_theta: torch.Tensor
    cos_phi: torch.Tensor
    sin_phi: torch.Tensor


_FACES: tuple[tuple[str, int, int], ...] = (
    ("x1_low", 0, -1),
    ("x1_high", 0, +1),
    ("x2_low", 1, -1),
    ("x2_high", 1, +1),
    ("x3_low", 2, -1),
    ("x3_high", 2, +1),
)
"""The six faces of the cuboid in canonical order. Each entry is
(face_name, axis_index_normal, sign) where sign = -1 for the low face
(x_i = 0) and +1 for the high face (x_i = L_i)"""


def _direction_components(
    phi: torch.Tensor, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (d1, d2, d3) = (sin φ cos θ, sin φ sin θ, cos φ)"""
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return sin_phi * cos_theta, sin_phi * sin_theta, cos_phi


def _is_outward_3d(
    face: str,
    phi: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Boolean mask: ray on ``face`` points away from the cuboid interior.

    ``eps`` is larger than the 2D path's ``1e-6`` to absorb fp32 noise in
    ``sin φ`` near the poles.
    """
    d1, d2, d3 = _direction_components(phi, theta)
    if face == "x1_low":
        return d1 <= -eps
    if face == "x1_high":
        return d1 >= eps
    if face == "x2_low":
        return d2 <= -eps
    if face == "x2_high":
        return d2 >= eps
    if face == "x3_low":
        return d3 <= -eps
    if face == "x3_high":
        return d3 >= eps
    raise ValueError(f"Unknown face: {face!r}")


def _face_axis_counts(
    physical_size: tuple[float, float, float], n_per_long: int
) -> tuple[int, int, int]:
    """Per-axis spatial sample counts, scaled by axis length.

    Returns (n1, n2, n3) for the three axes. The longest axis
    gets ``n_per_long``, the others scale proportionally and
    are floored at 2.
    """
    L_max = max(physical_size)
    return tuple(max(2, round(n_per_long * L / L_max)) for L in physical_size)  # type: ignore[return-value]


def _phi_delta(n_phi: int) -> float:
    """Half-cell offset that keeps φ strictly away from the poles.

    Matches the mesh solver's ``φ_m = (m + ½)·π / N_φ`` convention.
    """
    return math.pi / (2 * n_phi)


def sample_boundary_points_3d(
    physical_size: tuple[float, float, float],
    n_per_long_edge: int = 16,
    n_theta: int = 24,
    n_phi: int = 12,
    device: torch.device | None = None,
    *,
    mode: Literal["grid", "random"],
    generator: torch.Generator | None = None,
    phi_delta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``(x1, x2, x3, φ, θ)`` boundary phase-space points.

    Samples are drawn from all faces and are then masked by :func:`_is_outward_3d`
    to only keep outward-pointing rays. The φ grid lives strictly inside
    ``(δ, π - δ)``.

    Parameters
    ----------
    physical_size
        ``(L_x1, L_x2, L_x3)`` cuboid extents.
    n_per_long_edge
        Spatial samples on each face's longest axis. Other axes scale
        proportionally and floor at 2.
    n_theta, n_phi
        Number of θ and φ samples per spatial face point. The candidate
        samples are then filtered to outward-pointing rays, so the number of
        returned points is smaller than the total number of (spatial x φ x θ)
        candidates.
    device
        Target device for the returned tensors.
    mode
        ``grid`` for deterministic linspace sampling. ``random`` for
        independent uniform draws. In grid mode, edges and corners are
        sampled only once.
    generator
        Source of randomness for ``mode="random"``.
    phi_delta
        φ safe-band half-width to keep φ strictly away from the poles.
        When ``None`` falls back to the half-cell offset ``π / (2 N_φ)``.

    Returns
    -------
    x1, x2, x3, phi, theta
        1-D tensors of boundary samples, all on ``device``.
    """
    if mode not in {"grid", "random"}:
        raise ValueError(f"mode must be 'grid' or 'random', got {mode!r}")

    L = physical_size
    n_axes = _face_axis_counts(physical_size, n_per_long_edge)  # (n1, n2, n3)
    delta = _phi_delta(n_phi) if phi_delta is None else phi_delta

    if mode == "grid":
        thetas = torch.linspace(0, 2 * math.pi, n_theta + 1, device=device)[:-1]
        # Half-cell offset keeps phi strictly inside (δ, π − δ).
        phi_grid = torch.linspace(delta, math.pi - delta, n_phi, device=device)
    else:
        thetas = torch.rand(n_theta, generator=generator, device=device) * (2 * math.pi)
        phi_grid = delta + torch.rand(n_phi, generator=generator, device=device) * (
            math.pi - 2 * delta
        )

    # Precompute (phi, theta) pairs as a 2D grid expanded to (n_phi * n_theta,).
    phi_pairs = phi_grid.repeat_interleave(n_theta)
    theta_pairs = thetas.repeat(n_phi)
    n_angles = phi_pairs.numel()

    all_x1: list[torch.Tensor] = []
    all_x2: list[torch.Tensor] = []
    all_x3: list[torch.Tensor] = []
    all_phi: list[torch.Tensor] = []
    all_theta: list[torch.Tensor] = []

    for face_name, axis_normal, sign in _FACES:
        # x1 faces own all shared edges, x2 faces use `strict-interior` on x1,
        # x3 faces use `strict-interior` on x1 and x2.
        free_axes = tuple(a for a in (0, 1, 2) if a != axis_normal)
        if axis_normal == 0:
            strict_interior = (False, False)
        elif axis_normal == 1:
            strict_interior = (True, False)  # x1 strict, x3 full
        else:  # axis_normal == 2
            strict_interior = (True, True)  # x1 strict, x2 strict

        coord_per_axis: dict[int, torch.Tensor] = {}
        for free_axis, strict in zip(free_axes, strict_interior):
            n_a = n_axes[free_axis]
            L_a = L[free_axis]
            if mode == "grid":
                if strict:
                    coords = torch.linspace(0, L_a, n_a + 2, device=device)[1:-1]
                else:
                    coords = torch.linspace(0, L_a, n_a, device=device)
            else:
                coords = torch.rand(n_a, generator=generator, device=device) * L_a
            coord_per_axis[free_axis] = coords

        a_axis, b_axis = free_axes
        a_coords = coord_per_axis[a_axis]
        b_coords = coord_per_axis[b_axis]
        n_a = a_coords.numel()
        n_b = b_coords.numel()
        # 2D Cartesian product of the two free axes.
        a_grid = a_coords.repeat_interleave(n_b)
        b_grid = b_coords.repeat(n_a)
        n_spatial = a_grid.numel()
        # Fix the normal axis: x_normal = 0 or L_normal depending on sign.
        fixed_value = 0.0 if sign < 0 else L[axis_normal]
        fixed_grid = torch.full_like(a_grid, fixed_value)

        # Stitch back into (x1, x2, x3) according to which axis is fixed.
        coords3: list[torch.Tensor] = [None, None, None]  # type: ignore[list-item]
        coords3[axis_normal] = fixed_grid
        coords3[a_axis] = a_grid
        coords3[b_axis] = b_grid
        x1_spatial, x2_spatial, x3_spatial = coords3  # type: ignore[misc]

        # Pair every spatial sample with every (φ, θ) angle pair.
        x1_full = x1_spatial.repeat_interleave(n_angles)
        x2_full = x2_spatial.repeat_interleave(n_angles)
        x3_full = x3_spatial.repeat_interleave(n_angles)
        phi_full = phi_pairs.repeat(n_spatial)
        theta_full = theta_pairs.repeat(n_spatial)

        mask = _is_outward_3d(face_name, phi_full, theta_full)
        all_x1.append(x1_full[mask])
        all_x2.append(x2_full[mask])
        all_x3.append(x3_full[mask])
        all_phi.append(phi_full[mask])
        all_theta.append(theta_full[mask])

    return (
        torch.cat(all_x1),
        torch.cat(all_x2),
        torch.cat(all_x3),
        torch.cat(all_phi),
        torch.cat(all_theta),
    )


def bc_sampling_audit_3d(n_phi: int, n_theta: int) -> dict:
    """Sanity audit of the 3D outward filter at the four equator cardinals.

    On the equator (φ = π/2) the four cardinal θ ∈ {0, π/2, π, 3π/2}
    represent rays with exactly one non-zero direction component, so each
    maps to a single lateral face:

        (π/2, 0)    → ['x1_high']     (+x1 ray)
        (π/2, π/2)  → ['x2_high']     (+x2 ray)
        (π/2, π)    → ['x1_low']      (-x1 ray)
        (π/2, 3π/2) → ['x2_low']      (-x2 ray)

    Returns a dict with the following structure:
    {
        "outward_faces_at_cardinals": {
            <theta_value>: {
                "phi_rad": 1.5707963267948966,
                "theta_rad": 0.0,
                "expected": ["x1_high"],
                "actual": ["x1_high"],
                "healthy": True
            },
            ...
        },
        "n_phi": 12,
        "n_theta": 24,
        "all_healthy": True
    }
    """
    grid = torch.linspace(0, 2 * math.pi, n_theta + 1)[:-1]
    phi_eq = torch.tensor([math.pi / 2])
    cardinals: list[tuple[str, torch.Tensor, list[str]]] = [
        ("phi=pi/2, theta=0", torch.tensor([0.0]), ["x1_high"]),
        ("phi=pi/2, theta=pi/2", torch.tensor([math.pi / 2]), ["x2_high"]),
        ("phi=pi/2, theta=pi", torch.tensor([math.pi]), ["x1_low"]),
        ("phi=pi/2, theta=3pi/2", torch.tensor([3 * math.pi / 2]), ["x2_low"]),
    ]

    audit: dict = {
        "outward_faces_at_cardinals": {},
        "n_phi": int(n_phi),
        "n_theta": int(n_theta),
        "all_healthy": True,
    }
    # Verify each cardinal θ is exactly representable on the configured
    # theta grid, skip the audit entry if it is not.
    for label, th, expected in cardinals:
        target = float(th.item())
        idx = int(torch.argmin(torch.abs(grid - target)).item())
        if abs(grid[idx].item() - target) > 1e-3:
            continue
        th_on_grid = grid[idx : idx + 1]
        actual = [
            name
            for name, _, _ in _FACES
            if bool(_is_outward_3d(name, phi_eq, th_on_grid).item())
        ]
        healthy = actual == expected
        audit["outward_faces_at_cardinals"][label] = {
            "phi_rad": float(phi_eq.item()),
            "theta_rad": target,
            "expected": expected,
            "actual": actual,
            "healthy": healthy,
        }
        if not healthy:
            audit["all_healthy"] = False
    return audit


def compute_boundary_loss_3d(
    model: EscapeNet3D,
    physical_size: tuple[float, float, float],
    x1_bc: torch.Tensor,
    x2_bc: torch.Tensor,
    x3_bc: torch.Tensor,
    phi_bc: torch.Tensor,
    theta_bc: torch.Tensor,
    cond: torch.Tensor | None = None,
) -> BoundaryLoss3D:
    """Scale-aware MSE loss over all nine network outputs.

    Targets: û=0, σ̂=0, ŷᵢ=xᵢ for i ∈ {1,2,3},
    cos θ̂ = cos θ, sin θ̂ = sin θ, cos φ̂ = cos φ, sin φ̂ = sin φ.

    Position residuals are divided by ``L_xᵢ²`` so the y-components live
    on the same scale as the angle channels regardless of cuboid aspect
    ratio.
    """
    L_x1, L_x2, L_x3 = physical_size

    (
        u_pred,
        sigma_pred,
        y1_pred,
        y2_pred,
        y3_pred,
        c_th_pred,
        s_th_pred,
        c_ph_pred,
        s_ph_pred,
    ) = (
        model(x1_bc, x2_bc, x3_bc, phi_bc, theta_bc)
        if cond is None
        else model(x1_bc, x2_bc, x3_bc, phi_bc, theta_bc, cond)
    )

    loss_u = torch.mean(u_pred**2)
    loss_sigma = torch.mean(sigma_pred**2)
    loss_y1 = torch.mean((y1_pred - x1_bc) ** 2) / (L_x1 * L_x1)
    loss_y2 = torch.mean((y2_pred - x2_bc) ** 2) / (L_x2 * L_x2)
    loss_y3 = torch.mean((y3_pred - x3_bc) ** 2) / (L_x3 * L_x3)
    loss_cos_theta = torch.mean((c_th_pred - torch.cos(theta_bc)) ** 2)
    loss_sin_theta = torch.mean((s_th_pred - torch.sin(theta_bc)) ** 2)
    loss_cos_phi = torch.mean((c_ph_pred - torch.cos(phi_bc)) ** 2)
    loss_sin_phi = torch.mean((s_ph_pred - torch.sin(phi_bc)) ** 2)

    total = (
        loss_u
        + loss_sigma
        + loss_y1
        + loss_y2
        + loss_y3
        + loss_cos_theta
        + loss_sin_theta
        + loss_cos_phi
        + loss_sin_phi
    )

    return BoundaryLoss3D(
        total=total,
        u=loss_u.detach(),
        sigma=loss_sigma.detach(),
        y1=loss_y1.detach(),
        y2=loss_y2.detach(),
        y3=loss_y3.detach(),
        cos_theta=loss_cos_theta.detach(),
        sin_theta=loss_sin_theta.detach(),
        cos_phi=loss_cos_phi.detach(),
        sin_phi=loss_sin_phi.detach(),
    )
