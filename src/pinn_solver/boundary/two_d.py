"""Boundary condition sampling and loss for the 2D escape equations with
spherical coordinates (3D phase space).

On the spatial boundary ∂D of the domain [0, L_x1] x [0, L_x2], for every
outward-pointing ray direction θ all escape quantities are known:

    û = 0,  σ̂ = 0,  ŷ = (x₁, x₂),  θ̂ = θ

Outward-pointing means the characteristic direction (cos θ, sin θ)
points away from the interior at the given boundary point.

Sampling respects the length of the edges and counts the corners only once.

The loss is scale-normalized: position residuals are divided by ``L_x{1,2}²``
so all six BC components live on comparable magnitudes regardless of
domain size.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..networks import EscapeNet2D


@dataclass
class BoundaryLoss2D:
    """Boundary loss with per-component breakdown.

    ``total`` carries the autograd graph. ``backward()`` is called on it.
    The six per-component fields are detached scalars for logging and
    diagnostics.
    """

    total: torch.Tensor
    u: torch.Tensor
    sigma: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def _is_outward_2d(edge: str, theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Boolean mask: True where the ray direction is outward on the given edge.

    A ray is outward when its velocity component normal to the edge has the
    correct sign to exit through that wall:
      left   (x1=0):     outward iff cos θ ≤ -eps  (moving in -x1 direction)
      right  (x1=L_x1):  outward iff cos θ ≥  eps  (moving in +x1 direction)
      bottom (x2=0):     outward iff sin θ ≤ -eps  (moving in -x2 direction)
      top    (x2=L_x2):  outward iff sin θ ≥  eps  (moving in +x2 direction)

    The eps guard excludes tangential rays.
    """
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)

    if edge == "left":
        return cos_th <= -eps
    if edge == "right":
        return cos_th >= eps
    if edge == "bottom":
        return sin_th <= -eps
    if edge == "top":
        return sin_th >= eps
    raise ValueError(f"Unknown edge: {edge}")


def _edge_spatial_counts(
    physical_size: tuple[float, float], n_per_long_edge: int
) -> tuple[int, int]:
    """The number of samples per edge proportional to edge length
    (n_horizontal, n_vertical).

    Long edges receive ``n_per_long_edge`` samples, short edges receive
    ``round(n_per_long_edge · L_short / L_long)``, both floored at 2.
    """
    L_x1, L_x2 = physical_size
    L_max = max(L_x1, L_x2)
    n_horizontal = max(2, round(n_per_long_edge * L_x1 / L_max))
    n_vertical = max(2, round(n_per_long_edge * L_x2 / L_max))
    return n_horizontal, n_vertical


def sample_boundary_points_2d(
    physical_size: tuple[float, float],
    n_per_long_edge: int = 64,
    n_theta: int = 32,
    device: torch.device | None = None,
    *,
    mode: Literal["grid", "random"],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (x1, x2, θ) boundary phase-space points with outward-pointing
    rays.

    θ is sampled for each spatial point and then filtered to only keep
    outward-pointing rays. So, the number of returned points is not exactly
    ``4 * n_per_long_edge * n_theta``"""

    if mode not in {"grid", "random"}:
        raise ValueError(f"mode must be 'grid' or 'random', got {mode!r}")

    L_x1, L_x2 = physical_size
    n_horiz, n_vert = _edge_spatial_counts(physical_size, n_per_long_edge)

    if mode == "grid":
        thetas = torch.linspace(0, 2 * math.pi, n_theta + 1, device=device)[:-1]
    else:
        thetas = torch.rand(n_theta, generator=generator, device=device) * (2 * math.pi)

    edges: list[tuple[str, float | None, float | None, int]] = [
        ("left", 0.0, None, n_vert),
        ("right", L_x1, None, n_vert),
        ("bottom", None, 0.0, n_horiz),
        ("top", None, L_x2, n_horiz),
    ]

    all_x1: list[torch.Tensor] = []
    all_x2: list[torch.Tensor] = []
    all_theta: list[torch.Tensor] = []

    for edge_name, fixed_x1, fixed_x2, n_spatial in edges:
        # Vertical edges contain the corners
        # horizontal edges are strictly inside (0, L_x1)
        if fixed_x1 is not None:
            if mode == "grid":
                x2_vals = torch.linspace(0, L_x2, n_spatial, device=device)
            else:
                x2_vals = (
                    torch.rand(n_spatial, generator=generator, device=device) * L_x2
                )
            x1_vals = torch.full_like(x2_vals, fixed_x1)
        elif fixed_x2 is not None:
            if mode == "grid":
                x1_vals = torch.linspace(0, L_x1, n_spatial + 2, device=device)[1:-1]
            else:
                x1_vals = (
                    torch.rand(n_spatial, generator=generator, device=device) * L_x1
                )
            x2_vals = torch.full_like(x1_vals, fixed_x2)
        else:
            raise ValueError(f"Edge {edge_name} must have either x1 or x2 fixed")

        x1_grid = x1_vals.repeat_interleave(n_theta)
        x2_grid = x2_vals.repeat_interleave(n_theta)
        th_grid = thetas.repeat(x1_vals.numel())

        mask = _is_outward_2d(edge_name, th_grid)
        all_x1.append(x1_grid[mask])
        all_x2.append(x2_grid[mask])
        all_theta.append(th_grid[mask])

    return torch.cat(all_x1), torch.cat(all_x2), torch.cat(all_theta)


def bc_sampling_audit_2d(n_theta: int) -> dict:
    """Sanity audit of the 2D BC outward filter at cardinal θ.

    For each cardinal θ ∈ {0, π/2, π, 3π/2} that lies on the uniform BC grid
    (i.e. ``n_theta`` divisible by 4), report which walls ``_is_outward_2d``
    flagged as outward at that θ. A healthy sampler returns exactly one wall
    per cardinal θ.

    Returns a dict with the following structure:
    {
        "outward_walls_at_cardinals": {
            <theta_value>: {
                "theta_rad": 0.0,
                "expected": ["right"],
                "actual": ["right"],
                "healthy": True
            },
            ...
        },
        "all_healthy": True
    }
    """
    grid = torch.linspace(0, 2 * math.pi, n_theta + 1)[:-1]
    cardinals: dict[float, str] = {
        0.0: "right",
        math.pi / 2: "top",
        math.pi: "left",
        3 * math.pi / 2: "bottom",
    }
    audit: dict = {"outward_walls_at_cardinals": {}, "all_healthy": True}
    for theta_value, expected_wall in cardinals.items():
        idx = int(torch.argmin(torch.abs(grid - theta_value)).item())
        if abs(grid[idx].item() - theta_value) > 1e-3:
            continue
        actual = [
            edge
            for edge in ("left", "right", "bottom", "top")
            if _is_outward_2d(edge, grid[idx : idx + 1]).item()
        ]
        healthy = actual == [expected_wall]
        audit["outward_walls_at_cardinals"][f"{theta_value:.6f}"] = {
            "theta_rad": float(theta_value),
            "expected": [expected_wall],
            "actual": actual,
            "healthy": healthy,
        }
        if not healthy:
            audit["all_healthy"] = False
    return audit


def compute_boundary_loss_2d(
    model: EscapeNet2D,
    physical_size: tuple[float, float],
    x1_bc: torch.Tensor,
    x2_bc: torch.Tensor,
    theta_bc: torch.Tensor,
) -> BoundaryLoss2D:
    """Scale-aware MSE loss over all escape network outputs.

    Targets: û=0, σ̂=0, ŷ1=x1, ŷ2=x2, cos θ̂=cos θ, sin θ̂=sin θ.
    Position residuals are divided by L_x{1,2}² so the y-components live
    on the same scale as the others.
    """
    L_x1, L_x2 = physical_size

    u_pred, sigma_pred, y1_pred, y2_pred, c_pred, s_pred = model(x1_bc, x2_bc, theta_bc)

    loss_u = torch.mean(u_pred**2)
    loss_sigma = torch.mean(sigma_pred**2)
    loss_y1 = torch.mean((y1_pred - x1_bc) ** 2) / (L_x1 * L_x1)
    loss_y2 = torch.mean((y2_pred - x2_bc) ** 2) / (L_x2 * L_x2)
    loss_cos = torch.mean((c_pred - torch.cos(theta_bc)) ** 2)
    loss_sin = torch.mean((s_pred - torch.sin(theta_bc)) ** 2)

    total = loss_u + loss_sigma + loss_y1 + loss_y2 + loss_cos + loss_sin

    return BoundaryLoss2D(
        total=total,
        u=loss_u.detach(),
        sigma=loss_sigma.detach(),
        y1=loss_y1.detach(),
        y2=loss_y2.detach(),
        cos=loss_cos.detach(),
        sin=loss_sin.detach(),
    )
