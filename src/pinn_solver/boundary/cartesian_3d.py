"""Boundary sampling and loss for the 3D escape equations with Cartesian
coordinates (6D phase space). This is a parallel implementation of the
spherical version in ``three_d.py``.

On the spatial boundary ∂D of the cuboid, for every outward-pointing ray
direction ``d = (d₁, d₂, d₃)`` all escape quantities are known:

    û = 0,  σ̂ = 0,  ŷ = (x₁, x₂, x₃),  d̂ = d

Outward-pointing means that the characteristic direction ``d`` points away
from the interior at the given boundary point.

The loss is scale-normalized: ``yᵢ`` residuals are divided by ``L_xᵢ²`` so
all eight BC components live on comparable magnitudes regardless of cuboid
aspect ratio.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..networks import EscapeNetCartesian3D


@dataclass
class BoundaryLossCartesian3D:
    """Boundary loss with per-component breakdown.

    ``total`` carries the autograd graph. The eight per-component fields are
    detached for diagnostics.
    """

    total: torch.Tensor
    u: torch.Tensor
    sigma: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor
    y3: torch.Tensor
    d1: torch.Tensor
    d2: torch.Tensor
    d3: torch.Tensor


_FACES: tuple[tuple[str, int, int], ...] = (
    ("x1_low", 0, -1),
    ("x1_high", 0, +1),
    ("x2_low", 1, -1),
    ("x2_high", 1, +1),
    ("x3_low", 2, -1),
    ("x3_high", 2, +1),
)
"""The six faces of the cuboid in canonical order: (name, normal axis, sign).
sign = -1 for the low face (x_i = 0), +1 for the high face (x_i = L_i).
The outward unit normal is sign·e_axis.
"""


def _is_outward_cartesian(
    face: str,
    d1: torch.Tensor,
    d2: torch.Tensor,
    d3: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Boolean mask: ray direction ``d`` on ``face`` points outward.

    The outward normal of each face is ±e_axis. The ray exits iff the
    component of ``d`` along that normal is positive.

    ``eps`` only excludes rays exactly tangent to the face.
    """
    comp = {0: d1, 1: d2, 2: d3}
    for name, axis, sign in _FACES:
        if name == face:
            return (sign * comp[axis]) >= eps
    raise ValueError(f"Unknown face: {face!r}")


def _face_axis_counts(
    physical_size: tuple[float, float, float], n_per_long: int
) -> tuple[int, int, int]:
    """Per-axis spatial sample counts, scaled by axis length (floored at 2)."""
    L_max = max(physical_size)
    return tuple(  # type: ignore[return-value]
        max(2, round(n_per_long * L / L_max)) for L in physical_size
    )


def sample_boundary_points_cartesian_3d(
    physical_size: tuple[float, float, float],
    n_per_long_edge: int = 16,
    n_phi_grid: int = 9,
    n_theta_grid: int = 24,
    device: torch.device | None = None,
    *,
    mode: Literal["grid", "random"],
    generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Sample ``(x1, x2, x3, d1, d2, d3)`` boundary phase-space points.

    Samples are drawn on all faces of the cuboid and then filtered to
    keep only outward-pointing rays.

    Parameters
    ----------
    physical_size
        ``(L_x1, L_x2, L_x3)`` cuboid extents.
    n_per_long_edge
        Spatial samples on each face's longest axis, other axes scale
        proportionally and floor at 2.
    n_phi_grid, n_theta_grid
        Number of polar (φ) and azimuthal (θ) levels in the direction grid.
        In grid mode, φ uses a half-cell offset around poles (TODO: this
        should be no longer needed? Use Fibonacci sphere instead? Might need
        to solve the issue that the Fibonacci approach might not sample
        directions close to the cardinal directions causing some issues).
    mode
        ``"grid"`` uses a deterministic spatial linspace + explicit
        (θ, φ) direction grid. ``"random"`` uses uniform spatial draws
        + Gaussian-normalized directions. In grid mode, edges and corners
        are sampled only once.
    generator
        Source of randomness for ``mode="random"``.

    Returns
    -------
    x1, x2, x3, d1, d2, d3 : 1-D tensors of boundary samples on ``device``.
    """
    if mode not in {"grid", "random"}:
        raise ValueError(f"mode must be 'grid' or 'random', got {mode!r}")

    L = physical_size
    n_direction = n_phi_grid * n_theta_grid
    n_axes = _face_axis_counts(physical_size, n_per_long_edge)

    if mode == "random":
        g = torch.randn(n_direction, 3, generator=generator, device=device)
        dirs = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        phi_delta = math.pi / (2 * n_phi_grid)
        thetas_g = torch.linspace(0, 2 * math.pi, n_theta_grid + 1, device=device)[:-1]
        phis_g = torch.linspace(
            phi_delta, math.pi - phi_delta, n_phi_grid, device=device
        )
        ph_rep = phis_g.repeat_interleave(n_theta_grid)
        th_rep = thetas_g.repeat(n_phi_grid)
        sin_ph = torch.sin(ph_rep)
        dirs = torch.stack(
            [sin_ph * torch.cos(th_rep), sin_ph * torch.sin(th_rep), torch.cos(ph_rep)],
            dim=1,
        ).float()

    all_x1, all_x2, all_x3 = [], [], []
    all_d1, all_d2, all_d3 = [], [], []

    for face_name, axis_normal, sign in _FACES:
        free_axes = tuple(a for a in (0, 1, 2) if a != axis_normal)
        # Strict-interior on the lower-index to avoid duplicate
        # cuboid-edge spatial points in grid mode.
        if axis_normal == 0:
            strict = (False, False)
        elif axis_normal == 1:
            strict = (True, False)
        else:
            strict = (True, True)

        coord_per_axis: dict[int, torch.Tensor] = {}
        for free_axis, st in zip(free_axes, strict):
            n_a = n_axes[free_axis]
            L_a = L[free_axis]
            if mode == "grid":
                if st:
                    coords = torch.linspace(0, L_a, n_a + 2, device=device)[1:-1]
                else:
                    coords = torch.linspace(0, L_a, n_a, device=device)
            else:
                coords = torch.rand(n_a, generator=generator, device=device) * L_a
            coord_per_axis[free_axis] = coords

        a_axis, b_axis = free_axes
        a_coords = coord_per_axis[a_axis]
        b_coords = coord_per_axis[b_axis]
        n_a, n_b = a_coords.numel(), b_coords.numel()
        a_grid = a_coords.repeat_interleave(n_b)
        b_grid = b_coords.repeat(n_a)
        n_spatial = a_grid.numel()
        fixed_value = 0.0 if sign < 0 else L[axis_normal]
        fixed_grid = torch.full_like(a_grid, fixed_value)

        coords3: list[torch.Tensor] = [None, None, None]  # type: ignore[list-item]
        coords3[axis_normal] = fixed_grid
        coords3[a_axis] = a_grid
        coords3[b_axis] = b_grid
        x1_sp, x2_sp, x3_sp = coords3  # type: ignore[misc]

        # Keep only outward directions for this face, then pair with spatial.
        mask = _is_outward_cartesian(face_name, dirs[:, 0], dirs[:, 1], dirs[:, 2])
        face_dirs = dirs[mask]
        n_dir = face_dirs.shape[0]
        if n_dir == 0 or n_spatial == 0:
            continue

        x1_full = x1_sp.repeat_interleave(n_dir)
        x2_full = x2_sp.repeat_interleave(n_dir)
        x3_full = x3_sp.repeat_interleave(n_dir)
        d_full = face_dirs.repeat(n_spatial, 1)

        all_x1.append(x1_full)
        all_x2.append(x2_full)
        all_x3.append(x3_full)
        all_d1.append(d_full[:, 0])
        all_d2.append(d_full[:, 1])
        all_d3.append(d_full[:, 2])

    return (
        torch.cat(all_x1),
        torch.cat(all_x2),
        torch.cat(all_x3),
        torch.cat(all_d1),
        torch.cat(all_d2),
        torch.cat(all_d3),
    )


def _fibonacci_sphere(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Deterministic near-uniform set of ``n`` unit vectors on S²."""
    import math

    i = torch.arange(n, dtype=torch.float64, device=device)
    phi_golden = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    theta = phi_golden * i
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1).to(torch.float32)


def bc_sampling_audit_cartesian_3d(n_phi_grid: int = 9, n_theta_grid: int = 24) -> dict:
    """Sanity audit of the Cartesian BC sampler.

    Two checks:
    1. Each axis-aligned unit direction is outward at exactly one face.
    2. Grid mode places exact cardinal θ = 0°, 90°, 180°, 270° directions
       in the equatorial band (for n_theta_grid divisible by 4).

    Returns a dict with the following structure:
    {
        "outward_faces_at_cardinals": {
            <face_name>: {
                "direction": [d1, d2, d3],
                "expected": [face_name],
                "actual": [face_name],
                "healthy": True
            },
            ...
        },
        "grid_cardinal_coverage": {
            <face_name>: {
                "min_angular_gap_deg": 0.0,
                "covered": True
            },
            ...
        },
        "all_healthy": True
    }
    """
    cardinals = [
        ("+x1", (1.0, 0.0, 0.0), "x1_high"),
        ("-x1", (-1.0, 0.0, 0.0), "x1_low"),
        ("+x2", (0.0, 1.0, 0.0), "x2_high"),
        ("-x2", (0.0, -1.0, 0.0), "x2_low"),
        ("+x3", (0.0, 0.0, 1.0), "x3_high"),
        ("-x3", (0.0, 0.0, -1.0), "x3_low"),
    ]
    audit: dict = {
        "outward_faces_at_cardinals": {},
        "grid_cardinal_coverage": {},
        "all_healthy": True,
    }
    for label, (a, b, c), expected in cardinals:
        d1 = torch.tensor([a])
        d2 = torch.tensor([b])
        d3 = torch.tensor([c])
        actual = [
            name
            for name, _, _ in _FACES
            if bool(_is_outward_cartesian(name, d1, d2, d3).item())
        ]
        healthy = actual == [expected]
        audit["outward_faces_at_cardinals"][label] = {
            "direction": [a, b, c],
            "expected": [expected],
            "actual": actual,
            "healthy": healthy,
        }
        if not healthy:
            audit["all_healthy"] = False

    # Verify grid mode includes exact θ = 0°, 90°, 180°, 270° in equator.
    _, _, _, d1g, d2g, d3g = sample_boundary_points_cartesian_3d(
        (1.0, 1.0, 1.0),
        n_phi_grid=n_phi_grid,
        n_theta_grid=n_theta_grid,
        mode="grid",
    )
    dirs = torch.stack([d1g, d2g, d3g], dim=1)
    for label, target in [
        ("+x1", [1, 0, 0]),
        ("-x1", [-1, 0, 0]),
        ("+x2", [0, 1, 0]),
        ("-x2", [0, -1, 0]),
    ]:
        t = torch.tensor(target, dtype=torch.float32)
        min_dist = float((dirs - t).norm(dim=1).min())
        covered = min_dist < 1e-4
        audit["grid_cardinal_coverage"][label] = {
            "min_angular_gap_deg": math.degrees(min_dist),
            "covered": covered,
        }
        if not covered:
            audit["all_healthy"] = False

    return audit


def compute_boundary_loss_cartesian_3d(
    model: EscapeNetCartesian3D,
    physical_size: tuple[float, float, float],
    x1_bc: torch.Tensor,
    x2_bc: torch.Tensor,
    x3_bc: torch.Tensor,
    d1_bc: torch.Tensor,
    d2_bc: torch.Tensor,
    d3_bc: torch.Tensor,
    cond: torch.Tensor | None = None,
) -> BoundaryLossCartesian3D:
    """Scale-aware MSE over all eight network outputs.

    Targets: û=0, σ̂=0, ŷᵢ=xᵢ for i ∈ {1,2,3}, d̂=d. Position residuals are
    divided by ``L_xᵢ²`` so the y-components have a comparable scale regardless
    of cuboid aspect ratio.
    """
    L_x1, L_x2, L_x3 = physical_size

    # Pass ``cond`` only when conditioning is active
    args = (x1_bc, x2_bc, x3_bc, d1_bc, d2_bc, d3_bc)
    u_p, s_p, y1_p, y2_p, y3_p, e1_p, e2_p, e3_p = (
        model(*args) if cond is None else model(*args, cond)
    )

    loss_u = torch.mean(u_p**2)
    loss_sigma = torch.mean(s_p**2)
    loss_y1 = torch.mean((y1_p - x1_bc) ** 2) / (L_x1 * L_x1)
    loss_y2 = torch.mean((y2_p - x2_bc) ** 2) / (L_x2 * L_x2)
    loss_y3 = torch.mean((y3_p - x3_bc) ** 2) / (L_x3 * L_x3)
    loss_d1 = torch.mean((e1_p - d1_bc) ** 2)
    loss_d2 = torch.mean((e2_p - d2_bc) ** 2)
    loss_d3 = torch.mean((e3_p - d3_bc) ** 2)

    total = (
        loss_u + loss_sigma + loss_y1 + loss_y2 + loss_y3 + loss_d1 + loss_d2 + loss_d3
    )

    return BoundaryLossCartesian3D(
        total=total,
        u=loss_u.detach(),
        sigma=loss_sigma.detach(),
        y1=loss_y1.detach(),
        y2=loss_y2.detach(),
        y3=loss_y3.detach(),
        d1=loss_d1.detach(),
        d2=loss_d2.detach(),
        d3=loss_d3.detach(),
    )
