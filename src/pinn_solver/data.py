"""Convert a mesh_algorithm solution to PINN reference data.

The mesh algorithm computes the escape quantities on a discrete phase-space
grid. :func:`grid_to_tensors` flattens the accepted nodes into 1-D float32
tensors packaged as a :class:`ReferenceSolution` (2D) or
:class:`ReferenceSolution3D` (3D). The right dataclass is selected
automatically by inspecting the grid object (looks for the
``x3_coords`` attribute).
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReferenceSolution:
    """2D mesh-solver output flattened to 1-D tensors over accepted nodes.

    Field names mirror ``mesh_algorithm.PhaseSpaceGrid2D`` attributes plus
    the three phase-space coordinates the trainer needs to evaluate the
    network at. All tensors share length ``N_accepted`` and live on the
    same device.
    """

    x1: torch.Tensor
    x2: torch.Tensor
    theta: torch.Tensor
    u: torch.Tensor
    sigma: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor
    theta_exit: torch.Tensor
    cos_theta_exit: torch.Tensor
    sin_theta_exit: torch.Tensor


@dataclass
class ReferenceSolution3D:
    """3D mesh-solver output flattened to 1-D tensors over accepted nodes.

    Adds ``x3``, ``phi`` phase-space inputs and ``y3``, ``phi_exit``
    targets (plus the (cos, sin) embedding of ``phi_exit``). All tensors
    share length ``N_accepted`` and live on the same device.
    """

    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    phi: torch.Tensor
    theta: torch.Tensor
    u: torch.Tensor
    sigma: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor
    y3: torch.Tensor
    theta_exit: torch.Tensor
    phi_exit: torch.Tensor
    cos_theta_exit: torch.Tensor
    sin_theta_exit: torch.Tensor
    cos_phi_exit: torch.Tensor
    sin_phi_exit: torch.Tensor


def grid_to_tensors(
    grid,
    device: torch.device | None = None,
) -> ReferenceSolution | ReferenceSolution3D:
    """Pack accepted mesh-solver nodes into a reference dataclass.

    A node is "accepted" iff every escape quantity at that node is finite.
    Dispatches on ``hasattr(grid, "x3_coords")`` to handle both 2D
    (``PhaseSpaceGrid2D``) and 3D (``PhaseSpaceGrid3D``) grids.
    """
    if hasattr(grid, "x3_coords"):
        return _grid_to_tensors_3d(grid, device=device)
    return _grid_to_tensors_2d(grid, device=device)


def _grid_to_tensors_2d(grid, device: torch.device | None) -> ReferenceSolution:
    finite_mask = (
        np.isfinite(grid.u)
        & np.isfinite(grid.sigma)
        & np.isfinite(grid.y1)
        & np.isfinite(grid.y2)
        & np.isfinite(grid.theta_exit)
    )

    full_shape = (grid.N_x1, grid.N_x2, grid.N_theta)
    x1_full = np.broadcast_to(grid.x1_coords[:, None, None], full_shape)
    x2_full = np.broadcast_to(grid.x2_coords[None, :, None], full_shape)
    theta_full = np.broadcast_to(grid.theta_coords[None, None, :], full_shape)

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr[finite_mask], dtype=torch.float32, device=device)

    return ReferenceSolution(
        x1=_t(x1_full),
        x2=_t(x2_full),
        theta=_t(theta_full),
        u=_t(grid.u),
        sigma=_t(grid.sigma),
        y1=_t(grid.y1),
        y2=_t(grid.y2),
        theta_exit=_t(grid.theta_exit),
        cos_theta_exit=torch.cos(_t(grid.theta_exit)),
        sin_theta_exit=torch.sin(_t(grid.theta_exit)),
    )


def _grid_to_tensors_3d(grid, device: torch.device | None) -> ReferenceSolution3D:
    finite_mask = (
        np.isfinite(grid.u)
        & np.isfinite(grid.sigma)
        & np.isfinite(grid.y1)
        & np.isfinite(grid.y2)
        & np.isfinite(grid.y3)
        & np.isfinite(grid.theta_exit)
        & np.isfinite(grid.phi_exit)
    )

    full_shape = (grid.N_x1, grid.N_x2, grid.N_x3, grid.N_phi, grid.N_theta)
    x1_full = np.broadcast_to(grid.x1_coords[:, None, None, None, None], full_shape)
    x2_full = np.broadcast_to(grid.x2_coords[None, :, None, None, None], full_shape)
    x3_full = np.broadcast_to(grid.x3_coords[None, None, :, None, None], full_shape)
    phi_full = np.broadcast_to(grid.phi_coords[None, None, None, :, None], full_shape)
    theta_full = np.broadcast_to(grid.theta_coords[None, None, None, None, :], full_shape)

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr[finite_mask], dtype=torch.float32, device=device)

    theta_exit = _t(grid.theta_exit)
    phi_exit = _t(grid.phi_exit)
    return ReferenceSolution3D(
        x1=_t(x1_full),
        x2=_t(x2_full),
        x3=_t(x3_full),
        phi=_t(phi_full),
        theta=_t(theta_full),
        u=_t(grid.u),
        sigma=_t(grid.sigma),
        y1=_t(grid.y1),
        y2=_t(grid.y2),
        y3=_t(grid.y3),
        theta_exit=theta_exit,
        phi_exit=phi_exit,
        cos_theta_exit=torch.cos(theta_exit),
        sin_theta_exit=torch.sin(theta_exit),
        cos_phi_exit=torch.cos(phi_exit),
        sin_phi_exit=torch.sin(phi_exit),
    )


def subsample_reference(
    reference: ReferenceSolution | ReferenceSolution3D,
    max_points: int | None,
    *,
    generator: torch.Generator | None = None,
) -> ReferenceSolution | ReferenceSolution3D:
    """Return a fixed-size random subset of a reference dataclass.

    When ``max_points`` is ``None`` or already exceeds the number of
    accepted nodes, the original dataclass is returned unchanged. Works
    on both 2D and 3D reference variants.
    """
    if max_points is None:
        return reference
    if max_points <= 0:
        raise ValueError("max_points must be positive when provided")

    n_points = int(reference.x1.numel())
    if n_points <= max_points:
        return reference

    index = torch.randperm(
        n_points,
        device=reference.x1.device,
        generator=generator,
    )[:max_points]

    if isinstance(reference, ReferenceSolution3D):
        return ReferenceSolution3D(
            x1=reference.x1[index],
            x2=reference.x2[index],
            x3=reference.x3[index],
            phi=reference.phi[index],
            theta=reference.theta[index],
            u=reference.u[index],
            sigma=reference.sigma[index],
            y1=reference.y1[index],
            y2=reference.y2[index],
            y3=reference.y3[index],
            theta_exit=reference.theta_exit[index],
            phi_exit=reference.phi_exit[index],
            cos_theta_exit=reference.cos_theta_exit[index],
            sin_theta_exit=reference.sin_theta_exit[index],
            cos_phi_exit=reference.cos_phi_exit[index],
            sin_phi_exit=reference.sin_phi_exit[index],
        )
    return ReferenceSolution(
        x1=reference.x1[index],
        x2=reference.x2[index],
        theta=reference.theta[index],
        u=reference.u[index],
        sigma=reference.sigma[index],
        y1=reference.y1[index],
        y2=reference.y2[index],
        theta_exit=reference.theta_exit[index],
        cos_theta_exit=reference.cos_theta_exit[index],
        sin_theta_exit=reference.sin_theta_exit[index],
    )
