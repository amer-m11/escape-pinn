"""Boundary condition sampling and loss for the Eikonal escape equations.

The package houses dim/parametrization-specific implementations:

- :mod:`pinn_solver.boundary.two_d`: 4 edges of a rectangle, 6 BC channels.
- :mod:`pinn_solver.boundary.three_d`: 6 faces of a cuboid, 9 BC channels
  (spherical ``(φ,θ)`` momentum).
- :mod:`pinn_solver.boundary.cartesian_3d`: 6 faces, 8 BC channels
  (Cartesian unit-vector momentum ``d``).

The unqualified names are back-compat for existing call sites:
``BoundaryLoss`` and ``bc_sampling_audit`` are plain 2D aliases, while
``compute_boundary_loss`` and ``sample_boundary_points`` dispatch to the
right variant (by ``model.N_OUTPUTS`` / ``len(physical_size)``). 3D-aware
call sites should use the explicit ``_3d`` / ``_cartesian_3d`` suffix for
clarity.

TODO: clean this up
"""

from .cartesian_3d import (
    BoundaryLossCartesian3D,
    bc_sampling_audit_cartesian_3d,
    compute_boundary_loss_cartesian_3d,
    sample_boundary_points_cartesian_3d,
)
from .three_d import (
    BoundaryLoss3D,
    bc_sampling_audit_3d,
    compute_boundary_loss_3d,
    sample_boundary_points_3d,
)
from .two_d import (
    BoundaryLoss2D,
    bc_sampling_audit_2d,
    compute_boundary_loss_2d,
    sample_boundary_points_2d,
)

# Back-compat
# Existing 2D call sites import these.
BoundaryLoss = BoundaryLoss2D
bc_sampling_audit = bc_sampling_audit_2d


def compute_boundary_loss(model, *args, **kwargs):
    """Dispatch the boundary loss based on ``model.N_OUTPUTS``.

    9 -> spherical 3D, 8 -> Cartesian 3D, else (6) -> 2D. Stubs without
    ``N_OUTPUTS`` default to the 2D path.
    """
    n_out = getattr(model, "N_OUTPUTS", 6)
    if n_out == 9:
        return compute_boundary_loss_3d(model, *args, **kwargs)
    if n_out == 8:
        return compute_boundary_loss_cartesian_3d(model, *args, **kwargs)
    return compute_boundary_loss_2d(model, *args, **kwargs)


def sample_boundary_points(physical_size, *args, **kwargs):
    """Dispatch to the dim-specific sampler based on ``len(physical_size)``.

    2-tuple -> 2D path, 3-tuple -> 3D spherical path. The 3D sampler requires
    ``n_phi`` (passed via ``**kwargs``). The Cartesian boundary sampler is
    not reachable through this dispatcher, call
    :func:`sample_boundary_points_cartesian_3d` explicitly.
    """
    if len(physical_size) == 3:
        return sample_boundary_points_3d(physical_size, *args, **kwargs)
    return sample_boundary_points_2d(physical_size, *args, **kwargs)


__all__ = [
    "BoundaryLoss",
    "BoundaryLoss2D",
    "BoundaryLoss3D",
    "BoundaryLossCartesian3D",
    "bc_sampling_audit",
    "bc_sampling_audit_2d",
    "bc_sampling_audit_3d",
    "bc_sampling_audit_cartesian_3d",
    "compute_boundary_loss",
    "compute_boundary_loss_2d",
    "compute_boundary_loss_3d",
    "compute_boundary_loss_cartesian_3d",
    "sample_boundary_points",
    "sample_boundary_points_2d",
    "sample_boundary_points_3d",
    "sample_boundary_points_cartesian_3d",
]
