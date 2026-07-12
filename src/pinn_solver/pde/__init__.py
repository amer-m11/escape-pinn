"""PDE residual computation for the Eikonal escape equations.

The package houses dim-specific residual computers:

- :func:`compute_pde_residual_2d`: 2D physical / 3D phase space, 6x3 Jacobian.
- :func:`compute_pde_residual_3d`: 3D physical / 5D phase space, spherical
  ``(φ,θ)`` momentum, 9x5 Jacobian.
- :func:`compute_pde_residual_cartesian_3d`: 3D physical / 6D phase space,
  Cartesian unit vector momentum ``d``, 8x6 Jacobian.

:func:`compute_pde_residual` is a convenience dispatcher keyed on
``model.N_OUTPUTS`` (9 → spherical 3D, 8 → Cartesian 3D, else 2D). Dim-aware
call sites should invoke the specific function so the typing/arity is obvious
at the call site.

TODO: clean this up
"""

from .cartesian_3d import (
    compute_pde_residual_cartesian_3d,
    consistency_residual_cartesian_3d,
)
from .three_d import compute_pde_residual_3d, consistency_residual_3d
from .two_d import compute_pde_residual_2d, consistency_residual_2d


def compute_pde_residual(model, *args, **kwargs):
    """Dispatch to the residual computer based on ``model.N_OUTPUTS``.

    9 → spherical 3D, 8 → Cartesian 3D, else (6) → 2D. Stubs without
    ``N_OUTPUTS`` default to the 2D path.
    """
    n_out = getattr(model, "N_OUTPUTS", 6)
    if n_out == 9:
        return compute_pde_residual_3d(model, *args, **kwargs)
    if n_out == 8:
        return compute_pde_residual_cartesian_3d(model, *args, **kwargs)
    return compute_pde_residual_2d(model, *args, **kwargs)


__all__ = [
    "compute_pde_residual",
    "compute_pde_residual_2d",
    "compute_pde_residual_3d",
    "compute_pde_residual_cartesian_3d",
    "consistency_residual_2d",
    "consistency_residual_3d",
    "consistency_residual_cartesian_3d",
]
