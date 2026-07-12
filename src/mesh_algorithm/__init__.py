"""Marching algorithm for escape time computation.

Public API
----------
- :func:`PhaseSpaceGrid` - factory: dispatches on the length of
  ``physical_size`` to :class:`PhaseSpaceGrid2D` (len 2) or
  ``PhaseSpaceGrid3D`` (len 3).
- :class:`EscapeSolver` - one-pass mesh solver. Works on either grid implementation.
- :class:`NodeState` — node state enum used by both grid implementations.


Assumptions / Known limitations:
    - Angle continuation uses a linear approximation.
    - The angular dimension(s) periodic in θ wrap. In 3D φ is non-periodic.
    - The heap is ordered by σ̂. Both û and σ̂ increase monotonically along
      characteristics. σ̂ is parametric distance independent of the medium,
      so it ensures nodes closest to the boundary are accepted first even
      in slow media.
    - If the local cell characteristic finds no fully-accepted face along
      the trace, it returns the unreachable sentinel (σ̂ = ∞).
    - Travel-time accumulation uses n² at the cell's grid node (does not
      interpolate n² at the intersection point).
    - Pole sensitivity in 3D: the angular drift dθ/dσ in 3D carries
      a ``1/sin φ`` factor that diverges at the poles. The half-cell
      offset ``φ_m = (m + 0.5)·π/N_φ`` keeps ``sin φ`` bounded away from
      zero but does not eliminate the sensitivity at low N_φ.
"""

from .node_state import NodeState
from .solver import EscapeSolver
from .three_d.grid import PhaseSpaceGrid3D
from .two_d.grid import PhaseSpaceGrid2D


def PhaseSpaceGrid(
    grid_size: tuple[int, ...],
    physical_size: tuple[float, ...],
    slowness_fn,
    slowness_gradient_fn=None,
):
    """Build a phase-space grid for the 2D or 3D escape equations.

    Dispatches by ``len(physical_size)``:

    - ``len == 2`` → :class:`mesh_algorithm.two_d.grid.PhaseSpaceGrid2D`
      with ``grid_size = (N_x1, N_x2, N_theta)`` and
      ``physical_size = (L_x1, L_x2)``.
    - ``len == 3`` → :class:`mesh_algorithm.three_d.grid.PhaseSpaceGrid3D`
      with ``grid_size = (N_x1, N_x2, N_x3, N_phi, N_theta)`` and
      ``physical_size = (L_x1, L_x2, L_x3)``.

    Parameters
    ----------
    grid_size : tuple of int
        Number of grid points along each axis (spatial + angular).
    physical_size : tuple of float
        Physical extent along each spatial axis.
    slowness_fn : callable
        Slowness field ``n(x...) -> float``. Must be positive.
    slowness_gradient_fn : callable, optional
        Analytic gradient of n² with the same input arity as
        ``slowness_fn``. If ``None`` the gradient is computed numerically.
    """
    n_spatial = len(physical_size)
    if n_spatial == 2:
        return PhaseSpaceGrid2D(
            grid_size, physical_size, slowness_fn, slowness_gradient_fn
        )
    if n_spatial == 3:
        return PhaseSpaceGrid3D(
            grid_size, physical_size, slowness_fn, slowness_gradient_fn
        )
    raise ValueError(f"Unsupported physical dimension {n_spatial}; expected 2 or 3.")


__all__ = [
    "EscapeSolver",
    "NodeState",
    "PhaseSpaceGrid",
    "PhaseSpaceGrid2D",
    "PhaseSpaceGrid3D",
]
