"""Cell-face interpolation for the 2D escape-equation mesh solver.

In the 3D phase-space, each cell has six faces (±x1, ±x2, ±theta).
Each face spans 4 corner nodes which are interpolated bilinearly.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..helpers.discrete_indices import compute_discrete_indices
from ..helpers.n_linear import n_linear_interpolation
from ..node_state import NodeState

# Paper step 3b: no corners accepted -> don't update (return sentinel).
_NO_ACCEPTED = (np.inf, np.inf, np.nan, np.nan, np.nan)
_OUT_OF_BOUND_RESULT = (np.inf, np.inf, np.nan, np.nan, np.nan)


def _clamp_axis_indices(
    lo: int, hi: int, w: float, N: int
) -> tuple[int, int, float, int]:
    """Clamp an out-of-range index to the boundary. Note that clamping
    should not be the common case and may indicate a regression if it
    happens too often.

    Returns ``(lo, hi, w, fired)``; ``fired`` is 1 when a clamp was applied,
    feeding the per-solve clamp telemetry.
    """
    if lo < 0:
        return 0, 0, 0.0, 1
    if hi >= N:
        return N - 1, N - 1, 0.0, 1
    return lo, hi, w, 0


@dataclass(frozen=True, slots=True)
class FaceInterpolationResult2D:
    """Outcome of interpolating the escape fields on one 2D face.

    ``clamps`` counts intersections pulled to a boundary.
    """

    status: Literal["success", "partial", "no_accepted", "out_of_bounds"]
    values: tuple[float, float, float, float, float] | None = None
    clamps: int = 0


def interpolate_on_face_2d(
    grid,
    hit_face: Literal["x1", "x2", "theta"],
    face_shift: int,
    i: int,
    j: int,
    k: int,
    x1_intersection: float,
    x2_intersection: float,
    theta_intersection: float,
) -> FaceInterpolationResult2D:
    """Interpolate the five escape quantities on the hit face.

    Returns
    -------
    FaceInterpolationResult2D
        - ``success`` with ``values`` when all 4 corners are ACCEPTED.
        - ``no_accepted`` when no corners are ACCEPTED.
        - ``out_of_bounds`` when the interpolated face indices are invalid.
        - ``partial`` when some but not all corners are ACCEPTED.
    """
    N_x1, N_x2, N_theta = grid.N_x1, grid.N_x2, grid.N_theta
    n_clamps = 0

    if hit_face == "x1":
        i_face = i + face_shift
        if not (0 <= i_face < N_x1):
            return FaceInterpolationResult2D(status="out_of_bounds", clamps=n_clamps)

        j0, j1, w1 = compute_discrete_indices(x2_intersection, grid.dx2)
        j0, j1, w1, _c = _clamp_axis_indices(j0, j1, w1, N_x2)
        n_clamps += _c
        # θ half-cell offset: θ_k = (k + theta_offset)·dθ.
        k0, k1, w2 = compute_discrete_indices(
            theta_intersection, grid.dtheta, grid.theta_offset
        )

        if not (0 <= j0 < N_x2 and 0 <= j1 < N_x2):
            return FaceInterpolationResult2D(status="out_of_bounds", clamps=n_clamps)

        k0_w = k0 % N_theta
        k1_w = k1 % N_theta

        corners = [
            (i_face, j0, k0_w),
            (i_face, j1, k0_w),
            (i_face, j0, k1_w),
            (i_face, j1, k1_w),
        ]

    elif hit_face == "x2":
        j_face = j + face_shift
        if not (0 <= j_face < N_x2):
            return FaceInterpolationResult2D(status="out_of_bounds", clamps=n_clamps)

        i0, i1, w1 = compute_discrete_indices(x1_intersection, grid.dx1)
        i0, i1, w1, _c = _clamp_axis_indices(i0, i1, w1, N_x1)
        n_clamps += _c
        # θ half-cell offset: θ_k = (k + theta_offset)·dθ.
        k0, k1, w2 = compute_discrete_indices(
            theta_intersection, grid.dtheta, grid.theta_offset
        )

        if not (0 <= i0 < N_x1 and 0 <= i1 < N_x1):
            return FaceInterpolationResult2D(status="out_of_bounds", clamps=n_clamps)

        k0_w = k0 % N_theta
        k1_w = k1 % N_theta

        corners = [
            (i0, j_face, k0_w),
            (i1, j_face, k0_w),
            (i0, j_face, k1_w),
            (i1, j_face, k1_w),
        ]

    elif hit_face == "theta":
        k_face = (k + face_shift) % N_theta

        i0, i1, w1 = compute_discrete_indices(x1_intersection, grid.dx1)
        i0, i1, w1, _c = _clamp_axis_indices(i0, i1, w1, N_x1)
        n_clamps += _c
        j0, j1, w2 = compute_discrete_indices(x2_intersection, grid.dx2)
        j0, j1, w2, _c = _clamp_axis_indices(j0, j1, w2, N_x2)
        n_clamps += _c

        if not (0 <= i0 < N_x1 and 0 <= i1 < N_x1):
            return FaceInterpolationResult2D(status="out_of_bounds", clamps=n_clamps)
        if not (0 <= j0 < N_x2 and 0 <= j1 < N_x2):
            return FaceInterpolationResult2D(status="out_of_bounds", clamps=n_clamps)

        corners = [
            (i0, j0, k_face),
            (i1, j0, k_face),
            (i0, j1, k_face),
            (i1, j1, k_face),
        ]

    else:
        raise ValueError(f"Invalid hit_face: {hit_face}")

    statuses = [grid.status[c] for c in corners]
    if not all(s == NodeState.ACCEPTED for s in statuses):
        if not any(s == NodeState.ACCEPTED for s in statuses):
            return FaceInterpolationResult2D(status="no_accepted", clamps=n_clamps)
        return FaceInterpolationResult2D(status="partial", clamps=n_clamps)

    # NOTE: angular interpolation is valid as long as the four corners
    # span < π. Wraparound across θ=0/2π is not handled by the bilinear blend.
    return FaceInterpolationResult2D(
        status="success",
        values=n_linear_interpolation(
            (grid.u, grid.sigma, grid.y1, grid.y2, grid.theta_exit),
            corners,
            (w1, w2),
        ),  # type: ignore
        clamps=n_clamps,
    )
