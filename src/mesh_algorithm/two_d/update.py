"""Local cell characteristic for the 2D mesh solver.

Traces the parabolic-ray characteristic from a grid node forward to the
first cell face, interpolates the five escape quantities there, and adds
the per-segment travel time to produce a tentative value at the node. If
the hit face is only partially accepted, the trace continues into the next
cell up to the grid's L1 cell-diameter.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..diagnostics import FailureReason
from ..helpers.quadratic import solve_parabolic_intersection
from .face import interpolate_on_face_2d


def _max_continuation_for_grid(N_x1, N_x2, N_theta):
    """L1 cell-diameter of the 3D phase-space grid."""
    return N_x1 + N_x2 + N_theta


@dataclass(frozen=True, slots=True)
class CharacteristicResult2D:
    """Tentative update produced by the 2D local-cell characteristic.

    Contains the escape quantities (u, σ, y1, y2, θ_exit) plus diagnostic"""

    u: float
    sigma: float
    y1: float
    y2: float
    theta_exit: float
    failure_reason: int | None = None
    depth: int = 0
    """The number of continuation cells the trace traversed"""
    clamps: int = 0
    """Number of intersections clamped to the boundary during face interpolation"""

    @classmethod
    def unreachable(
        cls,
        failure_reason: int = FailureReason.NO_FACE_CANDIDATE,
        depth: int = 0,
        clamps: int = 0,
    ) -> "CharacteristicResult2D":
        """Factory for a sentinel result representing an unreachable node."""
        return cls(
            u=np.inf,
            sigma=np.inf,
            y1=np.nan,
            y2=np.nan,
            theta_exit=np.nan,
            failure_reason=int(failure_reason),
            depth=int(depth),
            clamps=int(clamps),
        )


def local_cell_characteristic_2d(
    grid, idx: tuple[int, int, int]
) -> CharacteristicResult2D:
    """Compute a tentative value for ``û_{ijk}`` via parabolic ray tracing.

    Trace forward along the ray direction θ using the parabolic
    approximation:

        x1(s) = x1₀ + n·cos(θ)·s + (1/4)·∂n²/∂x1 · s²
        x2(s) = x2₀ + n·sin(θ)·s + (1/4)·∂n²/∂x2 · s²
        θ(s)  = θ₀  + grad_theta · s

    Find the first cell face hit, interpolate from ACCEPTED neighbors
    there, and add the travel time ``û = interpolated_û + n²·Δσ``. If the
    hit face is only partially accepted, continue tracing into the next
    cell up to the grid's L1 cell-diameter (
    :func:`_max_continuation_for_grid`). If no face along the trace has
    all corners accepted, return the unreachable sentinel
    (``inf, inf, nan, nan, nan``).
    """
    i, j, k = idx
    max_continuation_cells = _max_continuation_for_grid(
        grid.N_x1, grid.N_x2, grid.N_theta
    )
    theta = grid.theta_coords[k]
    cos = math.cos(theta)
    sin = math.sin(theta)
    n0 = grid.slowness_at(i, j)
    n0_sq = n0**2

    dn2_dx1, dn2_dx2 = grid.slowness_gradient(i, j)

    # Angular drift: dθ/dσ from the Hamiltonian ray equations
    grad_theta = (cos * dn2_dx2 - sin * dn2_dx1) / (2 * n0)

    # Current position in physical coordinates
    current_x1 = grid.x1_coords[i]
    current_x2 = grid.x2_coords[j]
    current_theta = theta

    total_travel_time = 0.0
    total_sigma = 0.0
    total_clamps = 0

    # Current cell indices
    ci, cj, ck = i, j, k

    # Current local slowness and gradients
    current_n = n0
    current_n_squared = n0_sq
    current_dn2_dx1, current_dn2_dx2 = dn2_dx1, dn2_dx2
    current_cos, current_sin = cos, sin
    current_grad_theta = grad_theta

    for _iter_idx in range(max_continuation_cells):
        cells_traversed = _iter_idx + 1
        candidates = []  # (s, face_type, face_index_shift)

        # ===== Calculate time needed to hit next grid plane in each dimension =====

        # -- x1 face --
        if current_cos > 0 and ci + 1 < grid.N_x1:
            # Ray moves right => hits face at ci+1
            next_x1 = grid.x1_coords[ci + 1]
            face_shift_i = +1
        elif current_cos < 0 and ci >= 1:
            # Ray moves left => hits face at ci-1
            next_x1 = grid.x1_coords[ci - 1]
            face_shift_i = -1
        else:
            # Ray moves vertically or points out of bounds in x1 direction
            next_x1 = current_x1
            face_shift_i = 0

        t_x1 = (
            solve_parabolic_intersection(
                position=current_x1,
                velocity=current_n * current_cos,
                acceleration=0.25 * current_dn2_dx1,
                boundary=next_x1,
            )
            if face_shift_i != 0
            else float("inf")
        )
        if t_x1 > 1e-15 and t_x1 < math.inf:
            candidates.append((t_x1, "x1", face_shift_i))

        # -- x2 face --
        if current_sin > 0 and cj + 1 < grid.N_x2:
            next_x2 = grid.x2_coords[cj + 1]
            face_shift_j = +1
        elif current_sin < 0 and cj >= 1:
            next_x2 = grid.x2_coords[cj - 1]
            face_shift_j = -1
        else:
            next_x2 = current_x2
            face_shift_j = 0

        t_x2 = (
            solve_parabolic_intersection(
                position=current_x2,
                velocity=current_n * current_sin,
                acceleration=0.25 * current_dn2_dx2,
                boundary=next_x2,
            )
            if face_shift_j != 0
            else float("inf")
        )
        if t_x2 > 1e-15 and t_x2 < math.inf:
            candidates.append((t_x2, "x2", face_shift_j))

        # -- θ face --
        if abs(current_grad_theta) > 1e-15:
            if current_grad_theta > 0:
                face_shift_k = +1
                target_theta = grid.theta_coords[(ck + 1) % grid.N_theta]
                delta_theta = target_theta - current_theta
                if delta_theta <= 0:
                    delta_theta += 2 * math.pi
            else:
                face_shift_k = -1
                target_theta = grid.theta_coords[(ck - 1) % grid.N_theta]
                delta_theta = target_theta - current_theta
                if delta_theta >= 0:
                    delta_theta -= 2 * math.pi

            t_theta = delta_theta / current_grad_theta
            if t_theta > 1e-15:
                candidates.append((t_theta, "theta", face_shift_k))

        if not candidates:
            return CharacteristicResult2D.unreachable(
                FailureReason.NO_FACE_CANDIDATE,
                depth=cells_traversed,
                clamps=total_clamps,
            )

        # ===== Select the minimum time and compute intersection coordinates =====

        # Pick the face hit first (smallest positive t)
        candidates.sort(key=lambda c: c[0])
        min_t, hit_face, face_shift = candidates[0]

        if min_t == float("inf"):
            raise ValueError("Ray does not hit any face, should never happen")
        if face_shift == 0:
            raise ValueError("Ray does not move in this direction, should never happen")

        # Compute intersection coordinates
        x1_hit = (
            current_x1
            + current_n * current_cos * min_t
            + 0.25 * current_dn2_dx1 * min_t**2
        )
        x2_hit = (
            current_x2
            + current_n * current_sin * min_t
            + 0.25 * current_dn2_dx2 * min_t**2
        )
        theta_hit = current_theta + current_grad_theta * min_t

        # Accumulate travel time: du/dσ = n² => du = n² dσ.
        # NOTE: uses n² at the cell's grid node. For better accuracy, we
        # should evaluate n² at the intersection point or interpolate it.
        total_travel_time += current_n_squared * min_t
        total_sigma += min_t

        # --- Bilinear interpolation on the hit face ---
        face_result = interpolate_on_face_2d(
            grid,
            hit_face,
            face_shift,
            ci,
            cj,
            ck,
            x1_hit,
            x2_hit,
            theta_hit,
        )
        total_clamps += face_result.clamps

        if face_result.status == "success":
            assert face_result.values is not None
            interp_u, interp_sigma, interp_y1, interp_y2, interp_theta = (
                face_result.values
            )
            return CharacteristicResult2D(
                u=interp_u + total_travel_time,
                sigma=interp_sigma + total_sigma,
                y1=interp_y1,
                y2=interp_y2,
                theta_exit=interp_theta,
                depth=cells_traversed,
                clamps=total_clamps,
            )
        if face_result.status == "no_accepted":
            return CharacteristicResult2D.unreachable(
                FailureReason.FACE_WITHOUT_ACCEPTED_CORNERS,
                depth=cells_traversed,
                clamps=total_clamps,
            )
        if face_result.status == "out_of_bounds":
            return CharacteristicResult2D.unreachable(
                FailureReason.FACE_INTERPOLATION_OUT_OF_BOUNDS,
                depth=cells_traversed,
                clamps=total_clamps,
            )

        # Paper step 3c: partially accepted or no accepted nodes.
        # Continue tracing from the intersection point into the next cell.
        current_x1, current_x2, current_theta = x1_hit, x2_hit, theta_hit

        # Advance every face whose hit time is tied with min_t (within tolerance).
        tie_tol = 1e-12 + 1e-9 * min_t
        for t_k, face_k, shift_k in candidates:
            if t_k - min_t > tie_tol:
                continue
            if face_k == "x1":
                ci = ci + shift_k
            elif face_k == "x2":
                cj = cj + shift_k
            elif face_k == "theta":
                ck = (ck + shift_k) % grid.N_theta
            else:
                raise ValueError(f"Invalid hit face: {face_k}")

        if not (0 <= ci < grid.N_x1 and 0 <= cj < grid.N_x2):
            return CharacteristicResult2D.unreachable(
                FailureReason.CONTINUATION_SPATIAL_OOB,
                depth=cells_traversed,
                clamps=total_clamps,
            )

        # Update local slowness for the new cell
        current_n = grid.slowness_at(ci, cj)
        current_n_squared = current_n**2
        current_dn2_dx1, current_dn2_dx2 = grid.slowness_gradient(ci, cj)

        # Update direction (θ may have drifted)
        current_cos = math.cos(current_theta)
        current_sin = math.sin(current_theta)
        current_grad_theta = (
            current_cos * current_dn2_dx2 - current_sin * current_dn2_dx1
        ) / (2 * current_n)

    # Exhausted continuation attempts
    return CharacteristicResult2D.unreachable(
        FailureReason.CONTINUATION_EXHAUSTED,
        depth=max_continuation_cells,
        clamps=total_clamps,
    )
