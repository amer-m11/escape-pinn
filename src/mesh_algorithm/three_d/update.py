"""Local cell characteristic for the 3D mesh solver.

Traces the parabolic ray characteristic from a grid node forward through
the 5D phase space cell to the first face hit, then interpolates the
seven escape quantities at the hit point. Mirrors
:mod:`mesh_algorithm.two_d.update`
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..diagnostics import FailureReason
from ..helpers.quadratic import solve_parabolic_intersection
from .face import (
    _JFACE_NO_ACCEPTED,
    _JFACE_OOB,
    _JFACE_SUCCESS,
    _NODE_ACCEPTED,
    FACE_PHI,
    FACE_THETA,
    FACE_X1,
    FACE_X2,
    FACE_X3,
    _interpolate_on_face_3d_jit,
)


# Paper step 3c: continue tracing until all-accepted or no-accepted face.
def _max_continuation_for_grid(N_x1, N_x2, N_x3, N_phi, N_theta):
    """L1 cell-diameter of the 5D phase-space grid."""
    return N_x1 + N_x2 + N_x3 + N_phi + N_theta


@dataclass(frozen=True, slots=True)
class CharacteristicResult3D:
    """Tentative update produced by the 3D local-cell characteristic."""

    u: float
    sigma: float
    y1: float
    y2: float
    y3: float
    phi_exit: float
    theta_exit: float
    failure_reason: int | None = None
    depth: int = 0
    """Number of continuation cells the trace traversed (1 = first cell, no continuation)."""
    clamps: int = 0
    """Counts brackets pulled back to a boundary node during face interpolation."""

    @classmethod
    def unreachable(
        cls,
        failure_reason: int = FailureReason.NO_FACE_CANDIDATE,
        depth: int = 0,
        clamps: int = 0,
    ) -> "CharacteristicResult3D":
        """Factory for a sentinel result representing an unreachable node."""
        return cls(
            u=np.inf,
            sigma=np.inf,
            y1=np.nan,
            y2=np.nan,
            y3=np.nan,
            phi_exit=np.nan,
            theta_exit=np.nan,
            failure_reason=int(failure_reason),
            depth=int(depth),
            clamps=int(clamps),
        )


def local_cell_characteristic_3d(
    grid, idx: tuple[int, int, int, int, int]
) -> CharacteristicResult3D:
    """Compute a tentative value for ``û_{ijlmn}`` via parabolic ray tracing.

    Trace forward along the ray direction ``d = (sin φ cos θ, sin φ sin θ,
    cos φ)`` using the parabolic approximation per spatial axis and the
    linear approximation per angular axis:

        x1(s) = x1₀ + n·d1·s + (1/4)·∂n²/∂x1 · s²
        x2(s) = x2₀ + n·d2·s + (1/4)·∂n²/∂x2 · s²
        x3(s) = x3₀ + n·d3·s + (1/4)·∂n²/∂x3 · s²
        φ(s)  = φ₀  + grad_phi   · s
        θ(s)  = θ₀  + grad_theta · s

    Pick the first face hit, interpolate from ACCEPTED neighbors
    (quadrilinear, 16 corners), and add ``û = interpolated_û + n²·Δσ``.
    On a partially-accepted face continue tracing into the next cell up to
    the grid's L1 cell-diameter (:func:`_max_continuation_for_grid`).
    If no face along the trace has all corners accepted, return the
    unreachable sentinel.

    Delegates to :func:`_local_cell_char_3d_jit` for the numeric work.
    """
    i, j, l, m, n = idx
    max_continuation_cells = _max_continuation_for_grid(
        grid.N_x1, grid.N_x2, grid.N_x3, grid.N_phi, grid.N_theta
    )
    status_int, depth, clamps, values = _local_cell_char_3d_jit(
        i,
        j,
        l,
        m,
        n,
        grid.status,
        grid.u,
        grid.sigma,
        grid.y1,
        grid.y2,
        grid.y3,
        grid.phi_exit,
        grid.theta_exit,
        grid.x1_coords,
        grid.x2_coords,
        grid.x3_coords,
        grid.phi_coords,
        grid.theta_coords,
        grid.dx1,
        grid.dx2,
        grid.dx3,
        grid.dphi,
        grid.dtheta,
        grid.N_x1,
        grid.N_x2,
        grid.N_x3,
        grid.N_phi,
        grid.N_theta,
        grid._n,
        grid._n_grad,
        max_continuation_cells,
    )
    if status_int == _CHAR_SUCCESS:
        return CharacteristicResult3D(
            u=float(values[0]),
            sigma=float(values[1]),
            y1=float(values[2]),
            y2=float(values[3]),
            y3=float(values[4]),
            phi_exit=float(values[5]),
            theta_exit=float(values[6]),
            depth=int(depth),
            clamps=int(clamps),
        )
    return CharacteristicResult3D.unreachable(
        FailureReason(status_int), depth=int(depth), clamps=int(clamps)
    )


# ---------------------------------------------------------------------------
# JIT path: all arguments are plain numpy arrays and scalars so that
# Numba can compile without Python object overhead.
# ---------------------------------------------------------------------------

# Return status codes (0 = success, 1–6 match FailureReason integer values).
_CHAR_SUCCESS = 0
_CHAR_NO_CAND = int(FailureReason.NO_FACE_CANDIDATE)
_CHAR_NO_ACCEPTED = int(FailureReason.FACE_WITHOUT_ACCEPTED_CORNERS)
_CHAR_OOB = int(FailureReason.FACE_INTERPOLATION_OUT_OF_BOUNDS)
_CHAR_SPATIAL_OOB = int(FailureReason.CONTINUATION_SPATIAL_OOB)
_CHAR_PHI_OOB = int(FailureReason.CONTINUATION_PHI_OOB)
_CHAR_EXHAUSTED = int(FailureReason.CONTINUATION_EXHAUSTED)


@njit(cache=True)
def _local_cell_char_3d_jit(
    i,
    j,
    l,
    m,
    n,
    status_arr,
    u_arr,
    sigma_arr,
    y1_arr,
    y2_arr,
    y3_arr,
    phi_exit_arr,
    theta_exit_arr,
    x1_coords,
    x2_coords,
    x3_coords,
    phi_coords,
    theta_coords,
    dx1,
    dx2,
    dx3,
    dphi,
    dtheta,
    N_x1,
    N_x2,
    N_x3,
    N_phi,
    N_theta,
    n_arr,
    dn2_arr,
    max_continuation_cells,
):
    """JIT-compiled per-node characteristic tracer.

    Parameters mirror the fields of :class:`PhaseSpaceGrid3D` passed as
    plain numpy arrays plus integer/float scalars.  Returns
    ``(status_int, depth_int, n_clamps, float64[7])`` where
    ``status_int == _CHAR_SUCCESS (0)`` signals success, ``depth_int`` is the
    number of continuation cells traversed, ``n_clamps`` is the total bracket
    clamp fires across every face interpolation in the trace, and the array
    holds ``[û, σ̂, ŷ1, ŷ2, ŷ3, φ̂_exit, θ̂_exit]``. All other status codes map
    directly to :class:`FailureReason` integer values (1-6).

    Module-level integer constants (``FACE_X1``, ``_CHAR_SUCCESS``, etc.)
    are captured by Numba at compile time. So, no need to pass them as arguments.
    """
    dummy = np.zeros(7)
    total_clamps = 0

    phi = phi_coords[m]
    theta = theta_coords[n]
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    n0 = n_arr[i, j, l]
    n0_sq = n0 * n0

    dn2_dx1 = dn2_arr[i, j, l, 0]
    dn2_dx2 = dn2_arr[i, j, l, 1]
    dn2_dx3 = dn2_arr[i, j, l, 2]

    grad_phi = (
        cos_phi * cos_theta * dn2_dx1
        + cos_phi * sin_theta * dn2_dx2
        - sin_phi * dn2_dx3
    ) / (2.0 * n0)
    grad_theta = (-sin_theta * dn2_dx1 + cos_theta * dn2_dx2) / (2.0 * n0 * sin_phi)

    current_x1 = x1_coords[i]
    current_x2 = x2_coords[j]
    current_x3 = x3_coords[l]
    current_phi = phi
    current_theta = theta

    total_travel_time = 0.0
    total_sigma = 0.0

    # Current cell indices
    ci, cj, cl, cm, cn = i, j, l, m, n

    # Cached local quantities (refreshed on continuation)
    current_n = n0
    current_n_sq = n0_sq
    current_dn2_dx1 = dn2_dx1
    current_dn2_dx2 = dn2_dx2
    current_dn2_dx3 = dn2_dx3
    current_sin_phi = sin_phi
    current_cos_phi = cos_phi
    current_cos_theta = cos_theta
    current_sin_theta = sin_theta
    current_grad_phi = grad_phi
    current_grad_theta = grad_theta

    # Fixed-size candidate buffers (at most 5: x1, x2, x3, phi, theta).
    cand_t = np.empty(5)
    cand_face = np.empty(5, dtype=np.int64)
    cand_shift = np.empty(5, dtype=np.int64)

    for _iter_idx in range(max_continuation_cells):
        cells_traversed = _iter_idx + 1
        n_cands = 0

        # Spatial velocity components: v_a = n · d_a.
        v_x1 = current_n * current_sin_phi * current_cos_theta
        v_x2 = current_n * current_sin_phi * current_sin_theta
        v_x3 = current_n * current_cos_phi

        # ----- x1 face -----
        if v_x1 > 0.0 and ci + 1 < N_x1:
            next_x1 = x1_coords[ci + 1]
            fs_i = 1
        elif v_x1 < 0.0 and ci >= 1:
            next_x1 = x1_coords[ci - 1]
            fs_i = -1
        else:
            fs_i = 0
            next_x1 = current_x1
        if fs_i != 0:
            t = solve_parabolic_intersection(
                current_x1, v_x1, 0.25 * current_dn2_dx1, next_x1
            )
            if t > 1e-15 and t < math.inf:
                cand_t[n_cands] = t
                cand_face[n_cands] = FACE_X1
                cand_shift[n_cands] = fs_i
                n_cands += 1

        # ----- x2 face -----
        if v_x2 > 0.0 and cj + 1 < N_x2:
            next_x2 = x2_coords[cj + 1]
            fs_j = 1
        elif v_x2 < 0.0 and cj >= 1:
            next_x2 = x2_coords[cj - 1]
            fs_j = -1
        else:
            fs_j = 0
            next_x2 = current_x2
        if fs_j != 0:
            t = solve_parabolic_intersection(
                current_x2, v_x2, 0.25 * current_dn2_dx2, next_x2
            )
            if t > 1e-15 and t < math.inf:
                cand_t[n_cands] = t
                cand_face[n_cands] = FACE_X2
                cand_shift[n_cands] = fs_j
                n_cands += 1

        # ----- x3 face -----
        if v_x3 > 0.0 and cl + 1 < N_x3:
            next_x3 = x3_coords[cl + 1]
            fs_l = 1
        elif v_x3 < 0.0 and cl >= 1:
            next_x3 = x3_coords[cl - 1]
            fs_l = -1
        else:
            fs_l = 0
            next_x3 = current_x3
        if fs_l != 0:
            t = solve_parabolic_intersection(
                current_x3, v_x3, 0.25 * current_dn2_dx3, next_x3
            )
            if t > 1e-15 and t < math.inf:
                cand_t[n_cands] = t
                cand_face[n_cands] = FACE_X3
                cand_shift[n_cands] = fs_l
                n_cands += 1

        # ----- φ face (non-periodic, linear) -----
        if abs(current_grad_phi) > 1e-15:
            if current_grad_phi > 0.0 and cm + 1 < N_phi:
                fs_m = 1
                target_phi = phi_coords[cm + 1]
            elif current_grad_phi < 0.0 and cm >= 1:
                fs_m = -1
                target_phi = phi_coords[cm - 1]
            else:
                fs_m = 0
                target_phi = current_phi
            if fs_m != 0:
                t_phi = (target_phi - current_phi) / current_grad_phi
                if t_phi > 1e-15:
                    cand_t[n_cands] = t_phi
                    cand_face[n_cands] = FACE_PHI
                    cand_shift[n_cands] = fs_m
                    n_cands += 1

        # ----- θ face (periodic, linear) -----
        if abs(current_grad_theta) > 1e-15:
            if current_grad_theta > 0.0:
                fs_n = 1
                target_theta = theta_coords[(cn + 1) % N_theta]
                delta_theta = target_theta - current_theta
                if delta_theta <= 0.0:
                    delta_theta += 2.0 * math.pi
            else:
                fs_n = -1
                target_theta = theta_coords[(cn - 1) % N_theta]
                delta_theta = target_theta - current_theta
                if delta_theta >= 0.0:
                    delta_theta -= 2.0 * math.pi
            t_theta = delta_theta / current_grad_theta
            if t_theta > 1e-15:
                cand_t[n_cands] = t_theta
                cand_face[n_cands] = FACE_THETA
                cand_shift[n_cands] = fs_n
                n_cands += 1

        if n_cands == 0:
            return _CHAR_NO_CAND, cells_traversed, total_clamps, dummy

        # Find argmin over the n_cands candidates.
        best_k = 0
        for k in range(1, n_cands):
            if cand_t[k] < cand_t[best_k]:
                best_k = k
        min_t = cand_t[best_k]
        hit_face_int = cand_face[best_k]
        face_shift = cand_shift[best_k]

        # ----- Intersection coordinates -----
        x1_hit = current_x1 + v_x1 * min_t + 0.25 * current_dn2_dx1 * min_t * min_t
        x2_hit = current_x2 + v_x2 * min_t + 0.25 * current_dn2_dx2 * min_t * min_t
        x3_hit = current_x3 + v_x3 * min_t + 0.25 * current_dn2_dx3 * min_t * min_t
        phi_hit = current_phi + current_grad_phi * min_t
        theta_hit = current_theta + current_grad_theta * min_t

        # ----- Travel time + sigma -----
        # NOTE: uses n² at the cell's grid node. For better accuracy, we
        # should evaluate n² at the intersection point or interpolate it.
        total_travel_time += current_n_sq * min_t
        total_sigma += min_t

        # ----- Quadrilinear interpolation on the hit face -----
        fstatus, fclamps, fvals = _interpolate_on_face_3d_jit(
            hit_face_int,
            face_shift,
            ci,
            cj,
            cl,
            cm,
            cn,
            x1_hit,
            x2_hit,
            x3_hit,
            phi_hit,
            theta_hit,
            status_arr,
            u_arr,
            sigma_arr,
            y1_arr,
            y2_arr,
            y3_arr,
            phi_exit_arr,
            theta_exit_arr,
            x1_coords,
            x2_coords,
            x3_coords,
            phi_coords,
            theta_coords,
            dx1,
            dx2,
            dx3,
            dphi,
            dtheta,
            N_x1,
            N_x2,
            N_x3,
            N_phi,
            N_theta,
            _NODE_ACCEPTED,
        )

        total_clamps += fclamps

        if fstatus == _JFACE_SUCCESS:
            result = np.empty(7)
            result[0] = fvals[0] + total_travel_time
            result[1] = fvals[1] + total_sigma
            result[2] = fvals[2]
            result[3] = fvals[3]
            result[4] = fvals[4]
            result[5] = fvals[5]
            result[6] = fvals[6]
            return _CHAR_SUCCESS, cells_traversed, total_clamps, result
        if fstatus == _JFACE_NO_ACCEPTED:
            return _CHAR_NO_ACCEPTED, cells_traversed, total_clamps, dummy
        if fstatus == _JFACE_OOB:
            return _CHAR_OOB, cells_traversed, total_clamps, dummy

        # fstatus == PARTIAL: advance to next cell and continue.
        current_x1, current_x2, current_x3 = x1_hit, x2_hit, x3_hit

        # Advance every face index whose hit time is tied with min_t.
        tie_tol = 1e-12 + 1e-9 * min_t
        for k in range(n_cands):
            if cand_t[k] - min_t > tie_tol:
                continue
            face_k = cand_face[k]
            shift_k = cand_shift[k]
            if face_k == FACE_X1:
                ci = ci + shift_k
            elif face_k == FACE_X2:
                cj = cj + shift_k
            elif face_k == FACE_X3:
                cl = cl + shift_k
            elif face_k == FACE_PHI:
                cm = cm + shift_k
            else:  # FACE_THETA
                cn = (cn + shift_k) % N_theta

        # Pole reflection: a spatial/θ-face hit can drift φ past a pole.
        # Crossing a pole continues smoothly (φ↦−φ (or 2π−φ), θ↦θ+π).
        # A φ-face hit always lands on a node, so it never crosses a
        # pole. Hence the tied φ-face advance above never conflicts here.
        if phi_hit < 0.0:
            current_phi = -phi_hit
            current_theta = (theta_hit + math.pi) % (2.0 * math.pi)
            cm = 0
            cn = (cn + N_theta // 2) % N_theta
        elif phi_hit > math.pi:
            current_phi = 2.0 * math.pi - phi_hit
            current_theta = (theta_hit + math.pi) % (2.0 * math.pi)
            cm = N_phi - 1
            cn = (cn + N_theta // 2) % N_theta
        else:
            current_phi = phi_hit
            # Defensive normalization: keep current_theta in [0, 2π) so the
            # next iteration's delta-theta wrap logic stays correct.
            current_theta = theta_hit % (2.0 * math.pi)

        if ci < 0 or ci >= N_x1 or cj < 0 or cj >= N_x2 or cl < 0 or cl >= N_x3:
            return _CHAR_SPATIAL_OOB, cells_traversed, total_clamps, dummy
        if cm < 0 or cm >= N_phi:
            return _CHAR_PHI_OOB, cells_traversed, total_clamps, dummy

        # Refresh local slowness, gradients, direction, drifts for new cell.
        current_n = n_arr[ci, cj, cl]
        current_n_sq = current_n * current_n
        current_dn2_dx1 = dn2_arr[ci, cj, cl, 0]
        current_dn2_dx2 = dn2_arr[ci, cj, cl, 1]
        current_dn2_dx3 = dn2_arr[ci, cj, cl, 2]

        current_sin_phi = math.sin(current_phi)
        current_cos_phi = math.cos(current_phi)
        current_cos_theta = math.cos(current_theta)
        current_sin_theta = math.sin(current_theta)

        current_grad_phi = (
            current_cos_phi * current_cos_theta * current_dn2_dx1
            + current_cos_phi * current_sin_theta * current_dn2_dx2
            - current_sin_phi * current_dn2_dx3
        ) / (2.0 * current_n)
        current_grad_theta = (
            -current_sin_theta * current_dn2_dx1 + current_cos_theta * current_dn2_dx2
        ) / (2.0 * current_n * current_sin_phi)

    return _CHAR_EXHAUSTED, max_continuation_cells, total_clamps, dummy
