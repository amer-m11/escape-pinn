"""Cell-face interpolation for the 3D escape-equation mesh solver.

Each cell in the 5D phase-space grid has five face types (``x1``, ``x2``,
``x3``, ``phi``, ``theta``) and each face is a 4-axis manifold spanning
16 corner nodes. Interpolation is quadrilinear.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit

from ..helpers.discrete_indices import compute_discrete_indices
from ..helpers.n_linear import _n_linear_jit_3d
from ..node_state import NodeState

# Paper step 3b: no corners accepted -> don't update (return sentinel).
_NO_ACCEPTED = (np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan)
_OUT_OF_BOUND_RESULT = (np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan)

# Integer face-type constants used by the JIT path.
FACE_X1 = 0
FACE_X2 = 1
FACE_X3 = 2
FACE_PHI = 3
FACE_THETA = 4

# Face-interpolation status codes returned by the JIT function.
_JFACE_SUCCESS = 0
_JFACE_NO_ACCEPTED = 1
_JFACE_PARTIAL = 2
_JFACE_OOB = 3

# NodeState.ACCEPTED as a plain int for use inside @njit functions.
_NODE_ACCEPTED = int(NodeState.ACCEPTED)  # = 2


@njit(cache=True)
def _clamp_axis_indices(lo, hi, w, N):
    """Clamp an out-of-range bracket to the boundary.

    The parabolic ray step ``¼·∂n²/∂x_a·s²`` in
    :mod:`mesh_algorithm.three_d.update` can push the hit point coordinate
    past ``[0, (N-1)·d]`` when the medium has a strong gradient. Rather
    than failing the whole interpolation we collapse any out-of-range
    bracket to a one-corner sample at the nearest boundary node. Otherwise,
    the node would be left unresolved with an OOB error.

    Note that clamping could hide errors silently.
    For that reason we record the number of clamps applied. When nodes away
    from the boundary are affected, it indicates an error.

    Returns ``(lo, hi, w, fired)`` where ``fired`` is 1 when a clamp was
    applied and 0 otherwise.
    """
    if lo < 0:
        return 0, 0, 0.0, 1
    if hi >= N:
        return N - 1, N - 1, 0.0, 1
    return lo, hi, w, 0


@njit(cache=True)
def _assemble_corners_jit(
    face_int,
    face_pos,
    a0_lo,
    a0_hi,
    a1_lo,
    a1_hi,
    a2_lo,
    a2_hi,
    a3_lo,
    a3_hi,
    N_phi,
    N_theta,
):
    """Build the ``(16, 5)`` corner index array for a given face type.

    Mirrors the logic of :func:`_assemble_corners` but operates purely on
    integers so it can be called from other ``@njit`` functions.

    The varying-axis-to-slot mapping:

    * face 0 (x1): fixed at slot 0, varying slots = (1, 2, 3, 4)
    * face 1 (x2): fixed at slot 1, varying slots = (0, 2, 3, 4)
    * face 2 (x3): fixed at slot 2, varying slots = (0, 1, 3, 4)
    * face 3 (phi): fixed at slot 3, varying slots = (0, 1, 2, 4)
    * face 4 (theta): fixed at slot 4, varying slots = (0, 1, 2, 3)

    ``a0_lo/hi … a3_lo/hi`` are the low/high index pairs for varying axes
    0-3 (in the same bit order as :func:`n_linear_interpolation`).

    NOTE: When φ (slot 3) is outside ``[0, N_phi)`` we reflect it across
    the pole: ``m < 0 → (-1-m, θ+π)`` and ``m ≥ N_phi → (2·N_phi-1-m, θ+π)``.
    The θ (slot 4) flip is ``+N_theta/2`` (exact only for even ``N_theta``,
    which the grid constructor enforces.
    """
    corners = np.empty((16, 5), dtype=np.int64)

    if face_int == 0:
        s0, s1, s2, s3 = 1, 2, 3, 4
    elif face_int == 1:
        s0, s1, s2, s3 = 0, 2, 3, 4
    elif face_int == 2:
        s0, s1, s2, s3 = 0, 1, 3, 4
    elif face_int == 3:
        s0, s1, s2, s3 = 0, 1, 2, 4
    else:  # face_int == 4 (theta)
        s0, s1, s2, s3 = 0, 1, 2, 3

    for c_idx in range(16):
        corners[c_idx, face_int] = face_pos
        corners[c_idx, s0] = a0_hi if (c_idx & 1) else a0_lo
        corners[c_idx, s1] = a1_hi if ((c_idx >> 1) & 1) else a1_lo
        corners[c_idx, s2] = a2_hi if ((c_idx >> 2) & 1) else a2_lo
        corners[c_idx, s3] = a3_hi if ((c_idx >> 3) & 1) else a3_lo

        # Pole reflection (resolve φ to a real node).
        # φ is reflected in respect to the pole band center and θ is flipped
        # by half the period, which maps to the opposite node across the pole.
        m = corners[c_idx, 3]
        if m < 0 or m >= N_phi:
            if m < 0:
                m = -1 - m
            else:
                m = 2 * N_phi - 1 - m
            corners[c_idx, 4] = (corners[c_idx, 4] + N_theta // 2) % N_theta

            # Defensive: a φ excursion past a whole pole band.
            if m < 0:
                m = 0
            elif m >= N_phi:
                m = N_phi - 1

            corners[c_idx, 3] = m

    return corners


@njit(cache=True)
def _interpolate_on_face_3d_jit(
    hit_face_int,
    face_shift,
    i,
    j,
    l,
    m,
    n,
    x1_hit,
    x2_hit,
    x3_hit,
    phi_hit,
    theta_hit,
    status_arr,
    u,
    sigma,
    y1,
    y2,
    y3,
    phi_exit,
    theta_exit,
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
    accepted_int,
):
    """JIT-compiled quadrilinear face interpolation.

    Returns ``(status_int, n_clamps, float64[7])`` where ``status_int`` is one of:

    * ``_JFACE_SUCCESS`` (0) — all 16 corners accepted; values valid.
    * ``_JFACE_NO_ACCEPTED`` (1) — no corners accepted.
    * ``_JFACE_PARTIAL`` (2) — some corners accepted; values meaningless.
    * ``_JFACE_OOB`` (3) — face index out of bounds.

    ``n_clamps`` counts how many brackets `_clamp_axis_indices` pulled back to
    a boundary node.
    """
    dummy = np.zeros(7)
    n_clamps = 0

    if hit_face_int == 0:  # x1 face: fixed axis = i
        face_pos = i + face_shift
        if face_pos < 0 or face_pos >= N_x1:
            return _JFACE_OOB, n_clamps, dummy
        j_lo, j_hi, w_j = compute_discrete_indices(x2_hit, dx2, 0.0)
        j_lo, j_hi, w_j, _cj = _clamp_axis_indices(j_lo, j_hi, w_j, N_x2)
        n_clamps += _cj
        l_lo, l_hi, w_l = compute_discrete_indices(x3_hit, dx3, 0.0)
        l_lo, l_hi, w_l, _cl = _clamp_axis_indices(l_lo, l_hi, w_l, N_x3)
        n_clamps += _cl
        # φ not clamped here as it is resolved later by reflection
        m_lo, m_hi, w_m = compute_discrete_indices(phi_hit, dphi, 0.5)
        n_lo, n_hi, w_n = compute_discrete_indices(theta_hit, dtheta, 0.0)
        if j_lo < 0 or j_hi >= N_x2:
            return _JFACE_OOB, n_clamps, dummy
        if l_lo < 0 or l_hi >= N_x3:
            return _JFACE_OOB, n_clamps, dummy
        a0_lo, a0_hi = j_lo, j_hi
        a1_lo, a1_hi = l_lo, l_hi
        a2_lo, a2_hi = m_lo, m_hi
        a3_lo, a3_hi = n_lo % N_theta, n_hi % N_theta
        weights = np.array([w_j, w_l, w_m, w_n])

    elif hit_face_int == 1:  # x2 face: fixed axis = j
        face_pos = j + face_shift
        if face_pos < 0 or face_pos >= N_x2:
            return _JFACE_OOB, n_clamps, dummy
        i_lo, i_hi, w_i = compute_discrete_indices(x1_hit, dx1, 0.0)
        i_lo, i_hi, w_i, _ci = _clamp_axis_indices(i_lo, i_hi, w_i, N_x1)
        n_clamps += _ci
        l_lo, l_hi, w_l = compute_discrete_indices(x3_hit, dx3, 0.0)
        l_lo, l_hi, w_l, _cl = _clamp_axis_indices(l_lo, l_hi, w_l, N_x3)
        n_clamps += _cl
        # φ not clamped here as it is resolved later by reflection
        m_lo, m_hi, w_m = compute_discrete_indices(phi_hit, dphi, 0.5)
        n_lo, n_hi, w_n = compute_discrete_indices(theta_hit, dtheta, 0.0)
        if i_lo < 0 or i_hi >= N_x1:
            return _JFACE_OOB, n_clamps, dummy
        if l_lo < 0 or l_hi >= N_x3:
            return _JFACE_OOB, n_clamps, dummy
        a0_lo, a0_hi = i_lo, i_hi
        a1_lo, a1_hi = l_lo, l_hi
        a2_lo, a2_hi = m_lo, m_hi
        a3_lo, a3_hi = n_lo % N_theta, n_hi % N_theta
        weights = np.array([w_i, w_l, w_m, w_n])

    elif hit_face_int == 2:  # x3 face: fixed axis = l
        face_pos = l + face_shift
        if face_pos < 0 or face_pos >= N_x3:
            return _JFACE_OOB, n_clamps, dummy
        i_lo, i_hi, w_i = compute_discrete_indices(x1_hit, dx1, 0.0)
        i_lo, i_hi, w_i, _ci = _clamp_axis_indices(i_lo, i_hi, w_i, N_x1)
        n_clamps += _ci
        j_lo, j_hi, w_j = compute_discrete_indices(x2_hit, dx2, 0.0)
        j_lo, j_hi, w_j, _cj = _clamp_axis_indices(j_lo, j_hi, w_j, N_x2)
        n_clamps += _cj
        m_lo, m_hi, w_m = compute_discrete_indices(phi_hit, dphi, 0.5)
        n_lo, n_hi, w_n = compute_discrete_indices(theta_hit, dtheta, 0.0)
        if i_lo < 0 or i_hi >= N_x1:
            return _JFACE_OOB, n_clamps, dummy
        if j_lo < 0 or j_hi >= N_x2:
            return _JFACE_OOB, n_clamps, dummy
        a0_lo, a0_hi = i_lo, i_hi
        a1_lo, a1_hi = j_lo, j_hi
        a2_lo, a2_hi = m_lo, m_hi
        a3_lo, a3_hi = n_lo % N_theta, n_hi % N_theta
        weights = np.array([w_i, w_j, w_m, w_n])

    elif hit_face_int == 3:  # phi face: fixed axis = m
        face_pos = m + face_shift
        if face_pos < 0 or face_pos >= N_phi:
            return _JFACE_OOB, n_clamps, dummy
        i_lo, i_hi, w_i = compute_discrete_indices(x1_hit, dx1, 0.0)
        i_lo, i_hi, w_i, _ci = _clamp_axis_indices(i_lo, i_hi, w_i, N_x1)
        n_clamps += _ci
        j_lo, j_hi, w_j = compute_discrete_indices(x2_hit, dx2, 0.0)
        j_lo, j_hi, w_j, _cj = _clamp_axis_indices(j_lo, j_hi, w_j, N_x2)
        n_clamps += _cj
        l_lo, l_hi, w_l = compute_discrete_indices(x3_hit, dx3, 0.0)
        l_lo, l_hi, w_l, _cl = _clamp_axis_indices(l_lo, l_hi, w_l, N_x3)
        n_clamps += _cl
        n_lo, n_hi, w_n = compute_discrete_indices(theta_hit, dtheta, 0.0)
        if i_lo < 0 or i_hi >= N_x1:
            return _JFACE_OOB, n_clamps, dummy
        if j_lo < 0 or j_hi >= N_x2:
            return _JFACE_OOB, n_clamps, dummy
        if l_lo < 0 or l_hi >= N_x3:
            return _JFACE_OOB, n_clamps, dummy
        a0_lo, a0_hi = i_lo, i_hi
        a1_lo, a1_hi = j_lo, j_hi
        a2_lo, a2_hi = l_lo, l_hi
        a3_lo, a3_hi = n_lo % N_theta, n_hi % N_theta
        weights = np.array([w_i, w_j, w_l, w_n])

    else:  # hit_face_int == 4, theta face: fixed axis = n (periodic)
        face_pos = (n + face_shift) % N_theta
        i_lo, i_hi, w_i = compute_discrete_indices(x1_hit, dx1, 0.0)
        i_lo, i_hi, w_i, _ci = _clamp_axis_indices(i_lo, i_hi, w_i, N_x1)
        n_clamps += _ci
        j_lo, j_hi, w_j = compute_discrete_indices(x2_hit, dx2, 0.0)
        j_lo, j_hi, w_j, _cj = _clamp_axis_indices(j_lo, j_hi, w_j, N_x2)
        n_clamps += _cj
        l_lo, l_hi, w_l = compute_discrete_indices(x3_hit, dx3, 0.0)
        l_lo, l_hi, w_l, _cl = _clamp_axis_indices(l_lo, l_hi, w_l, N_x3)
        n_clamps += _cl
        m_lo, m_hi, w_m = compute_discrete_indices(phi_hit, dphi, 0.5)
        if i_lo < 0 or i_hi >= N_x1:
            return _JFACE_OOB, n_clamps, dummy
        if j_lo < 0 or j_hi >= N_x2:
            return _JFACE_OOB, n_clamps, dummy
        if l_lo < 0 or l_hi >= N_x3:
            return _JFACE_OOB, n_clamps, dummy
        a0_lo, a0_hi = i_lo, i_hi
        a1_lo, a1_hi = j_lo, j_hi
        a2_lo, a2_hi = l_lo, l_hi
        a3_lo, a3_hi = m_lo, m_hi
        weights = np.array([w_i, w_j, w_l, w_m])

    corners = _assemble_corners_jit(
        hit_face_int,
        face_pos,
        a0_lo,
        a0_hi,
        a1_lo,
        a1_hi,
        a2_lo,
        a2_hi,
        a3_lo,
        a3_hi,
        N_phi,
        N_theta,
    )

    n_accepted = 0
    for c_idx in range(16):
        if (
            status_arr[
                corners[c_idx, 0],
                corners[c_idx, 1],
                corners[c_idx, 2],
                corners[c_idx, 3],
                corners[c_idx, 4],
            ]
            == accepted_int
        ):
            n_accepted += 1

    if n_accepted == 0:
        return _JFACE_NO_ACCEPTED, n_clamps, dummy
    if n_accepted < 16:
        return _JFACE_PARTIAL, n_clamps, dummy

    values = _n_linear_jit_3d(
        u, sigma, y1, y2, y3, phi_exit, theta_exit, corners, weights
    )
    return _JFACE_SUCCESS, n_clamps, values


@dataclass(frozen=True, slots=True)
class FaceInterpolationResult3D:
    """Outcome of interpolating the escape fields on one 3D face."""

    status: Literal["success", "partial", "no_accepted", "out_of_bounds"]
    values: tuple[float, float, float, float, float, float, float] | None = None


def _assemble_corners(varying_axes_indices, fixed_axis_index, fixed_axis_position):
    """Build the 16 corner tuples for a 4-axis cell face.

    Parameters
    ----------
    varying_axes_indices : sequence of 4 ``(lo, hi)`` index pairs
        Index bracket for each varying axis, in canonical bit order
        (axis x corresponds to bit x of the corner index).
    fixed_axis_index : int
        Position in the 5-tuple ``(i, j, l, m, n)`` of the fixed axis.
    fixed_axis_position : int
        Value at the fixed axis.
    """
    corners = []
    for c_idx in range(16):
        slots = [None, None, None, None, None]
        slots[fixed_axis_index] = fixed_axis_position
        for axis_idx, (lo, hi) in enumerate(varying_axes_indices):
            bit = (c_idx >> axis_idx) & 1
            other_idx = _OTHER_AXIS_SLOTS[fixed_axis_index][axis_idx]
            slots[other_idx] = hi if bit else lo
        corners.append(tuple(slots))
    return corners


# For each "fixed axis" (the face type), list the indices of the 5-tuple
# (i, j, l, m, n) that the four varying axes occupy in canonical bit order.
_OTHER_AXIS_SLOTS = {
    # x1 face fixed at slot 0  -> vary (x2=1, x3=2, phi=3, theta=4)
    0: (1, 2, 3, 4),
    # x2 face fixed at slot 1  -> vary (x1=0, x3=2, phi=3, theta=4)
    1: (0, 2, 3, 4),
    # x3 face fixed at slot 2  -> vary (x1=0, x2=1, phi=3, theta=4)
    2: (0, 1, 3, 4),
    # phi face fixed at slot 3 -> vary (x1=0, x2=1, x3=2, theta=4)
    3: (0, 1, 2, 4),
    # theta face fixed at slot 4 -> vary (x1=0, x2=1, x3=2, phi=3)
    4: (0, 1, 2, 3),
}


def interpolate_on_face_3d(
    grid,
    hit_face: Literal["x1", "x2", "x3", "phi", "theta"],
    face_shift: int,
    i: int,
    j: int,
    l: int,
    m: int,
    n: int,
    x1_intersection: float,
    x2_intersection: float,
    x3_intersection: float,
    phi_intersection: float,
    theta_intersection: float,
):
    """Interpolate the seven escape quantities on the hit cell face.

    Delegates to :func:`_interpolate_on_face_3d_jit` for the numeric work
    and converts the integer status code back to a
    :class:`FaceInterpolationResult3D`.

    Returns
    -------
        FaceInterpolationResult3D
            - ``success`` with values when all 16 corners are ACCEPTED.
            - ``no_accepted`` when no corners are ACCEPTED.
            - ``out_of_bounds`` when the interpolated face indices are invalid.
            - ``partial`` when some but not all corners are ACCEPTED.
    """
    _FACE_INT = {
        "x1": FACE_X1,
        "x2": FACE_X2,
        "x3": FACE_X3,
        "phi": FACE_PHI,
        "theta": FACE_THETA,
    }
    hit_face_int = _FACE_INT[hit_face]

    jstatus, _n_clamps, values = _interpolate_on_face_3d_jit(
        hit_face_int,
        face_shift,
        i,
        j,
        l,
        m,
        n,
        x1_intersection,
        x2_intersection,
        x3_intersection,
        phi_intersection,
        theta_intersection,
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
        _NODE_ACCEPTED,
    )

    if jstatus == _JFACE_SUCCESS:
        return FaceInterpolationResult3D(
            status="success",
            values=(
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                float(values[6]),
            ),
        )
    if jstatus == _JFACE_NO_ACCEPTED:
        return FaceInterpolationResult3D(status="no_accepted")
    if jstatus == _JFACE_PARTIAL:
        return FaceInterpolationResult3D(status="partial")
    return FaceInterpolationResult3D(status="out_of_bounds")
