"""N-linear interpolation across arbitrary axes.

The caller supplies the corner indices in canonical bit-order:

    corner[c] maps to (bit_0, bit_1, ..., bit_{k-1})
    where bit_a = 1 means "high end of axis a", bit_a = 0 means "low end".

So for k=2: ``corners = [c_lo_lo, c_hi_lo, c_lo_hi, c_hi_hi]``.
"""

import numpy as np
from numba import njit

_EPSILON = 1e-7


def n_linear_interpolation(field_arrays, corners, weights):
    """Multilinear interpolation across an arbitrary number of axes.

    Parameters
    ----------
    field_arrays : sequence of ndarray
        Field arrays containing the values to interpolate. They must
        share the same indexing shape (e.g. ``grid.u``, ``grid.sigma``, …).
    corners : sequence of index tuples
        Used to access the field arrays at the corners of the interpolation
        hypercube. There must be ``2**k`` index tuples (k = len(weights)).
        See module docstring.
    weights : sequence of float
        ``k`` interpolation weights, each in ``[0, 1]``. Weight x moves
        from the low end (0) to the high end (1) of axis x.

    Returns
    -------
    tuple of float
        Interpolated value of each field using the provided weights. They
        are returned in the same order as ``field_arrays``.
    """
    n_axes = len(weights)
    n_corners = 1 << n_axes  # 2**n_axes
    if len(corners) != n_corners:
        raise ValueError(
            f"Expected {n_corners} corners for {n_axes} axes. Got {len(corners)}."
        )
    for w in weights:
        if w < -_EPSILON or w > 1 + _EPSILON:
            raise ValueError(f"Weights must be in [0,1]. Got {w}.")

    weight_products = [0.0] * n_corners
    for c_idx in range(n_corners):
        prod = 1.0
        for axis_idx, w in enumerate(weights):
            bit = (c_idx >> axis_idx) & 1
            prod *= w if bit else (1.0 - w)
        weight_products[c_idx] = prod

    return tuple(
        float(sum(field[corners[c]] * weight_products[c] for c in range(n_corners)))
        for field in field_arrays
    )


@njit(cache=True)
def _n_linear_jit_3d(u, sigma, y1, y2, y3, phi_exit, theta_exit, corners, weights):
    """Quadrilinear interpolation over 7 escape fields for the 3D path.

    JIT-compiled specialization of :func:`n_linear_interpolation` for the
    fixed 3D case: 7 fields, 16 corners, 4 axes.

    Parameters
    ----------
    u, sigma, y1, y2, y3, phi_exit, theta_exit : 5-D ``float64`` arrays
        Escape-field arrays from the phase-space grid, all sharing shape
        ``(N_x1, N_x2, N_x3, N_phi, N_theta)``.
    corners : ``int64[:, :]`` of shape ``(16, 5)``
        Corner indices in canonical bit order (bit ``x`` of corner index
        selects high/low end of axis ``x``).
    weights : ``float64[:]`` of shape ``(4,)``
        Interpolation weights for the four varying axes.

    Returns
    -------
    ``float64[:]`` of shape ``(7,)``
        Interpolated values in field order:
        ``[û, σ̂, ŷ1, ŷ2, ŷ3, φ̂_exit, θ̂_exit]``.
    """
    result = np.zeros(7)
    for c_idx in range(16):
        prod = 1.0
        for axis_idx in range(4):
            bit = (c_idx >> axis_idx) & 1
            w = weights[axis_idx]
            if bit:
                prod *= w
            else:
                prod *= 1.0 - w
        ci0 = corners[c_idx, 0]
        ci1 = corners[c_idx, 1]
        ci2 = corners[c_idx, 2]
        ci3 = corners[c_idx, 3]
        ci4 = corners[c_idx, 4]
        result[0] += u[ci0, ci1, ci2, ci3, ci4] * prod
        result[1] += sigma[ci0, ci1, ci2, ci3, ci4] * prod
        result[2] += y1[ci0, ci1, ci2, ci3, ci4] * prod
        result[3] += y2[ci0, ci1, ci2, ci3, ci4] * prod
        result[4] += y3[ci0, ci1, ci2, ci3, ci4] * prod
        result[5] += phi_exit[ci0, ci1, ci2, ci3, ci4] * prod
        result[6] += theta_exit[ci0, ci1, ci2, ci3, ci4] * prod
    return result
