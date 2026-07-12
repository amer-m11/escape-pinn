"""Map a continuous intersection coordinate to bracketing grid indices."""

import math

from numba import njit


@njit(cache=True)
def compute_discrete_indices(
    intersection: float, d: float, offset: float = 0.0
) -> tuple:
    """Map a continuous coordinate to bracketing grid indices and weight.

    When the intersection lands exactly on a grid line (within 1e-8),
    both returned indices coincide.

    Parameters
    ----------
    intersection : float
        The continuous coordinate along one axis.
    d : float
        Grid spacing along the same axis.
    offset : float, default 0
        Fractional shift of the grid relative to ``0``. With ``offset = 0``
        the grid lives at ``{0, d, 2d, ...}`` (spatial and θ axes). With
        ``offset = 0.5`` it lives at ``{0.5d, 1.5d, ...}``. This is used for
        the 3D ``φ`` axis to avoid the polar singularity.

    Returns
    -------
    tuple[int, int, float]
        Lower index, upper index, and interpolation weight (0 = lower, 1 = upper).
    """
    _EPS = 1e-8
    fractional_index = intersection / d - offset
    lower_index = int(math.floor(fractional_index))

    if abs(fractional_index - (lower_index + 1)) <= _EPS:
        # Hit exactly on the upper grid line
        lower_index = lower_index + 1
        upper_index = lower_index
    elif abs(fractional_index - lower_index) <= _EPS:
        # Hit exactly on the lower grid line
        upper_index = lower_index
    else:
        upper_index = lower_index + 1

    weight = fractional_index - lower_index
    return lower_index, upper_index, weight
