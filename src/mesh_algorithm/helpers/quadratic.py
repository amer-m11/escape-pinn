"""Dimension-agnostic parabolic trajectory intersection solver.

Used to find when a ray under the local parabolic approximation
``x(t) = position + velocity·t + acceleration·t²`` crosses a fixed
``boundary`` plane along one axis. Each spatial axis is solved
independently by the per-dim update routines.
"""

import math

from numba import njit


@njit(cache=True)
def solve_parabolic_intersection(
    position: float,
    velocity: float,
    acceleration: float,
    boundary: float,
) -> float:
    """Solve for the first positive intersection of a parabolic trajectory
    with a boundary plane.

    The trajectory is ``x(t) = position + velocity*t + acceleration*t²``.

    Returns
    -------
    float
        Smallest positive t at which the trajectory reaches ``boundary``.
        Returns ``math.inf`` when there is no motion, and ``math.nan``
        when there is no positive real intersection (trajectory curves away
        or is already past the face).
    """
    _EPS = 1e-12
    c = position - boundary

    if abs(acceleration) < _EPS:
        # Linear case: t = -c / velocity
        if abs(velocity) < _EPS:
            # No motion
            return math.inf
        t = -c / velocity
        if t <= _EPS:
            # Trajectory at or past the boundary face (e.g. a
            # corner hit where multiple faces were tied in the previous
            # iteration and only one cell index was advanced). Signal "no
            # positive intersection" so the caller drops this candidate.
            return math.nan
        return t

    # Quadratic case
    discriminant = velocity * velocity - 4.0 * acceleration * c
    if discriminant < 0.0:
        # Trajectory curves away from the boundary. No real intersection.
        return math.nan

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-velocity + sqrt_disc) / (2.0 * acceleration)
    t2 = (-velocity - sqrt_disc) / (2.0 * acceleration)

    # Return smallest positive root.
    best = math.inf
    if t1 >= _EPS:
        best = t1
    if t2 >= _EPS and t2 < best:
        best = t2
    if best == math.inf:
        # Parabolic trajectory doesn't reach the face in positive time
        # (analogous to the linear-case t≤0 branch above and the
        # discriminant<0 branch). Signal "no positive intersection" so the
        # caller drops this candidate.
        return math.nan
    return best
