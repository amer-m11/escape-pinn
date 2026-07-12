"""3D physical / 5D phase-space mesh grid.

Indices on the phase-space grid are 5-tuples ``(i, j, l, m, n)`` where:
    - ``i, j, l`` index the physical axes ``x1, x2, x3``
    - ``m`` indexes the polar angle ``φ ∈ [0, π]``
    - ``n`` indexes the azimuth ``θ ∈ [0, 2π)``

The unit ray direction is
    ``d = (sin φ cos θ, sin φ sin θ, cos φ)``.

``φ`` uses a **half-cell offset** ``φ_m = (m + 0.5) · π / N_φ`` so no grid
node lands exactly on a pole. The angular-drift formula carries a
``1/sin φ`` factor that we want to keep bounded.
"""

import itertools
import math
import warnings
from typing import Callable, Iterator

import numpy as np

from ..grid import Grid
from ..node_state import NodeState
from .update import CharacteristicResult3D, local_cell_characteristic_3d

Index3D = tuple[int, int, int, int, int]

_THETA_OFFSET_3D = 0.0
_PHI_OFFSET_3D = 0.5


class PhaseSpaceGrid3D(Grid):
    """5D phase-space grid ``(x1, x2, x3, φ, θ)`` for the 3D escape solver."""

    N_x1: int
    N_x2: int
    N_x3: int
    N_phi: int
    N_theta: int

    L_x1: float
    L_x2: float
    L_x3: float
    dx1: float
    dx2: float
    dx3: float
    dphi: float
    dtheta: float

    x1_coords: np.ndarray  # shape (N_x1,)
    x2_coords: np.ndarray  # shape (N_x2,)
    x3_coords: np.ndarray  # shape (N_x3,)
    phi_coords: np.ndarray  # shape (N_phi,), half-cell offset in [0, π]
    theta_coords: np.ndarray  # shape (N_theta,), in [0, 2π)

    slowness_fn: Callable[[float, float, float], float]
    slowness_gradient_fn: (
        Callable[[float, float, float], tuple[float, float, float]] | None
    )

    status: np.ndarray
    """node states: FAR, CONSIDERED, ACCEPTED — shape (N_x1, N_x2, N_x3, N_phi, N_theta)"""
    u: np.ndarray
    """û: escape time"""
    sigma: np.ndarray
    """σ̂: parametric distance to boundary along the characteristic"""
    y1: np.ndarray
    """ŷ1: exit position"""
    y2: np.ndarray
    """ŷ2: exit position"""
    y3: np.ndarray
    """ŷ3: exit position"""
    phi_exit: np.ndarray
    """φ̂: exit polar angle"""
    theta_exit: np.ndarray
    """θ̂: exit azimuth"""

    def __init__(
        self,
        grid_size: tuple[int, int, int, int, int],
        physical_size: tuple[float, float, float],
        slowness_fn: Callable[[float, float, float], float],
        slowness_gradient_fn: (
            Callable[[float, float, float], tuple[float, float, float]] | None
        ) = None,
    ):
        """
        Parameters
        ----------
        grid_size : tuple of int
            ``(N_x1, N_x2, N_x3, N_phi, N_theta)``: number of nodes per axis.
        physical_size : tuple of float
            ``(L_x1, L_x2, L_x3)``: physical domain extents.
        slowness_fn : callable
            ``n(x1, x2, x3) -> float``, the slowness field. Must be positive.
        slowness_gradient_fn : callable, optional
            ``grad(n²)(x1, x2, x3) -> (dn²/dx1, dn²/dx2, dn²/dx3)``.
            If ``None``, the gradient is computed numerically.
        """
        N_x1, N_x2, N_x3, N_phi, N_theta = grid_size
        if N_theta % 2 != 0:
            raise ValueError(
                f"N_theta={N_theta} must be even: the pole-reflected φ "
                "interpolation needs θ+π to be a grid node."
            )
        if N_phi < 4:
            amplification = 1.0 / math.sin(math.pi / (2 * N_phi))
            warnings.warn(
                f"N_phi={N_phi} below recommended minimum 4: "
                f"1/sin(π/(2·N_phi)) = {amplification:.2f} amplifies dθ/dσ "
                f"at the closest-to-pole grid line."
                "Use N_phi >= 8 for moderately curving slowness.",
                stacklevel=2,
            )
        self.N_x1, self.N_x2, self.N_x3 = N_x1, N_x2, N_x3
        self.N_phi, self.N_theta = N_phi, N_theta

        self.L_x1, self.L_x2, self.L_x3 = physical_size
        self.dx1 = self.L_x1 / (N_x1 - 1)
        self.dx2 = self.L_x2 / (N_x2 - 1)
        self.dx3 = self.L_x3 / (N_x3 - 1)
        self.dphi = math.pi / N_phi
        self.dtheta = (2 * math.pi) / N_theta

        self.x1_coords = np.linspace(0, self.L_x1, N_x1)
        self.x2_coords = np.linspace(0, self.L_x2, N_x2)
        self.x3_coords = np.linspace(0, self.L_x3, N_x3)
        # Half-cell offset on φ: dodge the exact poles where 1/sin φ blows up.
        self.phi_offset = _PHI_OFFSET_3D
        self.phi_coords = np.array(
            [(m + self.phi_offset) * self.dphi for m in range(N_phi)]
        )
        self.theta_offset = _THETA_OFFSET_3D
        self.theta_coords = np.array(
            [(n + self.theta_offset) * self.dtheta for n in range(N_theta)]
        )

        self.slowness_fn = slowness_fn
        self.slowness_gradient_fn = slowness_gradient_fn

        # Precompute slowness on the spatial grid.
        self._n = np.zeros((N_x1, N_x2, N_x3))
        self._n_grad = np.zeros((N_x1, N_x2, N_x3, 3))
        for i in range(N_x1):
            for j in range(N_x2):
                for l in range(N_x3):
                    self._n[i, j, l] = slowness_fn(
                        self.x1_coords[i], self.x2_coords[j], self.x3_coords[l]
                    )
                    if slowness_gradient_fn is not None:
                        self._n_grad[i, j, l] = slowness_gradient_fn(
                            self.x1_coords[i],
                            self.x2_coords[j],
                            self.x3_coords[l],
                        )
        self._n2 = self._n**2

        if slowness_gradient_fn is None:
            for i in range(N_x1):
                for j in range(N_x2):
                    for l in range(N_x3):
                        self._n_grad[i, j, l] = self._numerical_slowness_gradient(
                            i, j, l
                        )

        shape5 = (N_x1, N_x2, N_x3, N_phi, N_theta)
        self.status = np.full(shape5, NodeState.FAR, dtype=np.int32)
        self.u = np.full(shape5, np.inf, dtype=np.float64)
        self.sigma = np.full(shape5, np.inf, dtype=np.float64)
        self.y1 = np.full(shape5, np.nan, dtype=np.float64)
        self.y2 = np.full(shape5, np.nan, dtype=np.float64)
        self.y3 = np.full(shape5, np.nan, dtype=np.float64)
        self.phi_exit = np.full(shape5, np.nan, dtype=np.float64)
        self.theta_exit = np.full(shape5, np.nan, dtype=np.float64)

    # ===== Slowness access =====

    def slowness_at(self, i: int, j: int, l: int) -> float:
        return float(self._n[i, j, l])

    def slowness_gradient(self, i: int, j: int, l: int) -> tuple[float, float, float]:
        """Return ``(∂n²/∂x1, ∂n²/∂x2, ∂n²/∂x3)`` at grid node ``(i, j, l)``."""
        dn2 = self._n_grad[i, j, l]
        return float(dn2[0]), float(dn2[1]), float(dn2[2])

    # ===== Solver protocol =====

    def iter_indices(self) -> Iterator[Index3D]:
        """Iterate over every phase-space node index ``(i, j, l, m, n)``."""
        for i in range(self.N_x1):
            for j in range(self.N_x2):
                for l in range(self.N_x3):
                    for m in range(self.N_phi):
                        for n in range(self.N_theta):
                            yield i, j, l, m, n

    def is_boundary(self, idx: Index3D) -> bool:
        i, j, l, _, _ = idx
        return (
            i == 0
            or i == self.N_x1 - 1
            or j == 0
            or j == self.N_x2 - 1
            or l == 0
            or l == self.N_x3 - 1
        )

    def points_outward(self, idx: Index3D, eps: float = 1e-12) -> bool:
        """Check if the ray at ``idx`` points out of the cuboid through one
        of the six faces.

        ``eps`` is a numerical buffer so rays tangent to a face are not
        classified as outward.
        """
        i, j, l, m, n = idx
        phi = self.phi_coords[m]
        theta = self.theta_coords[n]
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        d1 = sin_phi * cos_theta  # x1 component
        d2 = sin_phi * sin_theta  # x2 component
        d3 = cos_phi  # x3 component

        if i == 0 and d1 <= -eps:
            return True
        if i == self.N_x1 - 1 and d1 >= eps:
            return True
        if j == 0 and d2 <= -eps:
            return True
        if j == self.N_x2 - 1 and d2 >= eps:
            return True
        if l == 0 and d3 <= -eps:
            return True
        if l == self.N_x3 - 1 and d3 >= eps:
            return True

        return False

    def accept_boundary_node(self, idx: Index3D) -> None:
        """Set the BC values at ``idx`` and mark ACCEPTED."""
        i, j, l, m, n = idx
        self.u[idx] = 0.0
        self.sigma[idx] = 0.0
        self.y1[idx] = self.x1_coords[i]
        self.y2[idx] = self.x2_coords[j]
        self.y3[idx] = self.x3_coords[l]
        self.phi_exit[idx] = self.phi_coords[m]
        self.theta_exit[idx] = self.theta_coords[n]
        self.status[idx] = NodeState.ACCEPTED

    def adjacent(self, idx: Index3D) -> list[Index3D]:
        """Immediate spatial (±1 in x1, x2, x3) + angular (±1 in φ, θ) neighbors.

        θ is periodic (wraps). φ is non-periodic (out-of-range is dropped).
        Spatial axes are clamped to the grid bounds.
        """
        i, j, l, m, n = idx
        result: list[Index3D] = []
        # Spatial
        for delta_i, delta_j, delta_l in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            ni, nj, nl = i + delta_i, j + delta_j, l + delta_l
            if 0 <= ni < self.N_x1 and 0 <= nj < self.N_x2 and 0 <= nl < self.N_x3:
                result.append((ni, nj, nl, m, n))
        # φ (clamped)
        for delta_m in (-1, 1):
            nm = m + delta_m
            if 0 <= nm < self.N_phi:
                result.append((i, j, l, nm, n))
        # θ (periodic)
        for delta_n in (-1, 1):
            nn = (n + delta_n) % self.N_theta
            result.append((i, j, l, m, nn))
        return result

    def octant_neighbors(self, idx: Index3D) -> list[Index3D]:
        """Return the up-to-31 octant neighbors of ``idx``.

        Enumerates the 2⁵ axis-shift combinations. Drops
        out-of-bounds spatial and φ steps and wraps θ steps.
        """
        i, j, l, m, n = idx
        phi = self.phi_coords[m]
        theta = self.theta_coords[n]
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        n0 = self.slowness_at(i, j, l)
        dn2_dx1, dn2_dx2, dn2_dx3 = self.slowness_gradient(i, j, l)

        # Hamilton angular drift. Half-cell φ offset keeps
        # sin_phi bounded away from zero so dθ/dσ stays finite.
        grad_phi = (
            cos_phi * cos_theta * dn2_dx1
            + cos_phi * sin_theta * dn2_dx2
            - sin_phi * dn2_dx3
        ) / (2 * n0)
        grad_theta = (-sin_theta * dn2_dx1 + cos_theta * dn2_dx2) / (2 * n0 * sin_phi)

        # The spatial direction signs come from the ray's d = (sin φ cos θ,
        # sin φ sin θ, cos φ).  sin φ ≥ 0 (half-cell offset), so:
        #   sign(d1) = sign(cos θ),  sign(d2) = sign(sin θ),
        #   sign(d3) = sign(cos φ).
        # Reverse sign because the march expands from the boundary inward.
        di_direction = 1 if cos_theta < 0 else -1
        dj_direction = 1 if sin_theta < 0 else -1
        dl_direction = 1 if cos_phi < 0 else -1
        dm_direction = 1 if grad_phi < 0 else -1
        dn_direction = 1 if grad_theta < 0 else -1

        neighbors: list[Index3D] = []
        for shifts in itertools.product(
            (0, di_direction),
            (0, dj_direction),
            (0, dl_direction),
            (0, dm_direction),
            (0, dn_direction),
        ):
            d_i, d_j, d_l, d_m, d_n = shifts
            if d_i == 0 and d_j == 0 and d_l == 0 and d_m == 0 and d_n == 0:
                continue
            ni = i + d_i
            nj = j + d_j
            nl = l + d_l
            nm = m + d_m
            nn = (n + d_n) % self.N_theta
            if not (
                0 <= ni < self.N_x1 and 0 <= nj < self.N_x2 and 0 <= nl < self.N_x3
            ):
                continue
            if not 0 <= nm < self.N_phi:
                continue
            neighbors.append((ni, nj, nl, nm, nn))
        return neighbors

    def compute_characteristic(self, idx: Index3D) -> CharacteristicResult3D:
        """Run the parabolic-ray characteristic and return a tentative update."""
        return local_cell_characteristic_3d(self, idx)

    def commit_characteristic(
        self, idx: Index3D, result: CharacteristicResult3D
    ) -> None:
        """Write all escape quantities from ``result`` and mark CONSIDERED."""
        self.u[idx] = result.u
        self.sigma[idx] = result.sigma
        self.y1[idx] = result.y1
        self.y2[idx] = result.y2
        self.y3[idx] = result.y3
        self.phi_exit[idx] = result.phi_exit
        self.theta_exit[idx] = result.theta_exit
        self.status[idx] = NodeState.CONSIDERED

    # ===== numerical slowness gradient =====

    def _numerical_slowness_gradient(
        self, i: int, j: int, l: int
    ) -> tuple[float, float, float]:
        """Central differences of n² at grid node ``(i, j, l)``."""
        n2 = self._n2

        if i == 0:
            dn2_dx1 = (n2[1, j, l] - n2[0, j, l]) / self.dx1
        elif i == self.N_x1 - 1:
            dn2_dx1 = (n2[-1, j, l] - n2[-2, j, l]) / self.dx1
        else:
            dn2_dx1 = (n2[i + 1, j, l] - n2[i - 1, j, l]) / (2 * self.dx1)

        if j == 0:
            dn2_dx2 = (n2[i, 1, l] - n2[i, 0, l]) / self.dx2
        elif j == self.N_x2 - 1:
            dn2_dx2 = (n2[i, -1, l] - n2[i, -2, l]) / self.dx2
        else:
            dn2_dx2 = (n2[i, j + 1, l] - n2[i, j - 1, l]) / (2 * self.dx2)

        if l == 0:
            dn2_dx3 = (n2[i, j, 1] - n2[i, j, 0]) / self.dx3
        elif l == self.N_x3 - 1:
            dn2_dx3 = (n2[i, j, -1] - n2[i, j, -2]) / self.dx3
        else:
            dn2_dx3 = (n2[i, j, l + 1] - n2[i, j, l - 1]) / (2 * self.dx3)

        return float(dn2_dx1), float(dn2_dx2), float(dn2_dx3)
