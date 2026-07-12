"""2D physical / 3D phase-space mesh grid."""

import math
from typing import Callable, Iterator

import numpy as np

from ..grid import Grid
from ..node_state import NodeState
from .update import CharacteristicResult2D, local_cell_characteristic_2d

THETA_OFFSET_2D = 0.5


class PhaseSpaceGrid2D(Grid):
    """3D phase-space grid ``(x1, x2, θ)`` for the 2D marching algorithm solver."""

    N_x1: int
    N_x2: int
    N_theta: int

    L_x1: float
    L_x2: float
    dx1: float
    dx2: float
    dtheta: float

    x1_coords: np.ndarray  # shape (N_x1,), values in [0, L_x1]
    x2_coords: np.ndarray  # shape (N_x2,), values in [0, L_x2]
    theta_coords: np.ndarray  # shape (N_theta,), values in [0, 2π)

    slowness_fn: Callable[[float, float], float]
    slowness_gradient_fn: Callable[[float, float], tuple[float, float]] | None
    """∇(n²)(x1, x2)"""

    status: np.ndarray
    """node states: FAR, CONSIDERED, ACCEPTED"""
    u: np.ndarray
    """û: escape time starting from node (i, j, k)"""
    sigma: np.ndarray
    """σ̂: parametric distance along the characteristic to the exit point."""
    y1: np.ndarray
    """ŷ1: exit position in x1 for ray starting from node (i, j, k)"""
    y2: np.ndarray
    """ŷ2: exit position in x2 for ray starting from node (i, j, k)"""
    theta_exit: np.ndarray
    """θ̂: exit angle for ray starting from node (i, j, k)"""

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        physical_size: tuple[float, float],
        slowness_fn: Callable[[float, float], float],
        slowness_gradient_fn: (
            Callable[[float, float], tuple[float, float]] | None
        ) = None,
    ):
        """
        Parameters
        ----------
        grid_size : tuple of int
            ``(N_x1, N_x2, N_theta)`` — number of grid points in x1, x2,
            and θ dimensions.
        physical_size : tuple of float
            ``(L_x1, L_x2)`` — physical domain extents.
        slowness_fn : callable
            ``n(x1, x2) -> float``, the slowness field. Must be positive.
        slowness_gradient_fn : callable, optional
            ``grad(n^2)(x1, x2) -> (dn2_dx1, dn2_dx2)``. If ``None`` the
            gradient is computed numerically via central differences on
            the precomputed n² grid.
        """
        N_x1, N_x2, N_theta = grid_size
        self.N_x1, self.N_x2, self.N_theta = N_x1, N_x2, N_theta

        self.L_x1, self.L_x2 = physical_size
        self.dx1 = 1.0 * self.L_x1 / (N_x1 - 1)
        self.dx2 = 1.0 * self.L_x2 / (N_x2 - 1)
        self.dtheta = (2 * math.pi) / N_theta

        self.theta_offset = THETA_OFFSET_2D
        self.x1_coords = np.linspace(0, self.L_x1, N_x1)
        self.x2_coords = np.linspace(0, self.L_x2, N_x2)
        self.theta_coords = np.array(
            [(k + THETA_OFFSET_2D) * self.dtheta for k in range(N_theta)]
        )
        """Offset by half a cell: theta_coords[k] = (k + THETA_OFFSET_2D) · 2π/N_theta."""

        self.slowness_fn = slowness_fn

        # Precompute slowness on the spatial grid
        self._n = np.zeros((N_x1, N_x2))
        self._n_grad = np.zeros((N_x1, N_x2, 2))  # (dn2_dx1, dn2_dx2)
        for i in range(N_x1):
            for j in range(N_x2):
                self._n[i, j] = slowness_fn(self.x1_coords[i], self.x2_coords[j])
                if slowness_gradient_fn is not None:
                    self._n_grad[i, j] = slowness_gradient_fn(
                        self.x1_coords[i], self.x2_coords[j]
                    )

        self._n2 = self._n**2

        if slowness_gradient_fn is None:
            for i in range(N_x1):
                for j in range(N_x2):
                    self._n_grad[i, j] = self._numerical_slowness_gradient(i, j)

        self.status = np.full((N_x1, N_x2, N_theta), NodeState.FAR, dtype=np.int32)
        self.u = np.full((N_x1, N_x2, N_theta), np.inf, dtype=np.float64)
        self.y1 = np.full((N_x1, N_x2, N_theta), np.nan, dtype=np.float64)
        self.y2 = np.full((N_x1, N_x2, N_theta), np.nan, dtype=np.float64)
        self.sigma = np.full((N_x1, N_x2, N_theta), np.inf, dtype=np.float64)
        self.theta_exit = np.full((N_x1, N_x2, N_theta), np.nan, dtype=np.float64)

    # ===== Slowness access =====

    def slowness_at(self, i: int, j: int) -> float:
        return float(self._n[i, j])

    def slowness_gradient(self, i: int, j: int) -> tuple[float, float]:
        """Return ``(∂n²/∂x1, ∂n²/∂x2)`` at grid node ``(i, j)``."""
        dn2_dx1, dn2_dx2 = self._n_grad[i, j]
        return dn2_dx1, dn2_dx2

    # ===== Solver protocol =====

    def iter_indices(self) -> Iterator[tuple[int, int, int]]:
        """Iterate over every phase-space node index ``(i, j, k)``."""
        for i in range(self.N_x1):
            for j in range(self.N_x2):
                for k in range(self.N_theta):
                    yield i, j, k

    def is_boundary(self, idx: tuple[int, int, int]) -> bool:
        i, j, _ = idx
        return i == 0 or i == self.N_x1 - 1 or j == 0 or j == self.N_x2 - 1

    def points_outward(self, idx: tuple[int, int, int], eps: float = 1e-12) -> bool:
        """Check if ray from ``(i, j)`` in direction θ_k points outward.

        ``eps`` is a numerical buffer so rays tangent to the boundary
        are not classified as outward-pointing."""
        i, j, k = idx
        theta = self.theta_coords[k]
        cos = math.cos(theta)
        sin = math.sin(theta)

        if i == 0 and cos <= -eps:  # ct ≤ ~0 => pointing left or tangent
            return True
        if i == self.N_x1 - 1 and cos >= eps:  # ct ≥ ~0 => pointing right or tangent
            return True
        if j == 0 and sin <= -eps:  # st ≤ ~0 => pointing down or tangent
            return True
        if j == self.N_x2 - 1 and sin >= eps:  # st ≥ ~0 => pointing up or tangent
            return True

        return False

    def accept_boundary_node(self, idx: tuple[int, int, int]) -> None:
        i, j, k = idx
        self.u[idx] = 0.0
        self.sigma[idx] = 0.0
        self.y1[idx] = self.x1_coords[i]
        self.y2[idx] = self.x2_coords[j]
        self.theta_exit[idx] = self.theta_coords[k]
        self.status[idx] = NodeState.ACCEPTED

    def adjacent(self, idx: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        i, j, k = idx
        result = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.N_x1 and 0 <= nj < self.N_x2:
                result.append((ni, nj, k))
        for dk in (-1, 1):
            nk = (k + dk) % self.N_theta
            result.append((i, j, nk))
        return result

    def octant_neighbors(self, idx: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        i, j, k = idx
        theta = self.theta_coords[k]
        cos = math.cos(theta)
        sin = math.sin(theta)

        n0 = self.slowness_at(i, j)
        dn2_dx1, dn2_dx2 = self.slowness_gradient(i, j)
        grad_theta = (cos * dn2_dx2 - sin * dn2_dx1) / (2 * n0)

        # < instead of >= to reverse direction (interior expansion)
        di_direction = 1 if cos < 0 else -1
        dj_direction = 1 if sin < 0 else -1
        dk_direction = 1 if grad_theta < 0 else -1

        neighbors = []
        for di in (0, di_direction):
            for dj in (0, dj_direction):
                for dk in (0, dk_direction):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    nk = (k + dk) % self.N_theta
                    if 0 <= ni < self.N_x1 and 0 <= nj < self.N_x2:
                        neighbors.append((ni, nj, nk))
        return neighbors

    def compute_characteristic(
        self, idx: tuple[int, int, int]
    ) -> CharacteristicResult2D:
        return local_cell_characteristic_2d(self, idx)

    def commit_characteristic(
        self,
        idx: tuple[int, int, int],
        result: CharacteristicResult2D,
    ) -> None:
        self.u[idx] = result.u
        self.sigma[idx] = result.sigma
        self.y1[idx] = result.y1
        self.y2[idx] = result.y2
        self.theta_exit[idx] = result.theta_exit
        self.status[idx] = NodeState.CONSIDERED

    # ===== numerical slowness gradient =====

    def _numerical_slowness_gradient(self, i: int, j: int) -> tuple[float, float]:
        """Returns (∂n²/∂x1, ∂n²/∂x2) by computing central differences at (i, j)."""

        n2 = self._n2

        if i == 0:
            dn2_dx1 = (n2[1, j] - n2[0, j]) / self.dx1
        elif i == self.N_x1 - 1:
            dn2_dx1 = (n2[-1, j] - n2[-2, j]) / self.dx1
        else:
            dn2_dx1 = (n2[i + 1, j] - n2[i - 1, j]) / (2 * self.dx1)

        if j == 0:
            dn2_dx2 = (n2[i, 1] - n2[i, 0]) / self.dx2
        elif j == self.N_x2 - 1:
            dn2_dx2 = (n2[i, -1] - n2[i, -2]) / self.dx2
        else:
            dn2_dx2 = (n2[i, j + 1] - n2[i, j - 1]) / (2 * self.dx2)

        return float(dn2_dx1), float(dn2_dx2)
