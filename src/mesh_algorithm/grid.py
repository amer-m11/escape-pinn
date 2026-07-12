from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np


class Grid(ABC):
    """Defines a protocol to be implemented by grid classes with any dimension.

    The grid should store all state needed by the mesh solver.
    """

    status: np.ndarray
    """node states: FAR, CONSIDERED, ACCEPTED"""
    sigma: np.ndarray
    """σ̂: parametric distance along the characteristic to the exit point."""
    u: np.ndarray
    """û: escape time starting from any node."""

    @abstractmethod
    def iter_indices(self) -> Iterator[tuple[int, ...]]:
        """Yield every phase-space node index."""
        ...

    @abstractmethod
    def is_boundary(self, idx: tuple) -> bool:
        """Return True if the node at idx is a boundary node."""
        ...

    @abstractmethod
    def points_outward(self, idx: tuple) -> bool:
        """Return True if the boundary node at idx points outward."""
        ...

    @abstractmethod
    def accept_boundary_node(self, idx: tuple) -> None:
        """Set the boundary-condition values at ``idx`` and mark ACCEPTED.

        For an outward-pointing ray starting on ∂D: û = σ̂ = 0, the exit
        position is the starting point itself, and the exit direction is
        the starting direction.
        """
        ...

    @abstractmethod
    def adjacent(self, idx: tuple) -> list[tuple]:
        """Return immediate spatial and angular neighbors (±1).

        θ is periodic, so angular neighbors wrap around.
        """
        ...

    @abstractmethod
    def octant_neighbors(self, idx: tuple) -> list[tuple]:
        """Return neighbor indices in the octant the ray at ``idx``
        points toward.

        The march expands outward from the boundary, so the relevant
        neighbors are those into which the characteristic flows when
        traced backward (i.e. the octant opposite the ray direction).
        """
        ...

    @abstractmethod
    def compute_characteristic(self, idx: tuple) -> Any:
        """Run the parabolic-ray characteristic and return a tentative update.

        the returned object should have a .sigma attribute"""
        ...

    @abstractmethod
    def commit_characteristic(self, idx: tuple, result: Any) -> None:
        """Write all per-node escape quantities from ``result`` and mark CONSIDERED."""
        ...
