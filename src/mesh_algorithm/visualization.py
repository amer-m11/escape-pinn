"""Minimal plotting for a solved mesh grid.

One public function: :func:`plot_escape_slices` — the escape time û at a
grid of evenly-spaced θ slices in a single figure (2D grids plot the
(x1, x2) plane per θ; 3D grids the mid-x3 plane at the middle φ node).
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_escape_slices(grid, n_plots: int = 8):
    """û heatmaps at ``n_plots`` evenly-spaced θ slices in one figure.

    ``grid`` is a solved 2D or 3D ``PhaseSpaceGrid`` (dispatched on the
    presence of ``x3_coords``). Returns the matplotlib figure.
    """
    is_3d = hasattr(grid, "x3_coords")
    step = max(1, grid.N_theta // n_plots)
    indices = list(range(0, grid.N_theta, step))[:n_plots]

    cols = min(4, len(indices))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_i, k in enumerate(indices):
        ax = axes[ax_i]
        if is_3d:
            u = grid.u[:, :, grid.N_x3 // 2, grid.N_phi // 2, k]
        else:
            u = grid.u[:, :, k]
        pc = ax.pcolormesh(grid.x1_coords, grid.x2_coords, u.T)
        fig.colorbar(pc, ax=ax, label="û")
        ax.set_title(f"θ = {math.degrees(grid.theta_coords[k]):.1f}°")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    for ax_i in range(len(indices), len(axes)):
        axes[ax_i].set_visible(False)

    if is_3d:
        phi = grid.phi_coords[grid.N_phi // 2]
        fig.suptitle(f"Escape time û  (mid-x3, φ = {math.degrees(phi):.1f}°)")
    else:
        fig.suptitle("Escape time û")
    fig.tight_layout()
    return fig
