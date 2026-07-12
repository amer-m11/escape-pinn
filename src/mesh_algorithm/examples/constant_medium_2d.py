"""Constant-medium verification in 2D: n(x1, x2) = 1 everywhere.

With n = 1 rays are straight, so the escape time from any interior point
along direction θ is the exact distance to the first box face.

Usage:
    PYTHONPATH=src python -m mesh_algorithm.examples.constant_medium_2d
"""

import argparse
import math

import numpy as np

from mesh_algorithm import EscapeSolver, PhaseSpaceGrid
from mesh_algorithm.node_state import NodeState

_TIGHT = 1e-9
_LOOSE = 0.10


def _exit_time(position, direction, box_lengths):
    """Analytic distance to the first box face along ``direction``."""
    candidates = []
    for d_val, x_val, length in zip(direction, position, box_lengths):
        if d_val > _TIGHT:
            candidates.append((length - x_val) / d_val)
        elif d_val < -_TIGHT:
            candidates.append((-x_val) / d_val)
    return min(candidates) if candidates else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-x1", type=int, default=21)
    parser.add_argument("--n-x2", type=int, default=21)
    parser.add_argument("--n-theta", type=int, default=16)
    parser.add_argument("--output-dir", default="output/mesh")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    grid = PhaseSpaceGrid(
        (args.n_x1, args.n_x2, args.n_theta),
        (1.0, 1.0),
        slowness_fn=lambda x1, x2: 1.0,
        slowness_gradient_fn=lambda x1, x2: (0.0, 0.0),
    )
    solver = EscapeSolver(grid, track_failure_reasons=True)
    solver.solve()
    diag = solver.diagnostics()
    for line in diag.summary_lines():
        print(line)

    n_accepted = int(np.sum(grid.status == NodeState.ACCEPTED))
    if n_accepted != diag.total_nodes:
        raise ValueError(f"Only {n_accepted}/{diag.total_nodes} nodes accepted.")

    # Centre-node analytic exit-time check across every θ. The only "loose"
    # case is the corner-tied diagonal direction (|cos θ| ≈ |sin θ|).
    cx1, cx2 = args.n_x1 // 2, args.n_x2 // 2
    centre = (grid.x1_coords[cx1], grid.x2_coords[cx2])
    box = (grid.L_x1, grid.L_x2)
    n_tight = n_loose = n_fail = 0
    for k in range(args.n_theta):
        theta = grid.theta_coords[k]
        direction = (math.cos(theta), math.sin(theta))
        analytic = _exit_time(centre, direction, box)
        tied = abs(abs(direction[0]) - abs(direction[1])) < 1e-9
        tol = _LOOSE * analytic if tied else _TIGHT
        if abs(grid.u[cx1, cx2, k] - analytic) <= tol:
            n_tight, n_loose = n_tight + (not tied), n_loose + tied
        else:
            n_fail += 1
    print(f"Exit-time checks: tight={n_tight}, loose(corner-tied)={n_loose}, fail={n_fail}")
    if n_fail:
        raise ValueError(f"{n_fail} centre-node exit-time checks failed.")

    if not args.no_plots:
        import os

        from mesh_algorithm.visualization import plot_escape_slices

        os.makedirs(args.output_dir, exist_ok=True)
        fig = plot_escape_slices(grid)
        path = os.path.join(args.output_dir, "constant_medium_2d_escape.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
