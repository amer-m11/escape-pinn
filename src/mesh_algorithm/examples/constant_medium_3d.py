"""Constant-medium verification in 3D: n(x1, x2, x3) = 1 everywhere.

With n = 1 rays are straight, so the escape time from any interior point
along direction (φ, θ) is the exact distance to the first cuboid face.

Usage:
    PYTHONPATH=src python -m mesh_algorithm.examples.constant_medium_3d
"""

import argparse
import math

import numpy as np

from mesh_algorithm import EscapeSolver, PhaseSpaceGrid
from mesh_algorithm.node_state import NodeState

_TIGHT = 1e-9
_LOOSE = 0.10


def _exit_time(position, direction, box_lengths):
    """Analytic distance to the first cuboid face along ``direction``."""
    candidates = []
    for d_val, x_val, length in zip(direction, position, box_lengths):
        if d_val > _TIGHT:
            candidates.append((length - x_val) / d_val)
        elif d_val < -_TIGHT:
            candidates.append((-x_val) / d_val)
    return min(candidates) if candidates else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-x1", type=int, default=11)
    parser.add_argument("--n-x2", type=int, default=11)
    parser.add_argument("--n-x3", type=int, default=11)
    parser.add_argument("--n-phi", type=int, default=4)
    parser.add_argument("--n-theta", type=int, default=8)
    parser.add_argument("--output-dir", default="output/mesh")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    grid = PhaseSpaceGrid(
        (args.n_x1, args.n_x2, args.n_x3, args.n_phi, args.n_theta),
        (1.0, 1.0, 1.0),
        slowness_fn=lambda x1, x2, x3: 1.0,
        slowness_gradient_fn=lambda x1, x2, x3: (0.0, 0.0, 0.0),
    )
    solver = EscapeSolver(grid, track_failure_reasons=True)
    solver.solve()
    diag = solver.diagnostics()
    for line in diag.summary_lines():
        print(line)

    n_accepted = int(np.sum(grid.status == NodeState.ACCEPTED))
    if n_accepted != diag.total_nodes:
        raise ValueError(f"Only {n_accepted}/{diag.total_nodes} nodes accepted.")

    # Centre-node analytic exit-time check across every (φ, θ). Directions
    # whose axis components nearly tie (ray pointing at an edge/corner) get
    # the loose tolerance, all others the tight one.
    ci, cj, cl = args.n_x1 // 2, args.n_x2 // 2, args.n_x3 // 2
    centre = (grid.x1_coords[ci], grid.x2_coords[cj], grid.x3_coords[cl])
    box = (grid.L_x1, grid.L_x2, grid.L_x3)
    n_tight = n_loose = n_fail = 0
    for m in range(args.n_phi):
        for k in range(args.n_theta):
            phi, theta = grid.phi_coords[m], grid.theta_coords[k]
            d = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            )
            analytic = _exit_time(centre, d, box)
            ad = sorted(abs(v) for v in d)
            tied = (ad[2] - ad[1]) < 1e-2 or (ad[1] - ad[0]) < 1e-2
            tol = _LOOSE * analytic if tied else _TIGHT
            if abs(grid.u[ci, cj, cl, m, k] - analytic) <= tol:
                n_tight, n_loose = n_tight + (not tied), n_loose + tied
            else:
                n_fail += 1
    print(f"Exit-time checks: tight={n_tight}, loose(tied)={n_loose}, fail={n_fail}")
    if n_fail:
        raise ValueError(f"{n_fail} centre-node exit-time checks failed.")

    if not args.no_plots:
        import os

        from mesh_algorithm.visualization import plot_escape_slices

        os.makedirs(args.output_dir, exist_ok=True)
        fig = plot_escape_slices(grid)
        path = os.path.join(args.output_dir, "constant_medium_3d_escape.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
