"""Variable-medium demonstration in 3D: isotropic Gaussian lens.

n(x1, x2, x3) = 1 + A·exp(-|x - c|² / (2σ²)),  c = (1, 1, 1),  A = 0.5,
σ = 0.3, on a 2×2×2 box.

Usage:
    PYTHONPATH=src python -m mesh_algorithm.examples.variable_medium_3d
"""

import argparse
import math

import numpy as np

from mesh_algorithm import EscapeSolver, PhaseSpaceGrid
from mesh_algorithm.node_state import NodeState

_MIN_ACCEPTANCE = 99.0


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

    centre, A, sigma = (1.0, 1.0, 1.0), 0.5, 0.3

    def slowness(x1, x2, x3):
        r2 = (x1 - centre[0]) ** 2 + (x2 - centre[1]) ** 2 + (x3 - centre[2]) ** 2
        return 1.0 + A * math.exp(-r2 / (2 * sigma**2))

    def slowness_gradient(x1, x2, x3):
        """Analytic ∇(n²) = 2n·∇n."""
        r2 = (x1 - centre[0]) ** 2 + (x2 - centre[1]) ** 2 + (x3 - centre[2]) ** 2
        exp_term = math.exp(-r2 / (2 * sigma**2))
        n = 1.0 + A * exp_term
        dn_dx1 = A * exp_term * (-(x1 - centre[0]) / sigma**2)
        dn_dx2 = A * exp_term * (-(x2 - centre[1]) / sigma**2)
        dn_dx3 = A * exp_term * (-(x3 - centre[2]) / sigma**2)
        return 2 * n * dn_dx1, 2 * n * dn_dx2, 2 * n * dn_dx3

    grid = PhaseSpaceGrid(
        (args.n_x1, args.n_x2, args.n_x3, args.n_phi, args.n_theta),
        (2.0, 2.0, 2.0),
        slowness_fn=slowness,
        slowness_gradient_fn=slowness_gradient,
    )
    solver = EscapeSolver(grid, track_failure_reasons=True)
    solver.solve()
    diag = solver.diagnostics()
    for line in diag.summary_lines():
        print(line)

    n_accepted = int(np.sum(grid.status == NodeState.ACCEPTED))
    pct = 100.0 * n_accepted / diag.total_nodes
    print(f"Nodes accepted: {n_accepted}/{diag.total_nodes} ({pct:.1f}%)")
    if pct < _MIN_ACCEPTANCE:
        raise ValueError(f"Acceptance {pct:.1f}% below minimum {_MIN_ACCEPTANCE}%.")

    if not args.no_plots:
        import os

        from mesh_algorithm.visualization import plot_escape_slices

        os.makedirs(args.output_dir, exist_ok=True)
        fig = plot_escape_slices(grid)
        path = os.path.join(args.output_dir, "variable_medium_3d_escape.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
