"""Rays and wavefronts in a linear-gradient medium: n = 1.5 + 0.2·x₁.

Rays curve toward the slower region (larger n).

Usage:
    PYTHONPATH=src python -m ray_tracing.examples.linear_gradient
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ray_tracing import LagrangianRayTracer2D


def main():
    tracer = LagrangianRayTracer2D(
        n_func=lambda x1, x2: 1.5 + 0.2 * x1,
        grad_n_func=lambda x1, x2: (0.2, 0.0),
    )

    rays = tracer.fan_out(
        sigma_linspace=(0.0, 5, 200),
        angles=np.linspace(0, 2 * np.pi, 36, endpoint=False),
    )

    max_error = max(tracer.check_hamiltonian_error(ray) for ray in rays)
    print(f"Maximum Hamiltonian error: {max_error:.2e}")
    if max_error > 1e-6:
        print("Warning: Hamiltonian error exceeds tolerance.")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: the refractive-index field.
    X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    N_val = np.vectorize(tracer.n)(X, Y)
    im = ax.contourf(X, Y, N_val, levels=30, cmap="Blues", alpha=0.4)
    fig.colorbar(im, ax=ax, label="Refractive Index $n(x)$")

    all_x1, all_x2, all_u = [], [], []
    for i, ray in enumerate(rays):
        label = "Ray Trajectories" if i == 0 else None
        ax.plot(ray.y[0], ray.y[1], color="dimgray", linewidth=1, zorder=2, label=label)
        all_x1.extend(ray.y[0])
        all_x2.extend(ray.y[1])
        all_u.extend(ray.y[4])

    # Wavefronts = contours of constant travel time u.
    contour = ax.tricontour(
        all_x1, all_x2, all_u, levels=[1, 2, 3, 4, 5, 6, 7],
        colors="crimson", linewidths=1, zorder=3,
    )
    ax.clabel(contour, levels=[2, 4], inline=True, fontsize=12, fmt="u=%1.0f")

    ax.set_aspect("equal")
    ax.set_title("Rays and Wavefronts in a Linear Gradient ($n(x) = 1.5 + 0.2x_1$)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend(loc="upper right")

    output_path = "./output/wavefront_linear_gradient.png"
    os.makedirs("./output", exist_ok=True)
    if "agg" in plt.get_backend().lower():
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
