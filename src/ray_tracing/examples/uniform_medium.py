"""Rays and wavefronts in a uniform medium: n ≡ 1 (straight rays, circles).

Usage:
    PYTHONPATH=src python -m ray_tracing.examples.uniform_medium
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ray_tracing import LagrangianRayTracer2D


def main():
    tracer = LagrangianRayTracer2D(
        n_func=lambda x1, x2: 1.0,
        grad_n_func=lambda x1, x2: (0.0, 0.0),
    )

    rays = tracer.fan_out(
        sigma_linspace=(0.0, 15.0, 150),
        angles=np.linspace(0, 2 * np.pi, 36, endpoint=False),
    )

    max_error = max(tracer.check_hamiltonian_error(ray) for ray in rays)
    print(f"Maximum Hamiltonian error: {max_error:.2e}")
    if max_error > 1e-6:
        print("Warning: Hamiltonian error exceeds tolerance.")

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, sol in enumerate(rays):
        label = "Ray Trajectories" if i == 0 else None
        ax.plot(sol.y[0], sol.y[1], color="dimgray", linewidth=1, zorder=1, label=label)

    # Wavefronts = contours of constant travel time u along the rays.
    for target_u in [1.0 * i for i in range(1, 8)]:
        wavefront_x1, wavefront_x2 = [], []
        for sol in rays:
            for i in range(len(sol.t)):
                if np.isclose(sol.y[4][i], target_u, atol=0.05):
                    wavefront_x1.append(sol.y[0][i])
                    wavefront_x2.append(sol.y[1][i])
                    break
        wavefront_x1.append(wavefront_x1[0])
        wavefront_x2.append(wavefront_x2[0])
        label = "Wavefronts" if target_u == 1.0 else ""
        ax.plot(wavefront_x1, wavefront_x2, color="crimson", linewidth=1,
                zorder=2, label=label)

    ax.set_aspect("equal")
    ax.set_title("Ray Trajectories and Wavefronts in a Uniform Medium")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.legend(loc="upper right")

    output_path = "./output/wavefront_uniform_medium.png"
    os.makedirs("./output", exist_ok=True)
    if "agg" in plt.get_backend().lower():
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
