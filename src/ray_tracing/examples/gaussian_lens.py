"""Wavefronts through a Gaussian lens: n = 1 + exp(-(x₁² + x₂²)/2).

A dense 360-ray fan from a point left of the lens. The wavefronts are
interpolated along each ray at fixed travel-time values, showing how the
lens folds them (the caustic).

Usage:
    PYTHONPATH=src python -m ray_tracing.examples.gaussian_lens
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ray_tracing import LagrangianRayTracer2D


def n_lens(x1: float, x2: float) -> float:
    return 1.0 + 1.0 * np.exp(-(x1**2 + x2**2) / 2.0)


def grad_n_lens(x1: float, x2: float) -> tuple[float, float]:
    exp_term = np.exp(-(x1**2 + x2**2) / 2.0)
    return -1.0 * x1 * exp_term, -1.0 * x2 * exp_term


def main():
    tracer = LagrangianRayTracer2D(n_lens, grad_n_lens)

    rays = tracer.fan_out(
        sigma_linspace=(0.0, 10, 200),
        angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
        initial_position=(-3.0, 0.0),
    )

    max_error = max(tracer.check_hamiltonian_error(ray) for ray in rays)
    print(f"Maximum Hamiltonian error: {max_error:.2e}")
    if max_error > 1e-6:
        print("Warning: Hamiltonian error exceeds tolerance.")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: the refractive-index field.
    X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    N_val = np.vectorize(tracer.n)(X, Y)
    im = ax.imshow(
        N_val,
        extent=(-4, 4, -4, 4),
        origin="lower",
        cmap="Blues",
        alpha=0.8,
        interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, label="Refractive Index $n(x)$")

    # Wavefronts: interpolate each ray's position at fixed travel-time values
    # (u strictly increases along a ray, so 1D interpolation is stable).
    for u_val in [0.25 * i for i in range(1, 100)]:
        wavefront_x1, wavefront_x2 = [], []
        for ray in rays:
            u_array = ray.y[4]
            if u_array[0] <= u_val <= u_array[-1]:
                wavefront_x1.append(np.interp(u_val, u_array, ray.y[0]))
                wavefront_x2.append(np.interp(u_val, u_array, ray.y[1]))
            else:
                # A ray that falls short breaks the loop with NaN (no line
                # drawn across the gap).
                wavefront_x1.append(np.nan)
                wavefront_x2.append(np.nan)

        if wavefront_x1 and not np.isnan(wavefront_x1).all():
            # The fan covers 360°: close the loop when both ends exist.
            if not np.isnan(wavefront_x1[0]) and not np.isnan(wavefront_x1[-1]):
                wavefront_x1.append(wavefront_x1[0])
                wavefront_x2.append(wavefront_x2[0])
            ax.plot(wavefront_x1, wavefront_x2, color="black", linewidth=0.5, zorder=3)

    ax.set_aspect("equal")
    ax.set_title("Wavefronts through a Lens ($n(x) = 1 + exp(-(x_1^2 + x_2^2)/2)$)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    output_path = "./output/wavefront_gaussian_lens.png"
    os.makedirs("./output", exist_ok=True)
    if "agg" in plt.get_backend().lower():
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
