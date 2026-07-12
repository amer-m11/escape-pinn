"""Lagrangian ray integration for the eikonal Hamiltonian H = ½(|p|² - n²)."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class RayState2D:
    """One ray's phase-space state: position, momentum, accumulated time."""

    x1: float
    x2: float
    px1: float
    px2: float
    u: float

    def to_list(self) -> list[float]:
        return [self.x1, self.x2, self.px1, self.px2, self.u]


class LagrangianRayTracer2D:
    """Integrate 2D rays of the eikonal Hamiltonian H = ½(|p|² - n²).

    ``n_func``/``grad_n_func`` give the slowness field and its gradient.
    """

    def __init__(
        self,
        n_func: Callable[[float, float], float],
        grad_n_func: Callable[[float, float], tuple[float, float]],
    ):
        self.n = n_func
        self.grad_n = grad_n_func

    def ray_odes(self, sigma: float, state: list[float]) -> list[float]:
        """Right-hand side of the Lagrangian ray ODEs.

        dx/dσ = p,  dp/dσ = n∇n,  du/dσ = |p|².
        """
        x1, x2, px1, px2, u = state

        n_value = self.n(x1, x2)
        dn_dx1, dn_dx2 = self.grad_n(x1, x2)

        return [
            px1,
            px2,
            n_value * dn_dx1,
            n_value * dn_dx2,
            px1**2 + px2**2,
        ]

    def trace_ray(
        self,
        initial_state: RayState2D,
        sigma_linspace: tuple[float, float, int],
        **kwargs,
    ) -> Any:
        """Integrate one ray over the given σ range (scipy ``solve_ivp``)."""
        sigma_start, sigma_end, sigma_num = sigma_linspace
        sigma_eval = np.linspace(sigma_start, sigma_end, sigma_num)
        kwargs.setdefault("rtol", 1e-8)
        kwargs.setdefault("atol", 1e-8)
        return solve_ivp(
            fun=self.ray_odes,
            t_span=(sigma_start, sigma_end),
            y0=initial_state.to_list(),
            t_eval=sigma_eval,
            **kwargs,
        )

    def check_hamiltonian_error(self, ray_solution) -> float:
        """Max |H| along the ray — H is conserved (0) on an exact trajectory."""
        x1, x2, px1, px2, u = ray_solution.y
        p_mag_sq = px1**2 + px2**2
        n_val_sq = np.vectorize(self.n)(x1, x2) ** 2
        H = 0.5 * p_mag_sq - 0.5 * n_val_sq
        return np.max(np.abs(H))

    def fan_out(
        self,
        sigma_linspace: tuple[float, float, int],
        angles: np.ndarray = np.linspace(0, 2 * np.pi, 36, endpoint=False),
        initial_position: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> list[Any]:
        """Trace a fan of rays from one point, one per launch angle."""
        rays = []
        x1, x2 = initial_position

        # |p| = n(x0) puts the launch state on the H = 0 surface.
        p_mag = self.n(x1, x2)
        for angle in angles:
            state = RayState2D(
                x1=x1,
                x2=x2,
                px1=p_mag * np.cos(angle),
                px2=p_mag * np.sin(angle),
                u=0.0,
            )
            rays.append(self.trace_ray(state, sigma_linspace, **kwargs))
        return rays
