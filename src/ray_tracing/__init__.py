"""Lagrangian ray tracing for the eikonal Hamiltonian H = ½(|p|² - n²).

Integrates individual ray trajectories (the Lagrangian view of the problem
the escape solvers treat in phase space) and fans them out to draw
wavefronts. See :mod:`ray_tracing.examples` for runnable demos.
"""

from .tracer import LagrangianRayTracer2D, RayState2D

__all__ = ["LagrangianRayTracer2D", "RayState2D"]
