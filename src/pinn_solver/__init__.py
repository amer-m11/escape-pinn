"""PINN solver for the Eikonal escape equations.

A Physics-Informed Neural Network that learns the five escape quantities
``(û, σ̂, ŷ₁, ŷ₂, θ̂)`` in the reduced phase space ``(x1, x2, θ)`` from a
single :class:`TrainConfiguration`. See :class:`pinn_solver.solver.PINNRunner`
for the top-level orchestrator and ``python -m pinn_solver.run --help`` for
the CLI entry point.
"""

from .boundary import BoundaryLoss, compute_boundary_loss, sample_boundary_points
from .configuration import TrainConfiguration
from .data import ReferenceSolution, ReferenceSolution3D, grid_to_tensors
from .loss import Loss, LossBreakdown
from .networks import (
    EscapeNet,
    EscapeNet2D,
    EscapeNet3D,
    FCMLPNet2D,
    FCMLPNet3D,
    NetworkFactory,
    SIRENNet2D,
    SIRENNet3D,
)
from .pde import compute_pde_residual
from .sampler import Samples, Samples2D, Samples3D, Sampler
from .solver import PINNRunner, TrainingResult
from .trainers import AdamTrainer, LbfgsTrainer, TrainHistory

__all__ = [
    "AdamTrainer",
    "BoundaryLoss",
    "EscapeNet",
    "EscapeNet2D",
    "EscapeNet3D",
    "FCMLPNet2D",
    "FCMLPNet3D",
    "LbfgsTrainer",
    "Loss",
    "LossBreakdown",
    "NetworkFactory",
    "PINNRunner",
    "ReferenceSolution",
    "ReferenceSolution3D",
    "SIRENNet2D",
    "SIRENNet3D",
    "Samples",
    "Samples2D",
    "Samples3D",
    "Sampler",
    "TrainConfiguration",
    "TrainHistory",
    "TrainingResult",
    "compute_boundary_loss",
    "compute_pde_residual",
    "grid_to_tensors",
    "sample_boundary_points",
]
