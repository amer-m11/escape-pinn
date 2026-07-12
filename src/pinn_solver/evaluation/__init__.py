"""Evaluation of a trained PINN.

- :mod:`.ode_reference`: the exact characteristic-ODE reference (per-ray
  integration)
- :mod:`.metrics`: the compact ``metrics.json`` run summary
- :mod:`.visualization`: the diagnostic figures (loss, û heatmaps,
  signed error vs the ODE reference)
- :mod:`.mesh_cache`: disk cache primitives for the reference tables
"""

from .metrics import compute_run_metrics, save_run_metrics
from .ode_reference import (
    get_or_build_ode_reference_points_2d,
    get_or_build_ode_reference_points_full,
    integrate_escape_rays,
)
from .visualization import (
    plot_all_angles,
    plot_all_angles_3d,
    plot_error_all_angles,
    plot_training_loss,
)

__all__ = [
    "compute_run_metrics",
    "save_run_metrics",
    "get_or_build_ode_reference_points_2d",
    "get_or_build_ode_reference_points_full",
    "integrate_escape_rays",
    "plot_all_angles",
    "plot_all_angles_3d",
    "plot_error_all_angles",
    "plot_training_loss",
]
