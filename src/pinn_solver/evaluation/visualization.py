"""Minimal plotting for a trained escape-equation PINN.

Three figures per run (2D and 3D variants):

- :func:`plot_training_loss` — loss curves over the training history.
- :func:`plot_all_angles` / :func:`plot_all_angles_3d` — û heatmaps on a grid
  of evenly-spaced θ slices (3D: on the mid-x3 plane at equator φ).
- :func:`plot_error_all_angles` — signed error (PINN − exact ODE) on the same
  slices; the reference rays are integrated on the fly with
  :func:`pinn_solver.evaluation.ode_reference.integrate_escape_rays`, so the plot works
  for any medium and stays exact at the caustic.
"""

import math
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..networks import EscapeNet  # noqa: E402
from .ode_reference import integrate_escape_rays  # noqa: E402

_EQUATOR_PHI = math.pi / 2


@contextmanager
def _on_cpu_eval(model: EscapeNet):
    """Temporarily relocate ``model`` to CPU + eval mode for plotting.

    Restores the original device and training mode on exit so a follow-up
    training step is not surprised to find its model on the wrong device.
    """
    original_device = next(model.parameters()).device
    was_training = model.training
    try:
        model.to("cpu")
        model.eval()
        yield model
    finally:
        model.to(original_device)
        if was_training:
            model.train()


def _evaluate_on_grid(
    model: EscapeNet,
    physical_size,
    theta_value: float,
    n_x1: int = 100,
    n_x2: int = 100,
):
    """û of a 2D model on a regular (x1, x2) grid at fixed θ (transposed
    for the ``contourf`` convention)."""
    L_x1, L_x2 = physical_size
    x1_1d = torch.linspace(0, L_x1, n_x1)
    x2_1d = torch.linspace(0, L_x2, n_x2)
    X1_t, X2_t = torch.meshgrid(x1_1d, x2_1d, indexing="ij")
    with _on_cpu_eval(model) as cpu_model, torch.no_grad():
        u = cpu_model(
            X1_t.reshape(-1),
            X2_t.reshape(-1),
            torch.full((X1_t.numel(),), theta_value),
        )[0]
    return X1_t.numpy().T, X2_t.numpy().T, u.reshape(n_x1, n_x2).numpy().T


def _evaluate_on_grid_3d(
    model: EscapeNet,
    physical_size,
    x3_value: float,
    phi_value: float,
    theta_value: float,
    n_x1: int = 80,
    n_x2: int = 80,
):
    """û of a spherical-signature 3D model on a fixed-(x3, φ, θ) slice."""
    L_x1, L_x2, _ = physical_size
    x1_1d = torch.linspace(0, L_x1, n_x1)
    x2_1d = torch.linspace(0, L_x2, n_x2)
    X1_t, X2_t = torch.meshgrid(x1_1d, x2_1d, indexing="ij")
    flat = X1_t.reshape(-1)
    with _on_cpu_eval(model) as cpu_model, torch.no_grad():
        u = cpu_model(
            flat,
            X2_t.reshape(-1),
            torch.full_like(flat, x3_value),
            torch.full_like(flat, phi_value),
            torch.full_like(flat, theta_value),
        )[0]
    return X1_t.numpy().T, X2_t.numpy().T, u.reshape(n_x1, n_x2).numpy().T


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------
def plot_training_loss(history, ax=None):
    """Loss curves (total, PDE, BC, and data/circle when active)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.loss) + 1)
    ax.semilogy(epochs, history.loss, label="Total")
    ax.semilogy(epochs, history.loss_pde, label="PDE", alpha=0.7)
    ax.semilogy(epochs, history.loss_bc, label="BC", alpha=0.7)
    if any(v > 0 for v in history.loss_data):
        ax.semilogy(epochs, history.loss_data, label="Data", alpha=0.7)
    if any(v > 0 for v in history.loss_circle):
        ax.semilogy(epochs, history.loss_circle, label="Circle", alpha=0.7)
    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Solution heatmaps
# ---------------------------------------------------------------------------
def plot_all_angles(model: EscapeNet, physical_size, n_plots: int = 8, **kwargs):
    """û heatmaps for evenly-spaced θ slices (2D)."""
    thetas = [2 * math.pi * i / n_plots for i in range(n_plots)]
    cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    defaults = {"levels": 30, "cmap": "viridis"}
    defaults.update(kwargs)
    for idx, th in enumerate(thetas):
        X1, X2, u = _evaluate_on_grid(model, physical_size, th)
        cs = axes[idx].contourf(X1, X2, u, **defaults)
        plt.colorbar(cs, ax=axes[idx])
        axes[idx].set_title(f"θ = {math.degrees(th):.0f}°")
        axes[idx].set_aspect("equal")
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("PINN escape time û")
    fig.tight_layout()
    return fig


def plot_all_angles_3d(model: EscapeNet, physical_size, n_plots: int = 8, **kwargs):
    """û heatmaps for evenly-spaced θ slices on the mid-x3 plane at equator φ
    (3D, spherical-signature model)."""
    thetas = [2 * math.pi * i / n_plots for i in range(n_plots)]
    cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    defaults = {"levels": 30, "cmap": "viridis"}
    defaults.update(kwargs)
    mid_x3 = 0.5 * physical_size[2]
    for idx, th in enumerate(thetas):
        X1, X2, u = _evaluate_on_grid_3d(
            model, physical_size, mid_x3, _EQUATOR_PHI, th
        )
        cs = axes[idx].contourf(X1, X2, u, **defaults)
        plt.colorbar(cs, ax=axes[idx])
        axes[idx].set_title(f"θ = {math.degrees(th):.0f}°")
        axes[idx].set_aspect("equal")
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("PINN û  (mid-x3 plane, equator φ)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Signed error vs the exact characteristic-ODE reference
# ---------------------------------------------------------------------------
def _ode_u_on_grid(cfg, X1, X2, theta_value: float, x3_value=None, phi_value=None):
    """Exact û on a plotting grid by integrating one ray per grid node.

    2D when ``x3_value is None``; otherwise the 3D slice at fixed (x3, φ).
    Non-exited (trapped) rays come back NaN and stay blank in the plot.
    """
    n_fn = cfg.slowness.get_slowness()
    g_fn = cfg.slowness.get_slowness_gradient()
    L = cfg.physical_size.as_tuple()
    x1 = torch.tensor(X1.T.ravel(), dtype=torch.float64)
    x2 = torch.tensor(X2.T.ravel(), dtype=torch.float64)
    if x3_value is None:
        x = torch.stack([x1, x2], dim=1)
        d = torch.stack(
            [torch.full_like(x1, math.cos(theta_value)),
             torch.full_like(x1, math.sin(theta_value))], dim=1,
        )
    else:
        x = torch.stack([x1, x2, torch.full_like(x1, x3_value)], dim=1)
        sp = math.sin(phi_value)
        d = torch.stack(
            [torch.full_like(x1, sp * math.cos(theta_value)),
             torch.full_like(x1, sp * math.sin(theta_value)),
             torch.full_like(x1, math.cos(phi_value))], dim=1,
        )
    res = integrate_escape_rays(x, d, n_fn, g_fn, L)
    u = res["u"].numpy().reshape(X1.T.shape).T
    return u


def plot_error_all_angles(model: EscapeNet, cfg, n_plots: int = 8, n_grid: int = 60):
    """Signed error (PINN − exact ODE) for evenly-spaced θ slices.

    2D models plot the (x1, x2) plane; 3D models (spherical signature — pass
    ``model.spherical_view()`` for Cartesian nets) plot the mid-x3 plane at
    equator φ. One reference ray is integrated per grid node.
    """
    is_3d = cfg.physical_size.dim == 3
    L = cfg.physical_size.as_tuple()
    thetas = [2 * math.pi * i / n_plots for i in range(n_plots)]
    cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, th in enumerate(thetas):
        if is_3d:
            X1, X2, u_pinn = _evaluate_on_grid_3d(
                model, L, 0.5 * L[2], _EQUATOR_PHI, th, n_x1=n_grid, n_x2=n_grid
            )
            u_ref = _ode_u_on_grid(cfg, X1, X2, th, x3_value=0.5 * L[2],
                                   phi_value=_EQUATOR_PHI)
        else:
            X1, X2, u_pinn = _evaluate_on_grid(
                model, L, th, n_x1=n_grid, n_x2=n_grid
            )
            u_ref = _ode_u_on_grid(cfg, X1, X2, th)
        diff = u_pinn - u_ref
        vmax = float(np.nanmax(np.abs(diff))) or 1e-12
        cs = axes[idx].pcolormesh(
            X1, X2, diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        plt.colorbar(cs, ax=axes[idx])
        axes[idx].set_title(f"θ = {math.degrees(th):.0f}°")
        axes[idx].set_aspect("equal")
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    title = "PINN − exact ODE"
    if is_3d:
        title += "  (mid-x3 plane, equator φ)"
    fig.suptitle(title)
    fig.tight_layout()
    return fig
