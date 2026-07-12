"""Compact per-run metrics for the PINN escape-equation solver.

One trained run is summarized into a small ``metrics.json``:

- ``run``      — provenance (timestamp, git SHA, host, torch/CUDA versions)
- ``config``   — the knobs the run was driven by (network, loss, optimizers)
- ``wall_time_seconds`` — per-phase timings supplied by the runner
- ``loss``     — final (and last-Adam) loss snapshots from the history
- ``pde_residual`` — per-channel |residual| max/mean at a random interior
  sample (finite-masked; ``pde_nan_fraction`` flags non-finite entries)
- ``bc_residual``  — per-component boundary loss on a fresh boundary set
- ``vs_reference_volume`` — û accuracy vs the EXACT characteristic-ODE
  reference at a Sobol sample over the whole trained phase-space volume
  (L2 / L∞ / p99 + a half-sample convergence self-check)
"""

import json
import math
import platform
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from ..boundary import (
    compute_boundary_loss_2d,
    compute_boundary_loss_3d,
    compute_boundary_loss_cartesian_3d,
    sample_boundary_points_2d,
    sample_boundary_points_3d,
    sample_boundary_points_cartesian_3d,
)
from ..configuration import TrainConfiguration
from ..networks import EscapeNet
from ..pde import (
    compute_pde_residual_2d,
    compute_pde_residual_3d,
    compute_pde_residual_cartesian_3d,
)
from ..trainers import TrainHistory

SCHEMA_VERSION = "public-1"

PDE_CHANNEL_NAMES = ("res_u", "res_sigma", "res_y1", "res_y2", "res_cos", "res_sin")
BC_COMPONENT_NAMES = ("u", "sigma", "y1", "y2", "cos", "sin")
PDE_CHANNEL_NAMES_3D = (
    "res_u", "res_sigma", "res_y1", "res_y2", "res_y3",
    "res_cos_theta", "res_sin_theta", "res_cos_phi", "res_sin_phi",
)
BC_COMPONENT_NAMES_3D = (
    "u", "sigma", "y1", "y2", "y3",
    "cos_theta", "sin_theta", "cos_phi", "sin_phi",
)
PDE_CHANNEL_NAMES_CART = (
    "res_u", "res_sigma", "res_y1", "res_y2", "res_y3",
    "res_d1", "res_d2", "res_d3",
)
BC_COMPONENT_NAMES_CART = ("u", "sigma", "y1", "y2", "y3", "d1", "d2", "d3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextmanager
def _on_cpu_eval(model: EscapeNet):
    """Temporarily move ``model`` to CPU + eval mode for measurement.

    Restores both on exit so a follow-up training step does not find its
    model on the wrong device.
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


def _git_sha() -> Optional[str]:
    """Return the current commit short SHA, or ``None`` if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _gpu_name() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - surface as missing rather than crash
        return None


def _network_shape(model) -> dict:
    """Return a JSON-serializable architecture description for ``model``."""
    if hasattr(model, "arch_details") and callable(model.arch_details):
        return model.arch_details()
    info: dict = {
        "arch_name": getattr(model, "arch_name", "unknown"),
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
    if hasattr(model, "N_OUTPUTS"):
        info.setdefault("n_outputs", int(model.N_OUTPUTS))
    return info


def _err_stats(errs: np.ndarray) -> dict:
    """L2 / L∞ / p99 / mean|err| summary of an error sample."""
    errs = np.asarray(errs, dtype=np.float64).ravel()
    abs_e = np.abs(errs)
    n = int(abs_e.size)
    return {
        "L2": float(np.sqrt(np.mean(errs ** 2))) if n else 0.0,
        "L_inf": float(abs_e.max()) if n else 0.0,
        "p99": float(np.percentile(abs_e, 99)) if n else 0.0,
        "mean_abs": float(abs_e.mean()) if n else 0.0,
        "n_points": n,
    }


# ---------------------------------------------------------------------------
# Loss history
# ---------------------------------------------------------------------------
def _final_loss_section(history: TrainHistory, adam_epochs: int) -> dict:
    """Return last-Adam and last-overall (post-L-BFGS) loss snapshots."""
    n = len(history.loss)

    def _at(i: int) -> dict:
        snap = {
            "total": float(history.loss[i]),
            "pde": float(history.loss_pde[i]),
            "bc": float(history.loss_bc[i]),
            "data": float(history.loss_data[i]),
        }
        if i < len(history.loss_circle):
            snap["circle"] = float(history.loss_circle[i])
        return snap

    section: dict = {"history_length": n, "adam_epochs": adam_epochs}
    if n == 0:
        return section
    last_adam_idx = min(adam_epochs, n) - 1
    section["last_adam"] = _at(last_adam_idx)
    section["final"] = _at(n - 1)
    return section


# ---------------------------------------------------------------------------
# PDE residual health at a random interior sample
# ---------------------------------------------------------------------------
def _pde_residual_stats(model, cfg: TrainConfiguration, n_points: int = 4096) -> dict:
    """Per-channel |PDE residual| max/mean at a uniform interior sample (Finite-masked)"""
    L = cfg.physical_size.as_tuple()
    is_3d = cfg.physical_size.dim == 3
    is_cartesian = is_3d and cfg.parametrization == "cartesian"
    slowness_fn = cfg.slowness.get_slowness()
    slowness_grad_fn = cfg.slowness.get_slowness_gradient()
    gen = torch.Generator().manual_seed(0)

    def _u(n):
        return torch.rand(n_points, generator=gen)

    if not is_3d:
        x1, x2 = _u(n_points) * L[0], _u(n_points) * L[1]
        theta = _u(n_points) * 2 * math.pi
        res = compute_pde_residual_2d(
            model, x1, x2, theta, slowness_fn, slowness_grad_fn
        )
        names = PDE_CHANNEL_NAMES
    elif is_cartesian:
        x1, x2, x3 = (_u(n_points) * L[i] for i in range(3))
        d = torch.nn.functional.normalize(
            torch.randn(n_points, 3, generator=gen), dim=1
        )
        res = compute_pde_residual_cartesian_3d(
            model, x1, x2, x3, d[:, 0], d[:, 1], d[:, 2],
            slowness_fn, slowness_grad_fn,
        )
        names = PDE_CHANNEL_NAMES_CART
    else:
        delta = cfg.boundary.resolved_phi_delta()
        x1, x2, x3 = (_u(n_points) * L[i] for i in range(3))
        phi = delta + _u(n_points) * (math.pi - 2 * delta)
        theta = _u(n_points) * 2 * math.pi
        res = compute_pde_residual_3d(
            model, x1, x2, x3, phi, theta, slowness_fn, slowness_grad_fn,
            pole_safe=bool(cfg.loss.pole_safe_residual),
        )
        names = PDE_CHANNEL_NAMES_3D

    per_channel = {}
    n_total = n_nonfinite = 0
    for name, r in zip(names, res):
        a = r.detach().to(torch.float64).abs().cpu().numpy().ravel()
        finite = np.isfinite(a)
        n_total += a.size
        n_nonfinite += int(a.size - finite.sum())
        af = a[finite]
        per_channel[name] = {
            "max": float(af.max()) if af.size else None,
            "mean": float(af.mean()) if af.size else None,
        }
    return {
        "per_channel": per_channel,
        "channel_names": list(names),
        "n_points": int(n_points),
        "nonfinite_fraction": (n_nonfinite / n_total) if n_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Boundary-condition residual health
# ---------------------------------------------------------------------------
def _bc_residual_stats(model, cfg: TrainConfiguration) -> dict:
    """Per-component BC loss on a freshly sampled boundary set."""
    L = cfg.physical_size.as_tuple()
    is_3d = cfg.physical_size.dim == 3
    is_cartesian = is_3d and cfg.parametrization == "cartesian"
    n_edge = int(cfg.boundary.physical_boundary_samples)
    n_theta = int(cfg.boundary.theta_samples_at_boundary)

    with torch.no_grad():
        if not is_3d:
            x1, x2, th = sample_boundary_points_2d(
                L, n_per_long_edge=n_edge, n_theta=n_theta, mode="grid"
            )
            bc = compute_boundary_loss_2d(model, L, x1, x2, th)
            names = BC_COMPONENT_NAMES
        elif is_cartesian:
            n_phi = int(cfg.boundary.phi_samples_at_boundary)
            pts = sample_boundary_points_cartesian_3d(
                L, n_per_long_edge=n_edge,
                n_phi_grid=n_phi if n_phi % 2 == 1 else n_phi + 1,
                n_theta_grid=n_theta, mode="grid",
            )
            bc = compute_boundary_loss_cartesian_3d(model, L, *pts)
            names = BC_COMPONENT_NAMES_CART
        else:
            n_phi = int(cfg.boundary.phi_samples_at_boundary)
            pts = sample_boundary_points_3d(
                L, n_per_long_edge=n_edge, n_theta=n_theta, n_phi=n_phi,
                mode="grid",
            )
            bc = compute_boundary_loss_3d(model, L, *pts)
            names = BC_COMPONENT_NAMES_3D

    return {
        "total": float(bc.total.item()),
        "per_component": {
            name: float(getattr(bc, name).item()) for name in names
        },
    }


# ---------------------------------------------------------------------------
# û accuracy vs the exact characteristic-ODE reference (full volume)
# ---------------------------------------------------------------------------
def _eval_u_at_points(model, x, phi, theta, chunk: int = 50_000) -> np.ndarray:
    """û of a spherical-signature 3D model at scattered phase-space points."""
    x = np.asarray(x, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty(x.shape[0], dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, x.shape[0], chunk):
            hi = min(lo + chunk, x.shape[0])
            args = [
                torch.tensor(v, dtype=torch.float32)
                for v in (x[lo:hi, 0], x[lo:hi, 1], x[lo:hi, 2],
                          phi[lo:hi], theta[lo:hi])
            ]
            out[lo:hi] = model(*args)[0].detach().to(torch.float64).cpu().numpy()
    return out


def _eval_u_at_points_2d(model, x, theta, chunk: int = 50_000) -> np.ndarray:
    """û of a 2D model ``model(x1, x2, θ)`` at scattered points."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty(x.shape[0], dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, x.shape[0], chunk):
            hi = min(lo + chunk, x.shape[0])
            args = [
                torch.tensor(v, dtype=torch.float32)
                for v in (x[lo:hi, 0], x[lo:hi, 1], theta[lo:hi])
            ]
            out[lo:hi] = model(*args)[0].detach().to(torch.float64).cpu().numpy()
    return out


def _np(a):
    return a.numpy() if hasattr(a, "numpy") else np.asarray(a)


def _vs_reference_volume_stats(model, volume_ref: dict, is_3d: bool) -> dict:
    """û accuracy vs the exact ODE reference at a fixed Sobol volume sample.

    Gridless and per-point (no interpolation). Returns ``global`` error stats
    over the finite points plus a ``convergence`` self-check (L2 over the
    first half of the sample — a large gap means the sample is too small for 
    a stable estimate).
    """
    x = _np(volume_ref["x"])
    theta = _np(volume_ref["theta"])
    u_ref = np.asarray(_np(volume_ref["u"]), dtype=np.float64)
    if is_3d:
        phi = _np(volume_ref["phi"])
        u_pinn = _eval_u_at_points(model, x, phi, theta)
    else:
        u_pinn = _eval_u_at_points_2d(model, x, theta)

    err = u_pinn - u_ref
    finite = np.isfinite(err)
    n_total = int(err.size)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {"available": False, "reason": "reference/model non-finite at every point"}

    ef = err[finite]
    half = max(1, n_finite // 2)
    l2_full = float(np.sqrt(np.mean(ef ** 2)))
    l2_half = float(np.sqrt(np.mean(ef[:half] ** 2)))
    rel_gap = abs(l2_full - l2_half) / l2_full if l2_full > 0 else 0.0

    return {
        "available": True,
        "reference": "ode_direct",
        "n_points": n_total,
        "n_points_finite": n_finite,
        "nonfinite_fraction": float((n_total - n_finite) / n_total) if n_total else 0.0,
        "seed": int(volume_ref.get("seed", 0)),
        "phi_delta": float(volume_ref.get("phi_delta", 0.0)) if is_3d else None,
        "global": _err_stats(ef),
        "convergence": {
            "L2": l2_full,
            "L2_halfN": l2_half,
            "rel_gap_halfN": float(rel_gap),
            "converged": bool(rel_gap < 0.03),
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_run_metrics(
    *,
    run_name: str,
    cfg: TrainConfiguration,
    model: EscapeNet,
    history: TrainHistory,
    volume_reference: dict | None = None,
    wall_time_seconds: dict | None = None,
    extra_config: dict | None = None,
    resolved_seed: Optional[int] = None,
    resolved_device: Optional[str] = None,
    **_ignored,
) -> dict:
    """Aggregate the metrics of one trained PINN run into a dict.

    ``volume_reference`` is the Sobol point table from
    :func:`pinn_solver.evaluation.ode_reference.get_or_build_ode_reference_points_full`
    (3D) or the 2D counterpart. Pass ``None`` to skip the accuracy block.
    ``model`` is read-only: temporarily moved to CPU + eval, then restored.
    ``resolved_device`` is the runtime-resolved device string.
    """
    is_3d = cfg.physical_size.dim == 3
    is_cartesian = is_3d and cfg.parametrization == "cartesian"

    with _on_cpu_eval(model) as cpu_model:
        pde_stats = _pde_residual_stats(cpu_model, cfg)
        bc_stats = _bc_residual_stats(cpu_model, cfg)
        if volume_reference is not None:
            volume_model = cpu_model.spherical_view() if is_cartesian else cpu_model
            volume_stats = _vs_reference_volume_stats(volume_model, volume_reference, is_3d)
        else:
            volume_stats = {"available": False, "reason": "no volume_reference supplied"}

    adam_cfg = cfg.optimizers.adam
    lbfgs_cfg = cfg.optimizers.lbfgs
    config = {
        "physical_size": [float(v) for v in cfg.physical_size.as_tuple()],
        "dim": int(cfg.physical_size.dim),
        "parametrization": str(cfg.parametrization),
        "n_collocation": int(adam_cfg.collocation_samples),
        "n_bc_per_edge": int(cfg.boundary.physical_boundary_samples),
        "n_bc_theta": int(cfg.boundary.theta_samples_at_boundary),
        "n_bc_phi": (
            None
            if cfg.boundary.phi_samples_at_boundary is None
            else int(cfg.boundary.phi_samples_at_boundary)
        ),
        "lambda_pde": float(cfg.loss.lambda_pde),
        "lambda_bc": float(cfg.loss.lambda_bc),
        "lambda_data": float(cfg.loss.lambda_data),
        "lambda_circle": float(cfg.loss.lambda_circle),
        "pole_safe_residual": bool(cfg.loss.pole_safe_residual),
        "adam_epochs": int(adam_cfg.epochs),
        "adam_lr": float(adam_cfg.lr),
        "lbfgs_steps": int(lbfgs_cfg.steps) if lbfgs_cfg is not None else 0,
        "device": resolved_device if resolved_device is not None else cfg.device,
        "network": _network_shape(model),
        "seed": cfg.seed,
        "resolved_seed": (
            int(resolved_seed) if resolved_seed is not None else cfg.seed
        ),
        "slowness": cfg.slowness.model_dump(),
    }
    if extra_config:
        config["extra"] = dict(extra_config)

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": run_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": _gpu_name(),
        },
        "config": config,
        "wall_time_seconds": dict(wall_time_seconds) if wall_time_seconds else None,
        "loss": _final_loss_section(history, adam_epochs=int(adam_cfg.epochs)),
        "pde_residual": pde_stats,
        "pde_nan_fraction": pde_stats.get("nonfinite_fraction"),
        "bc_residual": bc_stats,
        "vs_reference_volume": volume_stats,
    }


def save_run_metrics(metrics: dict, path: str) -> None:
    """Write a metrics dict to ``path`` as pretty-printed, key-sorted JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
