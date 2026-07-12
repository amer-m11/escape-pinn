"""Disk cache for the validation mesh (not the training supervision mesh)."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import types
from pathlib import Path
from typing import Any

import numpy as np


def mesh_cache_key(
    slowness_cfg, n_grid: int, n_theta: int, n_phi: int | None = None
) -> str:
    """Deterministic short hash of `(slowness config, grid dims)`.

    Uses ``slowness_cfg.model_dump(mode="json")`` so any field added to a
    slowness Pydantic model later automatically invalidates older cache
    entries (the new field shows up in the payload).
    """
    payload = {
        "slowness": slowness_cfg.model_dump(mode="json"),
        "n_grid": int(n_grid),
        "n_theta": int(n_theta),
    }
    if n_phi is not None:
        payload["n_phi"] = int(n_phi)
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _cached_grid_from_npz(path: Path) -> Any:
    """Reconstruct a minimal grid-like object from a cached `.npz`.

    Detects dim by checking whether ``x3_coords`` is present. The returned
    ``SimpleNamespace`` quacks like ``PhaseSpaceGrid2D`` (no ``x3_coords``)
    or ``PhaseSpaceGrid3D`` (full 3D field set) for the read-only callers.
    """
    with np.load(path) as data:
        x1_coords = data["x1_coords"]
        x2_coords = data["x2_coords"]
        theta_coords = data["theta_coords"]
        is_3d = "x3_coords" in data.files
        common = dict(
            N_x1=int(x1_coords.size),
            N_x2=int(x2_coords.size),
            N_theta=int(theta_coords.size),
            L_x1=float(x1_coords[-1]),
            L_x2=float(x2_coords[-1]),
            x1_coords=x1_coords,
            x2_coords=x2_coords,
            theta_coords=theta_coords,
            u=data["u"],
            sigma=data["sigma"],
            y1=data["y1"],
            y2=data["y2"],
            theta_exit=data["theta_exit"],
            status=data["status"],
        )
        if not is_3d:
            return types.SimpleNamespace(**common)
        x3_coords = data["x3_coords"]
        phi_coords = data["phi_coords"]
        return types.SimpleNamespace(
            **common,
            N_x3=int(x3_coords.size),
            N_phi=int(phi_coords.size),
            L_x3=float(x3_coords[-1]),
            x3_coords=x3_coords,
            phi_coords=phi_coords,
            y3=data["y3"],
            phi_exit=data["phi_exit"],
        )


def _write_cache_atomically(path: Path, grid) -> None:
    """Persist a solved grid to `path` via unique-temp + atomic rename.

    The temp filename embeds the writer's PID and 8 random hex chars so
    concurrent writers never touch the same temp file. ``os.replace`` is
    atomic.

    Dispatches on ``hasattr(grid, "x3_coords")`` to serialize either a 2D
    or 3D grid. The 3D path additionally writes ``x3_coords, phi_coords,
    y3, phi_exit``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.stem}.{os.getpid()}.{secrets.token_hex(4)}.partial.npz"
    )
    arrays = dict(
        x1_coords=np.asarray(grid.x1_coords),
        x2_coords=np.asarray(grid.x2_coords),
        theta_coords=np.asarray(grid.theta_coords),
        u=np.asarray(grid.u),
        sigma=np.asarray(grid.sigma),
        y1=np.asarray(grid.y1),
        y2=np.asarray(grid.y2),
        theta_exit=np.asarray(grid.theta_exit),
        status=np.asarray(grid.status),
    )
    if hasattr(grid, "x3_coords"):
        arrays.update(
            x3_coords=np.asarray(grid.x3_coords),
            phi_coords=np.asarray(grid.phi_coords),
            y3=np.asarray(grid.y3),
            phi_exit=np.asarray(grid.phi_exit),
        )
    np.savez_compressed(tmp, **arrays)  # type: ignore
    os.replace(tmp, path)
