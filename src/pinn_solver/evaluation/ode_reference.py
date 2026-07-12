"""Exact escape-quantity reference by characteristic-ODE ray integration.

The Fomel-Sethian escape quantities at a phase-space point ``(x, p)`` are the
result of following the characteristic flow ``R = (∇_p H, -∇_x H) = (p, n∇n)``
from that point until the ray exits the box ∂D, accumulating the per-quantity
source ``s`` (``∇₀q·R + s_q = 0`` => ``dq/dτ = -s_q`` along the flow, with the
boundary value ``q=0/x_exit`` at exit). For the Eikonal Hamiltonian:

    dx/dτ = p,   dp/dτ = n(x) ∇n(x),   |p| = n  (preserved on H=0)
    û  = ∫ n² dτ      (escape value;  BC û=0 at exit)
    σ̂  = ∫ 1  dτ = τ_exit   (escape parameter)
    ŷ  = x(τ_exit)          (exit position)
    d̂  = p(τ_exit)/|p|      (exit direction)

Disclaimer: the derivation above and its implementation in this module were
generated and implemented by Claude (Anthropic) and subsequently tested.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch


def integrate_escape_rays(
    x0: torch.Tensor,  # (N, 3) interior start points
    d0: torch.Tensor,  # (N, 3) unit start directions
    slowness_fn,  # (N,3)->(N,)
    slowness_grad_fn,  # (N,3)->(N,3)
    box: tuple[float, float, float],
    dtau: float = 5.0e-3,
    max_tau: float | None = None,
) -> dict:
    """Integrate the escape characteristic for a batch of rays to ∂D.

    Returns a dict of (N,) / (N,3) tensors: ``u`` (escape value), ``sigma``
    (escape parameter τ_exit), ``y_exit`` (3), ``d_exit`` (3, unit), and
    ``exited`` (bool — False => ray never left the box within ``max_tau`` or was
    a straggler dropped by the early-stop, marked NaN, mirroring the mesh's
    unaccepted nodes).
    """
    L = torch.tensor(box, dtype=torch.float64)
    diag = float(torch.linalg.norm(L))
    if max_tau is None:
        max_tau = 2.5 * diag
    n_steps = int(math.ceil(max_tau / dtau))

    x = x0.to(torch.float64).clone()
    n0 = slowness_fn(x)
    p = d0.to(torch.float64) * n0[:, None]  # |p| = n at the start
    N = x.shape[0]
    u = torch.zeros(N, dtype=torch.float64)
    tau = torch.zeros(N, dtype=torch.float64)
    exited = torch.zeros(N, dtype=torch.bool)
    y_exit = x.clone()
    d_exit = d0.to(torch.float64).clone()

    def rhs(xx, pp):
        n = slowness_fn(xx)
        g = slowness_grad_fn(xx)
        return pp, n[:, None] * g, n * n  # dx, dp, du

    def _out_of_box(xx):
        return (xx < 0.0).any(dim=1) | (xx > L).any(dim=1)

    with torch.no_grad():
        for _ in range(n_steps):
            idx = (~exited).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            xa, pa, ua, taua = x[idx], p[idx], u[idx], tau[idx]
            k1x, k1p, k1u = rhs(xa, pa)
            k2x, k2p, k2u = rhs(xa + 0.5 * dtau * k1x, pa + 0.5 * dtau * k1p)
            k3x, k3p, k3u = rhs(xa + 0.5 * dtau * k2x, pa + 0.5 * dtau * k2p)
            k4x, k4p, k4u = rhs(xa + dtau * k3x, pa + dtau * k3p)
            xn = xa + (dtau / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            pn = pa + (dtau / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
            un = ua + (dtau / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)

            out = _out_of_box(xn)  # crossed ∂D this step
            # Linear crossing fraction α∈[0,1] to the first face hit (inf where
            # an axis is not crossed -> clamp to 1, harmless for the full step).
            denom = xn - xa
            big = torch.full_like(xa, math.inf)
            alpha_hi = torch.where(xn > L, (L - xa) / denom, big)
            alpha_lo = torch.where(xn < 0.0, (0.0 - xa) / denom, big)
            alpha = torch.minimum(alpha_hi, alpha_lo).min(dim=1).values.clamp(0.0, 1.0)
            a = alpha[:, None]
            oc = out[:, None]
            x[idx] = torch.where(oc, xa + a * (xn - xa), xn)
            p[idx] = torch.where(oc, pa + a * (pn - pa), pn)
            u[idx] = torch.where(out, ua + alpha * (un - ua), un)
            tau[idx] = torch.where(out, taua + alpha * dtau, taua + dtau)
            if bool(out.any()):
                ex = idx[out]
                pe = p[ex]
                y_exit[ex] = x[ex]
                d_exit[ex] = pe / torch.linalg.norm(pe, dim=1, keepdim=True).clamp_min(
                    1e-30
                )
                exited[ex] = True

    nan = torch.full((N,), float("nan"), dtype=torch.float64)
    nan_d = torch.full(
        (N, x.shape[1]), float("nan"), dtype=torch.float64
    )  # dim-agnostic (2D/3D)
    return {
        "u": torch.where(exited, u, nan),
        "sigma": torch.where(exited, tau, nan),
        "y_exit": torch.where(exited[:, None], y_exit, nan_d),
        "d_exit": torch.where(exited[:, None], d_exit, nan_d),
        "exited": exited,
    }


def escape_reference_grid_3d(
    cfg,
    n_grid: int,
    n_theta: int,
    n_phi: int,
    dtau: float = 5.0e-3,
    n_x3: int = 1,
):
    """Exact escape ``û`` on the escape-quantity grid.

    Returns a mesh-grid-like ``SimpleNamespace`` (``u`` shape
    ``(n_grid, n_grid, n_x3, n_phi, n_theta)`` + ``*_coords``) so
    ``run._build_mesh_reference_fn_3d`` serves it through the same
    φ-interpolating loader as a real mesh. ``n_x3=1`` (default) computes only the
    mid-x₃ = L₃/2 slice (what the metric scores); ``n_x3>1`` computes the full
    volume (x₃ = linspace(0, L₃, n_x3)) so the PINN can be tested across the
    whole space, not just one hyperplane.
    """
    L1, L2, L3 = cfg.physical_size.as_tuple()
    slowness_fn = cfg.slowness.get_slowness()
    slowness_grad_fn = cfg.slowness.get_slowness_gradient()
    x1c = np.linspace(0.0, L1, n_grid)
    x2c = np.linspace(0.0, L2, n_grid)
    x3c = np.array([0.5 * L3]) if n_x3 <= 1 else np.linspace(0.0, L3, n_x3)
    n3 = len(x3c)
    # mesh-style cell-centred φ on [0,π] and θ on [0,2π) — what the loader/metric expect.
    phis = np.array([(k + 0.5) * math.pi / n_phi for k in range(n_phi)])
    thetas = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)

    X1, X2, X3 = np.meshgrid(x1c, x2c, x3c, indexing="ij")
    G = n_grid * n_grid * n3
    spatial = torch.stack(
        [
            torch.tensor(X1.reshape(-1), dtype=torch.float64),
            torch.tensor(X2.reshape(-1), dtype=torch.float64),
            torch.tensor(X3.reshape(-1), dtype=torch.float64),
        ],
        dim=1,
    )  # (G, 3), G = n_grid²·n3
    # All (φ, θ) directions, ip-major then it (matches the reshape below).
    dirs = torch.tensor(
        [
            [math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), math.cos(phi)]
            for phi in phis
            for th in thetas
        ],
        dtype=torch.float64,
    )  # (A, 3), A = n_phi * n_theta

    # One big batch: every (angle, spatial) ray integrated together (the per-step
    # torch overhead amortizes over the full stack instead of per-slice).
    A = dirs.shape[0]
    x0 = spatial.repeat(A, 1)  # (A·G, 3): spatial block per angle
    d0 = dirs.repeat_interleave(G, dim=0)  # (A·G, 3): each angle ×G
    res = integrate_escape_rays(
        x0, d0, slowness_fn, slowness_grad_fn, (L1, L2, L3), dtau=dtau
    )

    ye, de = res["y_exit"], res["d_exit"]
    theta_x = torch.atan2(de[:, 1], de[:, 0])
    phi_x = torch.acos(de[:, 2].clamp(-1.0, 1.0))

    def _grid(flat):  # (A·G,) -> (n_grid, n_grid, n3, n_phi, n_theta)
        arr = flat.reshape(
            n_phi, n_theta, n_grid, n_grid, n3
        ).numpy()  # (ip,it,i1,i2,i3)
        return np.transpose(arr, (2, 3, 4, 0, 1))

    fields = {
        "u": _grid(res["u"]),
        "sigma": _grid(res["sigma"]),
        "y1": _grid(ye[:, 0]),
        "y2": _grid(ye[:, 1]),
        "y3": _grid(ye[:, 2]),
        "theta_exit": _grid(theta_x),
        "phi_exit": _grid(phi_x),
        "status": _grid(res["exited"].to(torch.float64)),
    }

    return SimpleNamespace(
        **fields,
        x1_coords=x1c,
        x2_coords=x2c,
        x3_coords=x3c,
        phi_coords=phis,
        theta_coords=thetas,
        L_x1=L1,
        L_x2=L2,
        L_x3=L3,
        N_x1=n_grid,
        N_x2=n_grid,
        N_x3=n3,
        N_phi=n_phi,
        N_theta=n_theta,
    )


def sample_phase_points(cfg, n_points: int, seed: int, phi_delta: float | None = None):
    """Low-discrepancy (Sobol) sample of ``n_points`` over the trained 5D phase
    space: ``x ∈ box``, ``φ ∈ [δ, π−δ]`` (the trained safe band), ``θ ∈ [0,2π)``.

    Deterministic given ``(seed, n_points, δ)`` → the point set, the cached
    reference table, and therefore the cross-run comparison are byte-identical
    across runs (apples-to-apples ranking). Sobol (not i.i.d. uniform) gives an
    L2 estimate that converges ~1/N and does not *alias* the angular caustic
    kink the way a coarse structured θ/φ grid does.

    Cartesian and spherical runs use the *same* canonical band δ
    (``cfg.boundary.resolved_phi_delta()``, π/8 by default) so both
    parametrisations are scored on the identical region (fair ranking; the
    Cartesian polar caps it can additionally represent are a separate question).

    Returns a dict of float64 tensors: ``x`` (N,3), ``d`` (N,3 unit), ``phi``
    (N,), ``theta`` (N,), plus the scalar ``phi_delta``.
    """
    L1, L2, L3 = cfg.physical_size.as_tuple()
    delta = float(
        phi_delta if phi_delta is not None else cfg.boundary.resolved_phi_delta()
    )
    eng = torch.quasirandom.SobolEngine(dimension=5, scramble=True, seed=int(seed))
    u = eng.draw(int(n_points)).to(torch.float64)  # (N,5) in [0,1)
    x1 = L1 * u[:, 0]
    x2 = L2 * u[:, 1]
    x3 = L3 * u[:, 2]
    phi = delta + (math.pi - 2.0 * delta) * u[:, 3]
    theta = 2.0 * math.pi * u[:, 4]
    sp, cp = torch.sin(phi), torch.cos(phi)
    d = torch.stack([sp * torch.cos(theta), sp * torch.sin(theta), cp], dim=1)
    x = torch.stack([x1, x2, x3], dim=1)
    return {"x": x, "d": d, "phi": phi, "theta": theta, "phi_delta": delta}


def escape_reference_points(
    cfg,
    n_points: int,
    seed: int,
    dtau: float = 5.0e-3,
    phi_delta: float | None = None,
    full: bool = False,
) -> dict:
    """Exact escape quantities at a fixed Sobol phase-space sample (full volume,
    all φ in band, all θ) by **direct per-point characteristic integration** — no
    grid, no spatial/angular interpolation, exact at the caustic.

    This is the reference for the full-volume ``vs_reference`` metric block: it
    replaces interpolating a cached mesh/ODE grid (which re-smooths the C⁰ kink
    and is mid-x₃ only). Works for any slowness, constant medium included
    (straight rays → exact distance to ∂D). Returns the
    :func:`sample_phase_points` dict augmented with ``u`` (N,) and ``exited``
    (N, bool); non-exited rays are NaN in ``u`` and masked downstream.

    ``full=True`` additionally returns the OTHER escape channels the integrator
    already computes — ``sigma`` (N), ``y_exit`` (N,3), ``d_exit`` (N,3 unit),
    ``theta_exit``/``phi_exit`` (N) — for per-channel accuracy analysis (σ̂/ŷ/exit
    direction vs the exact reference, not just û).
    """
    pts = sample_phase_points(cfg, n_points, seed, phi_delta)
    n_fn = cfg.slowness.get_slowness()
    g_fn = cfg.slowness.get_slowness_gradient()
    L = cfg.physical_size.as_tuple()
    res = integrate_escape_rays(pts["x"], pts["d"], n_fn, g_fn, L, dtau=dtau)
    out = {
        **pts,
        "u": res["u"],
        "exited": res["exited"],
        "seed": int(seed),
        "n_points": int(n_points),
        "dtau": float(dtau),
    }
    if full:
        de = res["d_exit"]
        out.update(
            sigma=res["sigma"],
            y_exit=res["y_exit"],
            d_exit=de,
            theta_exit=torch.atan2(de[:, 1], de[:, 0]),
            phi_exit=torch.acos(de[:, 2].clamp(-1.0, 1.0)),
        )
    return out


def sample_phase_points_2d(cfg, n_points: int, seed: int) -> dict:
    """Sobol sample over the **2D** phase space: ``x ∈ box``, ``θ ∈ [0, 2π)``.

    The 2D analogue of :func:`sample_phase_points` — there is no φ pole band (the
    2D residual has no ``1/sin φ`` singularity), so the whole launch circle is
    sampled. Deterministic given ``(seed, n_points)``. Returns float64 tensors
    ``x`` (N,2), ``theta`` (N,), ``d`` (N,2 unit, = (cos θ, sin θ)).
    """
    L1, L2 = cfg.physical_size.as_tuple()
    eng = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=int(seed))
    u = eng.draw(int(n_points)).to(torch.float64)  # (N,3) in [0,1)
    x1 = L1 * u[:, 0]
    x2 = L2 * u[:, 1]
    theta = 2.0 * math.pi * u[:, 2]
    d = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    x = torch.stack([x1, x2], dim=1)
    return {"x": x, "theta": theta, "d": d}


def escape_reference_points_2d(
    cfg, n_points: int, seed: int, dtau: float = 5.0e-3, full: bool = False
) -> dict:
    """Exact 2D escape quantities at a fixed Sobol ``(x, θ)`` sample by direct
    per-point characteristic integration — the gold-standard reference for the
    2D full-volume ``vs_reference_volume`` block (mirrors
    :func:`escape_reference_points`). The integrator is dimension-agnostic, so
    this reuses it on 2-vectors. ``full=True`` additionally returns ``sigma``,
    ``y_exit`` (N,2), ``d_exit`` (N,2 unit) and ``theta_exit`` for per-channel
    accuracy. Works for any 2D slowness (constant → straight rays = exact
    distance to ∂D)."""
    pts = sample_phase_points_2d(cfg, n_points, seed)
    n_fn = cfg.slowness.get_slowness()
    g_fn = cfg.slowness.get_slowness_gradient()
    L = cfg.physical_size.as_tuple()
    res = integrate_escape_rays(pts["x"], pts["d"], n_fn, g_fn, L, dtau=dtau)
    out = {
        **pts,
        "u": res["u"],
        "exited": res["exited"],
        "seed": int(seed),
        "n_points": int(n_points),
        "dtau": float(dtau),
    }
    if full:
        de = res["d_exit"]
        out.update(
            sigma=res["sigma"],
            y_exit=res["y_exit"],
            d_exit=de,
            theta_exit=torch.atan2(de[:, 1], de[:, 0]),
        )
    return out


def get_or_build_ode_reference_points_2d(
    cfg,
    n_points: int,
    seed: int,
    cache_dir: str,
    dtau: float = 5.0e-3,
    full: bool = False,
) -> dict:
    """Cached :func:`escape_reference_points_2d`. Distinct ``ode_points_2d_``
    (û-only) / ``ode_points_2d_full_`` (all-channel) filename prefix so it never
    collides with the 3D point caches. Built once per (slowness, N, seed)."""
    import os
    from pathlib import Path

    key = _points_cache_key(cfg, n_points, seed, dtau, None)
    prefix = "ode_points_2d_full_" if full else "ode_points_2d_"
    path = Path(cache_dir) / f"{prefix}{key}.npz"
    base_keys = ("x", "theta", "d", "u", "exited")
    full_keys = ("sigma", "y_exit", "d_exit", "theta_exit")
    if path.exists():
        z = np.load(path)
        out = {k: torch.tensor(z[k]) for k in base_keys}
        out["exited"] = out["exited"].bool()
        for s in ("seed", "n_points", "dtau"):
            out[s] = z[s].item()
        if full:
            out.update({k: torch.tensor(z[k]) for k in full_keys})
        return out
    ref = escape_reference_points_2d(cfg, n_points, seed, dtau=dtau, full=full)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    save = {k: ref[k].numpy() for k in base_keys}
    if full:
        save.update({k: ref[k].numpy() for k in full_keys})
    np.savez_compressed(
        tmp, **save, seed=ref["seed"], n_points=ref["n_points"], dtau=ref["dtau"]
    )
    os.replace(tmp, path)
    return ref


def _points_cache_key(cfg, n_points, seed, dtau, phi_delta) -> str:
    import hashlib

    from .mesh_cache import mesh_cache_key

    base = mesh_cache_key(cfg.slowness, n_points, seed)  # folds slowness + N + seed
    payload = f"{base}|{cfg.physical_size.as_tuple()}|dtau={dtau}|delta={phi_delta}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_or_build_ode_reference_points(
    cfg,
    n_points: int,
    seed: int,
    cache_dir: str,
    dtau: float = 5.0e-3,
    phi_delta: float | None = None,
) -> dict:
    """Cached :func:`escape_reference_points`. Keyed on slowness + N + seed +
    domain + dtau + δ with an ``ode_points_`` filename prefix (never collides
    with the grid caches). Built once per (slowness, N, seed); every later run
    reads the table and only forwards its own model → the comparison costs a
    batched network eval, not an ODE re-solve.
    """
    import os
    from pathlib import Path

    delta = float(
        phi_delta if phi_delta is not None else cfg.boundary.resolved_phi_delta()
    )
    key = _points_cache_key(cfg, n_points, seed, dtau, delta)
    path = Path(cache_dir) / f"ode_points_{key}.npz"
    if path.exists():
        z = np.load(path)
        return {
            "x": torch.tensor(z["x"], dtype=torch.float64),
            "d": torch.tensor(z["d"], dtype=torch.float64),
            "phi": torch.tensor(z["phi"], dtype=torch.float64),
            "theta": torch.tensor(z["theta"], dtype=torch.float64),
            "u": torch.tensor(z["u"], dtype=torch.float64),
            "exited": torch.tensor(z["exited"], dtype=torch.bool),
            "phi_delta": float(z["phi_delta"]),
            "seed": int(z["seed"]),
            "n_points": int(z["n_points"]),
            "dtau": float(z["dtau"]),
        }
    ref = escape_reference_points(cfg, n_points, seed, dtau=dtau, phi_delta=delta)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        x=ref["x"].numpy(),
        d=ref["d"].numpy(),
        phi=ref["phi"].numpy(),
        theta=ref["theta"].numpy(),
        u=ref["u"].numpy(),
        exited=ref["exited"].numpy(),
        phi_delta=ref["phi_delta"],
        seed=ref["seed"],
        n_points=ref["n_points"],
        dtau=ref["dtau"],
    )
    os.replace(tmp, path)
    return ref


def get_or_build_ode_reference_points_full(
    cfg,
    n_points: int,
    seed: int,
    cache_dir: str,
    dtau: float = 5.0e-3,
    phi_delta: float | None = None,
) -> dict:
    """Cached ALL-CHANNEL exact-ODE point reference (û + σ̂ + ŷ + exit direction),
    for per-channel accuracy analysis. Same Sobol set as the û-only table but a
    distinct ``ode_points_full_`` prefix so it never collides. Built once per
    (slowness, N, seed)."""
    import os
    from pathlib import Path

    delta = float(
        phi_delta if phi_delta is not None else cfg.boundary.resolved_phi_delta()
    )
    key = _points_cache_key(cfg, n_points, seed, dtau, delta)
    path = Path(cache_dir) / f"ode_points_full_{key}.npz"
    keys = (
        "x",
        "d",
        "phi",
        "theta",
        "u",
        "sigma",
        "y_exit",
        "d_exit",
        "theta_exit",
        "phi_exit",
        "exited",
    )
    if path.exists():
        z = np.load(path)
        out = {k: torch.tensor(z[k]) for k in keys}
        out["exited"] = out["exited"].bool()
        for s in ("phi_delta", "seed", "n_points", "dtau"):
            out[s] = z[s].item()
        return out
    ref = escape_reference_points(
        cfg, n_points, seed, dtau=dtau, phi_delta=delta, full=True
    )
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        **{k: ref[k].numpy() for k in keys},
        phi_delta=ref["phi_delta"],
        seed=ref["seed"],
        n_points=ref["n_points"],
        dtau=ref["dtau"],
    )
    os.replace(tmp, path)
    return ref


def get_or_build_ode_reference(
    cfg, n_grid, n_theta, n_phi, cache_dir, dtau=5.0e-3, n_x3=1
):
    """Cached exact-ODE reference grid, keyed like the mesh cache but with an
    ``ode_`` filename prefix so it never collides with a mesh solve. ``n_x3>1``
    builds the full volume and gets an ``_x3{n_x3}`` filename tag so it never
    collides with the mid-x₃ (``n_x3=1``) cache.

    Returned object is loader-compatible with ``run._build_mesh_reference_fn_3d``.
    """
    import os
    from pathlib import Path

    from .mesh_cache import _cached_grid_from_npz, mesh_cache_key

    key = mesh_cache_key(cfg.slowness, n_grid, n_theta, n_phi=n_phi)
    tag = "" if n_x3 <= 1 else f"_x3{int(n_x3)}"
    path = Path(cache_dir) / f"ode_{key}{tag}.npz"
    if path.exists():
        return _cached_grid_from_npz(path)
    grid = escape_reference_grid_3d(cfg, n_grid, n_theta, n_phi, dtau=dtau, n_x3=n_x3)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        u=grid.u,
        sigma=grid.sigma,
        y1=grid.y1,
        y2=grid.y2,
        y3=grid.y3,
        theta_exit=grid.theta_exit,
        phi_exit=grid.phi_exit,
        status=grid.status,
        x1_coords=grid.x1_coords,
        x2_coords=grid.x2_coords,
        x3_coords=grid.x3_coords,
        phi_coords=grid.phi_coords,
        theta_coords=grid.theta_coords,
    )
    os.replace(tmp, path)
    return _cached_grid_from_npz(path)
