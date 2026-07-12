"""PDE residual computation for the 3D Eikonal escape equations."""

from typing import Callable

import torch

from ..networks import EscapeNet3D


def compute_pde_residual_3d(
    model: EscapeNet3D,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    slowness_fn: Callable[[torch.Tensor], torch.Tensor],
    slowness_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    pole_safe: bool = False,
    cond: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate the nine escape-equation residuals at 5D phase-space points.

    Computes the derivatives using a per-sample Jacobian via
    ``torch.func.vmap(jacrev(...))``, which is functionally equivalent to 45
    sequential ``torch.autograd.grad`` calls.

    Parameters
    ----------
    model : EscapeNet3D
    x1, x2, x3, phi, theta : torch.Tensor
        1-D tensors (length N) of phase-space coordinates. ``phi`` must lie
        strictly inside ``(0, π)``. The sampler is responsible for
        excluding a small δ-band around each pole. Coordinates do not need
        ``requires_grad=True``. The per-sample Jacobian uses ``torch.func``.
    slowness_fn : callable
        ``n(x) -> (N,)`` slowness values, where ``x`` is the ``(N, 3)``
        stack of spatial coordinates.
    slowness_grad_fn : callable, optional
        ``∇n(x) -> (N, 3)`` analytic spatial gradient. If omitted, autograd
        computes it through ``slowness_fn``.
    pole_safe : bool
        When ``True``, return the whole residual (advection, ``g_θ`` term, and
        source) multiplied through by ``sin φ``: the ``g_θ = (…)/sin φ`` term's
        division then cancels, and the sources become ``sin φ·n²`` / ``sin φ``.
        The result is finite at the poles φ ∈ {0, π} and has the same zero set
        as the standard residual, so the sampler can include them and the PINN
        can cover near-polar ray directions. ``False`` (default) returns the
        standard residual.

    Returns
    -------
    res_u, res_sigma, res_y1, res_y2, res_y3,
    res_cos_theta, res_sin_theta, res_cos_phi, res_sin_phi : torch.Tensor
        Residuals of the nine escape equations at every input point.
    """

    def _stacked_outputs(
        s_x1: torch.Tensor,
        s_x2: torch.Tensor,
        s_x3: torch.Tensor,
        s_ph: torch.Tensor,
        s_th: torch.Tensor,
        *rest: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the network on a single point, returning a ``(9,)`` vector.

        ``*rest`` carries the optional per-sample conditioning vector. ``jacrev``
        covers only the five phase coords (``argnums=(0..4)``), never ``cond``.
        """
        return torch.stack(model(s_x1, s_x2, s_x3, s_ph, s_th, *rest))

    _vmap_args = (
        (x1, x2, x3, phi, theta) if cond is None else (x1, x2, x3, phi, theta, cond)
    )
    jac_x1, jac_x2, jac_x3, jac_ph, jac_th = torch.func.vmap(
        torch.func.jacrev(_stacked_outputs, argnums=(0, 1, 2, 3, 4))
    )(*_vmap_args)

    # fmt: off
    # Each ``jac_*`` has shape ``(N, 9)``. Unbinding the last axis splits it
    # into nine per-output (N,) tensors in canonical order.
    du_dx1, ds_dx1, dy1_dx1, dy2_dx1, dy3_dx1, dcth_dx1, dsth_dx1, dcph_dx1, dsph_dx1 = jac_x1.unbind(dim=-1)
    du_dx2, ds_dx2, dy1_dx2, dy2_dx2, dy3_dx2, dcth_dx2, dsth_dx2, dcph_dx2, dsph_dx2 = jac_x2.unbind(dim=-1)
    du_dx3, ds_dx3, dy1_dx3, dy2_dx3, dy3_dx3, dcth_dx3, dsth_dx3, dcph_dx3, dsph_dx3 = jac_x3.unbind(dim=-1)
    du_dph, ds_dph, dy1_dph, dy2_dph, dy3_dph, dcth_dph, dsth_dph, dcph_dph, dsph_dph = jac_ph.unbind(dim=-1)
    du_dth, ds_dth, dy1_dth, dy2_dth, dy3_dth, dcth_dth, dsth_dth, dcph_dth, dsph_dth = jac_th.unbind(dim=-1)
    # fmt: on

    # Slowness field
    # --------------
    # The ``+ 0.0 * x_n.sum(-1)`` term forces a graph connection so
    # disconnected slowness functions (i.e. not dependent on input)
    # return zero gradients instead of raising.
    x_stack = torch.stack((x1, x2, x3), dim=-1)
    if slowness_grad_fn is None:
        x_n = x_stack.detach().requires_grad_(True)
        n = slowness_fn(x_n) + 0.0 * x_n.sum(dim=-1)
        (dn_dx,) = torch.autograd.grad(n.sum(), (x_n,), create_graph=False)
        n = n.detach()
    else:
        n = slowness_fn(x_stack)
        dn_dx = slowness_grad_fn(x_stack)
    dn_dx1, dn_dx2, dn_dx3 = dn_dx.unbind(dim=-1)

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    cos_ph = torch.cos(phi)
    sin_ph = torch.sin(phi)

    R_x1 = n * sin_ph * cos_th
    R_x2 = n * sin_ph * sin_th
    R_x3 = n * cos_ph

    g_phi = cos_ph * cos_th * dn_dx1 + cos_ph * sin_th * dn_dx2 - sin_ph * dn_dx3

    n2 = n * n

    if pole_safe:
        # Residual × sin φ: the θ-term's division cancels, so nothing
        # diverges at φ → 0/π. g_θ_num = sin φ · g_θ (no division performed).
        g_theta_num = cos_th * dn_dx2 - sin_th * dn_dx1

        def _advect(d_dx1, d_dx2, d_dx3, d_dph, d_dth):
            return (
                sin_ph * (d_dx1 * R_x1 + d_dx2 * R_x2 + d_dx3 * R_x3 + d_dph * g_phi)
                + d_dth * g_theta_num
            )

        u_source = sin_ph * n2
        sigma_source = sin_ph
    else:
        # Standard residual. g_θ = (cos θ ∂₂n − sin θ ∂₁n) / sin φ diverges at
        # φ ∈ {0, π}. The sampler keeps φ strictly inside (δ, π − δ).
        g_theta = (cos_th * dn_dx2 - sin_th * dn_dx1) / sin_ph

        def _advect(d_dx1, d_dx2, d_dx3, d_dph, d_dth):
            return (
                d_dx1 * R_x1
                + d_dx2 * R_x2
                + d_dx3 * R_x3
                + d_dph * g_phi
                + d_dth * g_theta
            )

        u_source = n2
        sigma_source = 1.0

    res_u = _advect(du_dx1, du_dx2, du_dx3, du_dph, du_dth) + u_source
    res_sigma = _advect(ds_dx1, ds_dx2, ds_dx3, ds_dph, ds_dth) + sigma_source
    res_y1 = _advect(dy1_dx1, dy1_dx2, dy1_dx3, dy1_dph, dy1_dth)
    res_y2 = _advect(dy2_dx1, dy2_dx2, dy2_dx3, dy2_dph, dy2_dth)
    res_y3 = _advect(dy3_dx1, dy3_dx2, dy3_dx3, dy3_dph, dy3_dth)
    res_cos_theta = _advect(dcth_dx1, dcth_dx2, dcth_dx3, dcth_dph, dcth_dth)
    res_sin_theta = _advect(dsth_dx1, dsth_dx2, dsth_dx3, dsth_dph, dsth_dth)
    res_cos_phi = _advect(dcph_dx1, dcph_dx2, dcph_dx3, dcph_dph, dcph_dth)
    res_sin_phi = _advect(dsph_dx1, dsph_dx2, dsph_dx3, dsph_dph, dsph_dth)

    return (
        res_u,
        res_sigma,
        res_y1,
        res_y2,
        res_y3,
        res_cos_theta,
        res_sin_theta,
        res_cos_phi,
        res_sin_phi,
    )


def consistency_residual_3d(
    model: EscapeNet3D,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    slowness_fn: Callable[[torch.Tensor], torch.Tensor],
    slowness_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    dsigma: float,
) -> tuple[torch.Tensor, ...]:
    """Finite difference characteristic-consistency residuals.

    Mirrors the 2D version (see ``consistency_residual_2d``).

    φ is kept in the trained band by the sampler, so the ``1/sin φ``
    factor in ``g_θ`` stays bounded.
    """
    with torch.no_grad():
        x = torch.stack((x1, x2, x3), dim=-1)
        if slowness_grad_fn is None:
            xg = x.detach().requires_grad_(True)
            n = slowness_fn(xg) + 0.0 * xg.sum(dim=-1)
            (dn,) = torch.autograd.grad(n.sum(), (xg,), create_graph=False)
            n = n.detach()
        else:
            n = slowness_fn(x)
            dn = slowness_grad_fn(x)
        dn1, dn2, dn3 = dn.unbind(dim=-1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)
        R_x1 = n * sin_p * cos_t
        R_x2 = n * sin_p * sin_t
        R_x3 = n * cos_p
        g_phi = cos_p * cos_t * dn1 + cos_p * sin_t * dn2 - sin_p * dn3
        g_theta = (cos_t * dn2 - sin_t * dn1) / sin_p
        x1p = x1 + dsigma * R_x1
        x2p = x2 + dsigma * R_x2
        x3p = x3 + dsigma * R_x3
        php = phi + dsigma * g_phi
        thp = theta + dsigma * g_theta
        n2_dsig = n * n * dsigma

    base = model(x1, x2, x3, phi, theta)
    step = model(x1p, x2p, x3p, php, thp)
    diffs = [step[i] - base[i] for i in range(9)]
    diffs[0] = diffs[0] + n2_dsig  # û source
    diffs[1] = diffs[1] + dsigma  # σ̂ source
    return tuple(diffs)
