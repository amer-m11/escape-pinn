"""PDE residual for the 3D escape equations in Cartesian-momentum form.

The reduced phase space is ``(x₁, x₂, x₃, d₁, d₂, d₃)`` with |d| = 1.
"""

from typing import Callable

import torch

from ..networks import EscapeNetCartesian3D


def compute_pde_residual_cartesian_3d(
    model: EscapeNetCartesian3D,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    d3: torch.Tensor,
    slowness_fn: Callable[[torch.Tensor], torch.Tensor],
    slowness_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
]:
    """Evaluate the eight Cartesian-momentum escape equation residuals.

    Parameters
    ----------
    model : EscapeNetCartesian3D
    x1, x2, x3, d1, d2, d3 : torch.Tensor
        1-D tensors (length N) of phase space coordinates. ``(d₁,d₂,d₃)``
        should be unit vectors. Coordinates need not carry
        ``requires_grad`` since the per-sample Jacobian uses ``torch.func``.
    slowness_fn : callable
        ``n(x) -> (N,)`` for the ``(N, 3)`` spatial stack.
    slowness_grad_fn : callable, optional
        ``∇n(x) -> (N, 3)`` or autograd through ``slowness_fn`` if omitted.

    Returns
    -------
    res_u, res_sigma, res_y1, res_y2, res_y3, res_d1, res_d2, res_d3
    """

    def _stacked_outputs(
        s_x1: torch.Tensor,
        s_x2: torch.Tensor,
        s_x3: torch.Tensor,
        s_d1: torch.Tensor,
        s_d2: torch.Tensor,
        s_d3: torch.Tensor,
        *rest: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the network at one point, returning an ``(8,)`` vector.

        ``*rest`` carries the optional per-sample conditioning vector ``s_cond``.
        ``jacrev`` differentiates only ``argnums=(0..5)`` (the six phase coords),
        so ``cond`` rides along (vmapped) but is never differentiated.
        """
        return torch.stack(model(s_x1, s_x2, s_x3, s_d1, s_d2, s_d3, *rest))

    # fmt: off
    _vmap_args = ((x1, x2, x3, d1, d2, d3) if cond is None else (x1, x2, x3, d1, d2, d3, cond))
    jac_x1, jac_x2, jac_x3, jac_d1, jac_d2, jac_d3 = torch.func.vmap(torch.func.jacrev(_stacked_outputs, argnums=(0, 1, 2, 3, 4, 5)))(*_vmap_args)

    # Each ``jac_*`` is ``(N, 8)``, unbind the last axis into eight (N,)
    # per-output tensors in canonical order.
    du_dx1, ds_dx1, dy1_dx1, dy2_dx1, dy3_dx1, de1_dx1, de2_dx1, de3_dx1 = jac_x1.unbind(-1)
    du_dx2, ds_dx2, dy1_dx2, dy2_dx2, dy3_dx2, de1_dx2, de2_dx2, de3_dx2 = jac_x2.unbind(-1)
    du_dx3, ds_dx3, dy1_dx3, dy2_dx3, dy3_dx3, de1_dx3, de2_dx3, de3_dx3 = jac_x3.unbind(-1)
    du_dd1, ds_dd1, dy1_dd1, dy2_dd1, dy3_dd1, de1_dd1, de2_dd1, de3_dd1 = jac_d1.unbind(-1)
    du_dd2, ds_dd2, dy1_dd2, dy2_dd2, dy3_dd2, de1_dd2, de2_dd2, de3_dd2 = jac_d2.unbind(-1)
    du_dd3, ds_dd3, dy1_dd3, dy2_dd3, dy3_dd3, de1_dd3, de2_dd3, de3_dd3 = jac_d3.unbind(-1)
    # fmt: on

    # Slowness field. Same autograd graph-connection trick as the spherical
    # path: ``+ 0.0 * x_n.sum(-1)`` forces a connection so disconnected
    # slowness functions yield zero gradients instead of raising.
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

    # Spatial velocity R_x = n·d.
    R_x1 = n * d1
    R_x2 = n * d2
    R_x3 = n * d3

    # Tangential direction drift g_d = (I − d·dᵀ)∇n = ∇n − d(d·∇n). The
    # vector is tangent to the unit sphere (g_d·d = 0 at |d|=1).
    d_dot_grad_n = d1 * dn_dx1 + d2 * dn_dx2 + d3 * dn_dx3
    g_d1 = dn_dx1 - d1 * d_dot_grad_n
    g_d2 = dn_dx2 - d2 * d_dot_grad_n
    g_d3 = dn_dx3 - d3 * d_dot_grad_n

    def _advect(d_dx1, d_dx2, d_dx3, d_dd1, d_dd2, d_dd3):
        """advection along the ray for one output channel."""
        return (
            d_dx1 * R_x1
            + d_dx2 * R_x2
            + d_dx3 * R_x3
            + d_dd1 * g_d1
            + d_dd2 * g_d2
            + d_dd3 * g_d3
        )

    n2 = n * n
    res_u = _advect(du_dx1, du_dx2, du_dx3, du_dd1, du_dd2, du_dd3) + n2
    res_sigma = _advect(ds_dx1, ds_dx2, ds_dx3, ds_dd1, ds_dd2, ds_dd3) + 1.0
    res_y1 = _advect(dy1_dx1, dy1_dx2, dy1_dx3, dy1_dd1, dy1_dd2, dy1_dd3)
    res_y2 = _advect(dy2_dx1, dy2_dx2, dy2_dx3, dy2_dd1, dy2_dd2, dy2_dd3)
    res_y3 = _advect(dy3_dx1, dy3_dx2, dy3_dx3, dy3_dd1, dy3_dd2, dy3_dd3)
    res_d1 = _advect(de1_dx1, de1_dx2, de1_dx3, de1_dd1, de1_dd2, de1_dd3)
    res_d2 = _advect(de2_dx1, de2_dx2, de2_dx3, de2_dd1, de2_dd2, de2_dd3)
    res_d3 = _advect(de3_dx1, de3_dx2, de3_dx3, de3_dd1, de3_dd2, de3_dd3)

    return (res_u, res_sigma, res_y1, res_y2, res_y3, res_d1, res_d2, res_d3)


def consistency_residual_cartesian_3d(
    model: EscapeNetCartesian3D,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    d3: torch.Tensor,
    slowness_fn: Callable[[torch.Tensor], torch.Tensor],
    slowness_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    dsigma: float,
) -> tuple[torch.Tensor, ...]:
    """Cartesian finite difference characteristic consistency.

    Mirrors the 2D version (see ``consistency_residual_2d``).

    Steps by ``Δσ`` along ``V = (R_x, g_d)`` with ``R_x = n·d`` and the bounded
    tangential drift ``g_d = (I - d·dᵀ)∇n``. The direction step is renormalized
    to the unit sphere.
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
        d_dot_grad = d1 * dn1 + d2 * dn2 + d3 * dn3
        g_d1 = dn1 - d1 * d_dot_grad
        g_d2 = dn2 - d2 * d_dot_grad
        g_d3 = dn3 - d3 * d_dot_grad
        x1p = x1 + dsigma * n * d1
        x2p = x2 + dsigma * n * d2
        x3p = x3 + dsigma * n * d3
        d1p = d1 + dsigma * g_d1
        d2p = d2 + dsigma * g_d2
        d3p = d3 + dsigma * g_d3
        dnorm = torch.sqrt(d1p * d1p + d2p * d2p + d3p * d3p).clamp_min(1e-12)
        d1p, d2p, d3p = d1p / dnorm, d2p / dnorm, d3p / dnorm
        n2_dsig = n * n * dsigma

    base = model(x1, x2, x3, d1, d2, d3)
    step = model(x1p, x2p, x3p, d1p, d2p, d3p)
    diffs = [step[i] - base[i] for i in range(8)]
    diffs[0] = diffs[0] + n2_dsig
    diffs[1] = diffs[1] + dsigma
    return tuple(diffs)
