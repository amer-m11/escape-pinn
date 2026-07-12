"""PDE residual computation for the 2D Eikonal escape equations.

The Escape Equations are defined in phase space y = (x, p) as

    1 + ∇₀σ̂ · R(y₀)         = 0          (escape parameter)
        ∇₀ŷ · R(y₀)         = 0          (escape position, vector eq.)
        ∇₀û · R(y₀) + r(y₀) = 0          (escape value)

where for the Eikonal Hamiltonian H(x, p) = ½|p|² - ½n²(x):

    R(y)  = (∇_p H, -∇_x H) = (p,  n ∇n)
    r(y)  =  p · ∇_p H = p · p = n²(x)

Under the constraint H = 0 we have |p| = n(x), so p has a single degree
of freedom (the ray angle θ), p = n(cos θ, sin θ). The residuals below are
therefore evaluated in the reduced phase space (x₁, x₂, θ) (a 6x3 Jacobian),
with angular drift g_θ = dθ/dσ = -sin θ ∂₁n + cos θ ∂₂n.
"""

from typing import Callable

import torch

from ..networks import EscapeNet2D


def compute_pde_residual_2d(
    model: EscapeNet2D,
    x1: torch.Tensor,
    x2: torch.Tensor,
    theta: torch.Tensor,
    slowness_fn: Callable[[torch.Tensor], torch.Tensor],
    slowness_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Evaluate the six escape-equation residuals at phase-space points.

    The exit angle θ̂ is represented as the pair (cos θ̂, sin θ̂) since they are
    smooth functions which respect the periodicity of the angle.

    Parameters
    ----------
    model : EscapeNet2D
    x1, x2, theta : torch.Tensor
        1-D tensors (length N) of phase-space coordinates. They do not need
        ``requires_grad=True``: the per-sample Jacobian uses ``torch.func``,
        which traces the function functionally rather than relying on a
        leaf-tensor autograd graph on the inputs.
    slowness_fn : callable
        ``n(x) -> (N,)`` slowness values, where ``x`` is the ``(N, 2)`` stack
        of spatial coordinates.
    slowness_grad_fn : callable, optional
        ``∇n(x) -> (N, 2)`` spatial gradient. If omitted, the gradient is
        obtained from ``slowness_fn`` via ``torch.autograd.grad``. Pass an
        explicit callable when an analytic gradient is available: it is
        cheaper than the autograd path and avoids allocating a duplicate
        autograd graph for ``n``.

    Returns
    -------
    res_u, res_sigma, res_y1, res_y2, res_cos_theta, res_sin_theta : torch.Tensor
        Residual of each escape equation at every phase-space point.

    Per-sample Jacobian via ``torch.func``
    --------------------------------------
    We need every partial derivative of every output w.r.t. every
    phase-space coordinate, arranged as a 6 x 3 Jacobian matrix:

                      x₁            x₂            θ
                   ┌──────────────────────────────────────┐
            û      │  ∂û/∂x₁       ∂û/∂x₂        ∂û/∂θ    │
            σ̂      │  ∂σ̂/∂x₁       ∂σ̂/∂x₂        ∂σ̂/∂θ    │
            ŷ₁     │  ∂ŷ₁/∂x₁      ∂ŷ₁/∂x₂       ∂ŷ₁/∂θ   │
            ŷ₂     │  ∂ŷ₂/∂x₁      ∂ŷ₂/∂x₂       ∂ŷ₂/∂θ   │
            cos θ̂  │  ∂cosθ̂/∂x₁    ∂cosθ̂/∂x₂     ∂cosθ̂/∂θ │
            sin θ̂  │  ∂sinθ̂/∂x₁    ∂sinθ̂/∂x₂     ∂sinθ̂/∂θ │
                   └──────────────────────────────────────┘

    NOTE: Computing this with the autograd requires eighteen separate backward
    passes over the network. This matrix is computed in a single composite
    transform.

    This is functionally equivalent to eighteen ``torch.autograd.grad`` calls
    with ``grad_outputs=ones`` and ``create_graph=True``.
    """

    def _stacked_outputs(
        s_x1: torch.Tensor, s_x2: torch.Tensor, s_th: torch.Tensor
    ) -> torch.Tensor:
        """Network evaluation at a single phase-space point.

        Evaluates the network on a single phase-space point (0-D tensors)
        and returns a (6,) vector ``[û, σ̂, ŷ₁, ŷ₂, cos θ̂, sin θ̂]``.
        """
        u, sigma, y1, y2, c_th, s_th = model(s_x1, s_x2, s_th)
        return torch.stack([u, sigma, y1, y2, c_th, s_th])

    jac_x1, jac_x2, jac_th = torch.func.vmap(
        torch.func.jacrev(_stacked_outputs, argnums=(0, 1, 2))
    )(x1, x2, theta)

    du_dx1, ds_dx1, dy1_dx1, dy2_dx1, dcos_dx1, dsin_dx1 = jac_x1.unbind(dim=-1)
    du_dx2, ds_dx2, dy1_dx2, dy2_dx2, dcos_dx2, dsin_dx2 = jac_x2.unbind(dim=-1)
    du_dth, ds_dth, dy1_dth, dy2_dth, dcos_dth, dsin_dth = jac_th.unbind(dim=-1)

    # Slowness field
    # --------------
    x_stack = torch.stack((x1, x2), dim=-1)
    if slowness_grad_fn is None:
        # 0.0 * x_n.sum(dim=-1) is a trick to ensure that the autograd graph is
        # connected to the input tensor, even if the slowness function does not
        # depend on it.
        x_n = x_stack.detach().requires_grad_(True)
        n = slowness_fn(x_n) + 0.0 * x_n.sum(dim=-1)
        (dn_dx,) = torch.autograd.grad(n.sum(), (x_n,), create_graph=False)
        n = n.detach()
    else:
        n = slowness_fn(x_stack)
        dn_dx = slowness_grad_fn(x_stack)
    dn_dx1, dn_dx2 = dn_dx.unbind(dim=-1)

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)

    R_x1 = n * cos_th
    R_x2 = n * sin_th

    g_theta = -sin_th * dn_dx1 + cos_th * dn_dx2

    n2 = n * n

    res_u = du_dx1 * R_x1 + du_dx2 * R_x2 + du_dth * g_theta + n2
    res_sigma = ds_dx1 * R_x1 + ds_dx2 * R_x2 + ds_dth * g_theta + 1.0
    res_y1 = dy1_dx1 * R_x1 + dy1_dx2 * R_x2 + dy1_dth * g_theta
    res_y2 = dy2_dx1 * R_x1 + dy2_dx2 * R_x2 + dy2_dth * g_theta
    res_cos_theta = dcos_dx1 * R_x1 + dcos_dx2 * R_x2 + dcos_dth * g_theta
    res_sin_theta = dsin_dx1 * R_x1 + dsin_dx2 * R_x2 + dsin_dth * g_theta

    return res_u, res_sigma, res_y1, res_y2, res_cos_theta, res_sin_theta


def consistency_residual_2d(
    model: EscapeNet2D,
    x1: torch.Tensor,
    x2: torch.Tensor,
    theta: torch.Tensor,
    slowness_fn: Callable[[torch.Tensor], torch.Tensor],
    slowness_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    dsigma: float,
) -> tuple[torch.Tensor, ...]:
    """Finite difference characteristic consistency residuals.

    Stepping a phase-space point by ``Δσ`` along the characteristic velocity
    ``V = (R_x₁, R_x₂, g_θ)`` must affect the escape quantities as follows:
        - ``û`` by ``-n²·Δσ``
        - ``σ̂`` by ``−Δσ``
        - leave ``ŷ``, ``(cos θ̂, sin θ̂)`` invariant (same exit).
    This couples the channels across a finite step, penalizing non-physical
    fields that satisfy the pointwise residual yet drift over a finite distance.

    The step geometry (``V``) is detached so only the model evaluations carry gradient.

    Returns the six per-channel consistency residuals (≈0 at the solution).
    """
    with torch.no_grad():
        x = torch.stack((x1, x2), dim=-1)
        if slowness_grad_fn is None:
            xg = x.detach().requires_grad_(True)
            n = slowness_fn(xg) + 0.0 * xg.sum(dim=-1)
            (dn,) = torch.autograd.grad(n.sum(), (xg,), create_graph=False)
            n = n.detach()
        else:
            n = slowness_fn(x)
            dn = slowness_grad_fn(x)
        dn1, dn2 = dn.unbind(dim=-1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R_x1 = n * cos_t
        R_x2 = n * sin_t
        g_theta = -sin_t * dn1 + cos_t * dn2
        x1p = x1 + dsigma * R_x1
        x2p = x2 + dsigma * R_x2
        thp = theta + dsigma * g_theta
        n2_dsig = n * n * dsigma

    u, sg, y1, y2, c, s = model(x1, x2, theta)
    up, sgp, y1p, y2p, cp, sp = model(x1p, x2p, thp)
    return (
        up - u + n2_dsig,
        sgp - sg + dsigma,
        y1p - y1,
        y2p - y2,
        cp - c,
        sp - s,
    )
