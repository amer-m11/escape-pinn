"""Conditioning helpers for the slowness-conditioned (generalization) PINN.

The training path threads a per-point ``cond`` vector through the forward
(see ``EscapeNet*3D.forward(..., cond=...)``). For *evaluation* on one fixed
held-out medium we want the entire existing diagnostic/metric surface (which
calls the network with its plain un-conditioned signature) to work unchanged.

:func:`bind_fixed_conditioning` wraps a conditioned model and binds a constant
descriptor, presenting the standard signature and broadcasting the bound vector
to every point. The wrapper subclasses the matching ``EscapeNet*3D`` base, so
``isinstance`` dispatch, ``spherical_view()``, ``arch_details()`` and the
CPU/eval moves in the metric all behave as if it were the original network.

# TODO: this is a bit of a hack, refactor.
"""

from __future__ import annotations

import torch

from .base_network import EscapeNet3D, EscapeNetCartesian3D


class _FixedCondMixin:
    """Shared broadcast of the bound descriptor to a batch (or scalar) input."""

    def _broadcast(self, ref: torch.Tensor) -> torch.Tensor:
        cond = self.cond_vec.to(ref.device)
        if ref.dim() == 0:  # per-sample vmap/jacrev scalar path
            return cond
        return cond.unsqueeze(0).expand(ref.shape[0], -1)


class _FixedCondCartesian3D(_FixedCondMixin, EscapeNetCartesian3D):
    """A conditioned Cartesian model with one medium baked in (eval-only)."""

    N_OUTPUTS = 8

    def __init__(self, inner: EscapeNetCartesian3D, cond_vec: torch.Tensor):
        super().__init__()
        self.inner = inner  # registered submodule → .to()/.eval() propagate
        self.register_buffer("cond_vec", cond_vec.detach().clone())
        self.L_x1, self.L_x2, self.L_x3 = inner.L_x1, inner.L_x2, inner.L_x3
        self.arch_name = getattr(inner, "arch_name", "fixed-cond-cart-3d")

    def forward(self, x1, x2, x3, d1, d2, d3, cond=None):
        return self.inner(x1, x2, x3, d1, d2, d3, self._broadcast(x1))

    def arch_details(self) -> dict:
        return self.inner.arch_details()


class _FixedCondSpherical3D(_FixedCondMixin, EscapeNet3D):
    """A conditioned spherical model with one medium baked in (eval-only)."""

    N_OUTPUTS = 9

    def __init__(self, inner: EscapeNet3D, cond_vec: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("cond_vec", cond_vec.detach().clone())
        self.L_x1, self.L_x2, self.L_x3 = inner.L_x1, inner.L_x2, inner.L_x3
        self.arch_name = getattr(inner, "arch_name", "fixed-cond-3d")

    def forward(self, x1, x2, x3, phi, theta, cond=None):
        return self.inner(x1, x2, x3, phi, theta, self._broadcast(x1))

    def arch_details(self) -> dict:
        return self.inner.arch_details()


def bind_fixed_conditioning(model, cond_vec) -> torch.nn.Module:
    """Wrap a conditioned 3D model with a constant ``cond`` descriptor.

    ``cond_vec`` is the **normalized** descriptor (shape ``(C,)``), typically
    ``family.normalize(params)`` for a single held-out medium. Returns a module
    presenting the model's plain (un-conditioned) signature, so the existing
    metric / diagnostic machinery scores it on that fixed medium unchanged.
    """
    cond_vec = torch.as_tensor(cond_vec)
    if cond_vec.dim() != 1:
        raise ValueError(
            f"cond_vec must be 1-D (C,); got shape {tuple(cond_vec.shape)}"
        )
    if isinstance(model, EscapeNetCartesian3D):
        return _FixedCondCartesian3D(model, cond_vec)
    if isinstance(model, EscapeNet3D):
        return _FixedCondSpherical3D(model, cond_vec)
    raise TypeError(
        "bind_fixed_conditioning supports EscapeNetCartesian3D / EscapeNet3D; "
        f"got {type(model).__name__}"
    )
