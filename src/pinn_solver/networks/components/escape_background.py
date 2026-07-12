"""Analytic straight ray escape distance background for the
factored-Eikonal ansatz."""

import torch

_BIG = 1.0e30
_EPS_D = 1.0e-6


def _calculate_straight_exit_time(
    x: torch.Tensor, length: float, d: torch.Tensor
) -> torch.Tensor:
    """Time to exit within ``[0, length]`` from position ``x`` along direction
    ``d`` (one component of the direction vector) assuming a straight ray path.

    ``+∞`` when ``|d| ≤ eps`` (absorbs float32 noise in cos/sin at cardinal
    angles).
    """
    d_pos = torch.where(d > _EPS_D, d, torch.ones_like(d))  # set to +1 for d≈0
    d_neg = torch.where(d < -_EPS_D, d, -torch.ones_like(d))  # set to -1 for d≈0
    t_pos = (length - x) / d_pos
    t_neg = -x / d_neg
    big = torch.full_like(x, _BIG)
    return torch.where(d > _EPS_D, t_pos, torch.where(d < -_EPS_D, t_neg, big))


def background_escape_2d(x1, x2, theta, physical_size) -> torch.Tensor:
    """Straight ray minimum time from ``(x1,x2)`` to ∂D along ``(cos θ, sin θ)``."""
    L1, L2 = physical_size
    d1, d2 = torch.cos(theta), torch.sin(theta)
    t1 = _calculate_straight_exit_time(x1, L1, d1)
    t2 = _calculate_straight_exit_time(x2, L2, d2)
    return torch.minimum(t1, t2)


def background_escape_3d(x1, x2, x3, phi, theta, physical_size) -> torch.Tensor:
    """Straight ray minimum time along the spherical direction ``(φ, θ)``."""
    L1, L2, L3 = physical_size
    sin_p = torch.sin(phi)
    d1 = sin_p * torch.cos(theta)
    d2 = sin_p * torch.sin(theta)
    d3 = torch.cos(phi)
    t = _calculate_straight_exit_time(x1, L1, d1)
    t = torch.minimum(t, _calculate_straight_exit_time(x2, L2, d2))
    t = torch.minimum(t, _calculate_straight_exit_time(x3, L3, d3))
    return t


def background_escape_cartesian_3d(
    x1, x2, x3, d1, d2, d3, physical_size
) -> torch.Tensor:
    """Straight ray minimum time along the unit direction ``(d1,d2,d3)``."""
    L1, L2, L3 = physical_size
    t = _calculate_straight_exit_time(x1, L1, d1)
    t = torch.minimum(t, _calculate_straight_exit_time(x2, L2, d2))
    t = torch.minimum(t, _calculate_straight_exit_time(x3, L3, d3))
    return t
