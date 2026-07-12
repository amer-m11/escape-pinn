import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .network import NetworkConfiguration
from .optimizers import OptimizersConfiguration
from .slowness import GaussianLensSlowness, SlownessSpecification
from .types import LossType, PhysicalSize, SampleMode

DEFAULT_PHI_SAFE_DELTA = math.pi / 8
"""Default φ safe-band half-width in 3D. Rays within δ of either pole
(φ ∈ [0, δ) ∪ (π−δ, π]) are excluded from PDE collocation because the
angular-drift term g_θ carries a 1/sin φ factor that is unbounded at the
poles. δ = π/8 keeps sin φ ≥ sin(π/8) ≈ 0.38 (≤ 2.6× amplification)."""


class BoundarySpecification(BaseModel):
    """Boundary condition specification."""

    physical_boundary_samples: int
    """Number of spatial points sampled per boundary edge (2D) or face (3D).
    Pre-sampled once and reused."""
    theta_samples_at_boundary: int
    """Number of θ samples for boundary collocation. Each physical boundary point is sampled
    with this many θ values. Only outward pointing rays survive. Pre-sampled once and reused."""
    phi_samples_at_boundary: Optional[int] = None
    """Number of φ samples for boundary collocation in 3D. Required when ``physical_size``
    is 3D (``L_x3`` set), ignored otherwise."""
    phi_safe_delta: Optional[float] = None
    """φ safe-band half-width (3D). Single source of truth shared by the
    interior collocation sampler, the boundary sampler, and the metric
    φ-sweep so all three agree on the trained band ``φ ∈ [δ, π−δ]``.
    ``None`` resolves to :data:`DEFAULT_PHI_SAFE_DELTA` (π/8). Ignored in 2D."""
    sample_mode: SampleMode
    """How to sample boundary collocation points."""
    resample: bool = False
    """Whether to resample boundary collocation points every Adam epoch. L-BFGS keeps
    the boundary set frozen to avoid line-search thrashing. Only meaningful when
    ``sample_mode='random'``."""

    def resolved_phi_delta(self) -> float:
        """The effective φ safe-band half-width (config value or π/8 default)."""
        return (
            self.phi_safe_delta
            if self.phi_safe_delta is not None
            else DEFAULT_PHI_SAFE_DELTA
        )

    @field_validator("phi_safe_delta")
    @classmethod
    def _delta_in_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 < v < math.pi / 2:
            raise ValueError(f"phi_safe_delta must lie in (0, π/2), got {v!r}")
        return v


DataChannel = Literal[
    "u",
    "sigma",
    "y1",
    "y2",
    "cos",
    "sin",  # 2D tokens
    "y3",
    "cos_theta",
    "sin_theta",
    "cos_phi",
    "sin_phi",  # 3D spherical tokens
    "d1",
    "d2",
    "d3",  # 3D Cartesian tokens
]
"""Token set for the supervised-data channel filter.

- ``{u, sigma, y1, y2, cos, sin}`` map to the 6-output 2D network.
- ``{u, sigma, y1, y2, y3, cos_theta, sin_theta, cos_phi, sin_phi}``
map to the 9-output spherical 3D network.
- ``{u, sigma, y1, y2, y3, d1, d2, d3}`` map to the 8-output Cartesian
momentum network (the exit *direction* is a unit 3-vector.
The cross-field validator on ``TrainConfiguration`` enforces 
dim + parametrization consistency.
"""

_ALL_DATA_CHANNELS_2D: list[DataChannel] = ["u", "sigma", "y1", "y2", "cos", "sin"]
_ALL_DATA_CHANNELS_3D: list[DataChannel] = [
    "u",
    "sigma",
    "y1",
    "y2",
    "y3",
    "cos_theta",
    "sin_theta",
    "cos_phi",
    "sin_phi",
]
_ALL_DATA_CHANNELS_CART_3D: list[DataChannel] = [
    "u",
    "sigma",
    "y1",
    "y2",
    "y3",
    "d1",
    "d2",
    "d3",
]
_ALL_DATA_CHANNELS: list[DataChannel] = _ALL_DATA_CHANNELS_2D  # default = 2D


def full_data_channels(dim: int, parametrization: str) -> list[DataChannel]:
    """The full supervised-channel set for a (dim, parametrization) pair."""
    if dim != 3:
        return list(_ALL_DATA_CHANNELS_2D)
    if parametrization == "cartesian":
        return list(_ALL_DATA_CHANNELS_CART_3D)
    return list(_ALL_DATA_CHANNELS_3D)


class LossConfiguration(BaseModel):
    lambda_pde: float
    """Loss weight for the PDE residual term."""
    lambda_bc: float
    """Loss weight for the boundary condition term. λ_bc > λ_pde is standard PINN practice."""
    lambda_data: float
    """Loss weight for the optional supervised data-fit term."""

    lambda_circle: float = 0.0
    """Loss weight for the unit-circle regularizer ``mean((c²+s²-1)²)`` over
    interior collocation points. Pins ``(cos θ̂, sin θ̂)`` (and ``(cos φ̂,
    sin φ̂)`` in 3D) to the unit circle globally. The boundary loss only
    enforces it on ∂D. Default 0 (off) so existing runs are unaffected."""

    lambda_consistency: float = 0.0
    """Loss weight for the finite-difference characteristic-consistency term. 
    Steps each collocation point by ``Δσ`` along the characteristic velocity
    and penalizes deviation from the source law (û drops by n²·Δσ, σ̂ by Δσ, 
    ŷ + exit-direction invariant). Couples the channels over a finite step,
    suppressing non-physical fields that satisfy the pointwise residual. 
    Default 0 (off)."""

    consistency_dsigma: float = 0.02
    """Step size Δσ for the consistency term (phase-space arc length). Small
    enough that the first-order source law holds, large enough to couple the
    channels over a finite distance. Only used when ``lambda_consistency > 0``."""

    viscosity_eps: float = 0.0
    """3D only: vanishing-viscosity regularization. Adds ``-ε·∇²f`` (the spatial
    Laplacian of û and σ̂) to their PDE residuals before the MSE, so the trained
    field is the artificial-viscosity solution ``∇f·R - ε∇²f = source``. The
    textbook selector of the correct viscosity solution at the Hamilton-Jacobi
    caustic/exit-switch kink. ``0.0`` (default) = off. Anneal ε->0 with
    ``viscosity_anneal`` so the bias vanishes as training converges. Adds 2nd-order
    autodiff (~2-3× residual cost). No effect in 2D."""

    viscosity_anneal: Literal["none", "cosine"] = "none"
    """Schedule for ``viscosity_eps`` over the Adam phase. ``cosine`` ramps ε from
    its configured value -> 0 (half-cosine), so viscosity regularizes early training
    and vanishes by the end. ``none`` (default) keeps ε fixed. Ignored unless
    ``viscosity_eps > 0``."""

    causality_eps: float = 0.0
    """3D only: causality-weighted residual (Wang 2022). Each collocation point's
    PDE residual is weighted by ``exp(-ε·M_i)`` where ``M_i`` is the cumulative
    residual of points causally upstream (smaller predicted σ̂ = nearer the exit
    boundary, where the escape source σ̂=0 lives), normalized to [0,1]. The network
    fits the boundary-near characteristics first and only emphasizes downstream
    points once upstream residuals fall - resolving the caustic in causal order
    instead of averaging across it. The gate self-relaxes to ~uniform as residuals
    fall during training. ``ε`` is O(1) (the normalization makes it independent of
    collocation count). ε≈3 down-weights the most-downstream point to ~5%. ``0.0``
    (default) = off (uniform weighting). No effect in 2D."""

    pole_safe_residual: bool = False
    """3D only: train against the ``sin φ``-multiplied residual so the g_θ
    ``1/sin φ`` term's division cancels and the residual is finite at the
    poles. Lets the φ safe-band δ shrink toward 0 for near-full angular
    coverage. No effect in 2D."""

    function: LossType
    """Loss function to use for PDE, BC, and data terms."""

    data_channels: Optional[list[DataChannel]] = None
    """Channels included in the supervised data MSE. ``None`` (default)
    resolves to the dim-appropriate full set at the
    :class:`TrainConfiguration` level (all 6 channels in 2D, all 9 in 3D).
    Explicit lists are validated against the active dim."""

    @field_validator("data_channels")
    @classmethod
    def _dedupe_and_validate(
        cls, v: Optional[list[DataChannel]]
    ) -> Optional[list[DataChannel]]:
        if v is None:
            return None  # resolved per-dim at the TrainConfiguration level
        if not v:
            raise ValueError(
                "data_channels must be non-empty. Set lambda_data=0 to "
                "disable supervised data, do not pass []"
            )
        canonical: list[DataChannel] = list(_ALL_DATA_CHANNELS_2D)
        for ch in _ALL_DATA_CHANNELS_3D + _ALL_DATA_CHANNELS_CART_3D:
            if ch not in canonical:
                canonical.append(ch)
        seen: set[str] = set()
        ordered: list[DataChannel] = []
        for ch in canonical:
            if ch in v and ch not in seen:
                ordered.append(ch)
                seen.add(ch)
        return ordered


class TrainConfiguration(BaseModel):
    net: NetworkConfiguration
    slowness: SlownessSpecification
    physical_size: PhysicalSize
    """Physical domain extents (L_x1, L_x2). Used for BC sampling and 
    position residual normalization."""

    @model_validator(mode="before")
    @classmethod
    def _cartesian_unit_norm_default(cls, data):
        """Default ``net.unit_norm_output=True`` for Cartesian runs.

        The Cartesian direction output ``d̂`` is a native unit vector, so hard
        unit-norm costs nothing in accuracy and drives the interior unit-norm
        drift to ~0 (``sphere_drift`` 0.03→3.6e-7 on the lens). Applied only
        when the field is not explicitly set, so ``--set net.unit_norm_output=false``
        (or an explicit YAML value) still wins. Spherical is unaffected (hard-norming
        the four coupled (cos,sin) angle channels fights the BC. it *hurts* there)."""
        if isinstance(data, dict) and data.get("parametrization") == "cartesian":
            net = data.get("net")
            if isinstance(net, dict) and "unit_norm_output" not in net:
                net["unit_norm_output"] = True
        return data

    parametrization: Literal["spherical", "cartesian"] = "spherical"
    """Phase-space momentum parametrization (3D only).

    - ``spherical`` (default): the ray direction is carried by polar angles
      ``(φ, θ)``. The angular-drift term has a ``1/sin φ`` factor singular
      at the poles, so the φ-band ``(δ, π−δ)`` excludes a near-pole cap.
    - ``cartesian``: the ray direction is a unit 3-vector ``d``. The angular
      drift is ``(I - d·dᵀ)∇n`` (bounded everywhere), so the full sphere of
      directions is trainable.
    
    Ignored in 2D."""

    boundary: BoundarySpecification

    loss: LossConfiguration

    device: Literal["auto", "cpu", "cuda"] = "cpu"
    """Device to run training on. Defaults to ``cpu``. ``auto`` resolves to
    CUDA when available, else CPU. ``cuda`` fails at runner construction when
    CUDA is unavailable."""

    optimizers: OptimizersConfiguration

    seed: Optional[int] = None
    """Random seed for reproducibility.

    Covers torch's default CPU+CUDA RNGs (weight init) plus a dedicated
    :class:`torch.Generator` consumed by the interior collocation sampler.
    When ``None`` a 64-bit seed is drawn at runtime and recorded.
    """

    deterministic: bool = False
    """If true, force bit-exact reproducibility on CUDA by enabling
    :func:`torch.use_deterministic_algorithms` (with ``warn_only=True``)
    and setting ``CUBLAS_WORKSPACE_CONFIG`` at runner construction time.

    Default false because some kernels (e.g. scatter-reduce on CUDA) lack
    a deterministic implementation and slow down 1.5-2×. Turn on for
    parity / regression runs.
    """

    @model_validator(mode="after")
    def _check_dim_consistency(self) -> "TrainConfiguration":
        dim = self.physical_size.dim
        if self.parametrization == "cartesian" and dim != 3:
            raise ValueError(
                "parametrization='cartesian' is only valid for 3D physical_size"
            )
        if dim == 3 and self.boundary.phi_samples_at_boundary is None:
            raise ValueError(
                "boundary.phi_samples_at_boundary is required when physical_size is 3D"
            )
        if isinstance(self.slowness, GaussianLensSlowness):
            if len(self.slowness.center) != dim:
                raise ValueError(
                    f"slowness.center has length {len(self.slowness.center)} "
                    f"but physical_size is {dim}D"
                )
        # data_channels default resolution + dim/parametrization token check.
        full_set = full_data_channels(dim, self.parametrization)
        if self.loss.data_channels is None:
            self.loss.data_channels = list(full_set)
        else:
            allowed = set(full_set)
            channels = set(self.loss.data_channels)
            if not channels.issubset(allowed):
                stray = sorted(channels - allowed)
                raise ValueError(
                    f"data_channels {stray} are not valid for a {dim}D "
                    f"{self.parametrization} run, allowed: {sorted(allowed)}"
                )
        return self
