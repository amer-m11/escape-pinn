from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class _NetworkConfiguration(BaseModel):
    """Configuration for the PINN network architecture."""

    arch_name: str

    unit_norm_output: bool = False
    """Normalize the exit-angle outputs onto the unit circle/sphere: (cos θ̂,
    sin θ̂) [+ (cos φ̂, sin φ̂) in 3D-spherical] / d̂ in 3D-Cartesian. Makes the
    recovered exit direction exact and drives circle/sphere drift to zero by
    construction. Default off."""

    factored_eikonal: bool = False
    """Factored-Eikonal ansatz: û = (analytic straight ray distance-to-∂D) + NN,
    σ̂ likewise (additive, no mask). The background is the exact n=1 escape time,
    so the network learns only the slowness perturbation. The BC loss still pins
    û=0 on outward rays (where the background is 0). The background is
    direction-dependent, so it does not impose the wrong û=0 on inward boundary
    rays. Default off. Mutually exclusive with the other û/σ̂ assembly ansätze
    (``domain_decomp`` / ``selector_head``)."""

    exit_time_features: bool = False
    """Feed the three normalized straight-ray exit times (+ their min) as
    extra trunk inputs (3D-Cartesian only). Composes with any trunk and with 
    the other û ansätze."""

    exit_time_sign: bool = False
    """Append the 3 per-axis ``sign(dᵢ)`` indicators to the exit-time inputs
    (a "discontinuity source" a smooth net can multiply to build a step). Ignored
    unless ``exit_time_features``. 3D only."""

    conditioning_dim: int = 0
    """Slowness-CONDITIONING input width (generalization). When > 0 the 3D
    forward takes an extra ``cond: (..., conditioning_dim)`` tensor concatenated to
    the trunk inputs, so a single network solves a *family* of media. Param mode fills
    it with the normalized lens descriptor (``GaussianLensFamilySlowness.param_dim``).
    Encoder mode with the probe-grid slowness (width ``n_probe^d``) run through a field
    encoder. ``0`` (default) -> no conditioning. Auto-set by the runner from the 
    family's ``cond_dim``. 3D only."""

    field_encoder: bool = False
    """Encoder mode DeepONet-style branch encoder: map the raw conditioning vector
    (a probe-grid slowness field of width ``conditioning_dim``) to a latent of width
    ``encoder_latent_dim`` before the trunk. Off (default) -> the conditioning
    vector is fed to the trunk directly (param mode). Requires ``conditioning_dim > 0``.
    3D only (Cartesian + spherical)."""

    encoder_latent_dim: int = 16
    """Field-encoder latent width (trunk-input contribution). Ignored unless
    ``field_encoder``."""

    encoder_hidden: int = 64
    """Field-encoder hidden width. Ignored unless ``field_encoder``."""

    encoder_depth: int = 2
    """Field-encoder hidden-layer count. Ignored unless ``field_encoder``."""

    domain_decomp: Optional[int] = None
    """Domain-decomposition ansatz (FBPINN-style): tile the spatial box into an
    ``domain_decomp³`` grid of overlapping subdomains, each a smooth subnet,
    combined by a normalized cos²-window partition of unity. Localizes the
    caustic/medial kink to subdomain boundaries. ``None``/absent = off (single
    network). Mutually exclusive with ``factored_eikonal`` / ``selector_head``.
    3D-spherical only for now. Cost: ``domain_decomp³`` full subnets."""

    domain_decomp_overlap: float = 0.25
    """Subdomain overlap fraction (window half-width = (0.5+overlap)·spacing).
    >0 -> neighbors overlap so the partition of unity covers the whole box.
    Ignored unless ``domain_decomp``."""

    selector_head: bool = False
    """Exit-wall selector head: blend ``selector_branches`` smooth branch-nets by 
    a learned gate ``w=softmax(s·g(x,d))`` with an independent gate sub-net ``g`` and
    a learnable slope ``s``. A sharp learned gate can approach a value step. 3D 
    (Cartesian + spherical). Mutually exclusive with ``factored_eikonal`` / 
    ``domain_decomp`` (all reshape the û output)."""

    selector_branches: int = 2
    """Number of selector-head branches. Ignored unless ``selector_head``."""

    @model_validator(mode="after")
    def _exclusive_output_assembly(self):
        # factored_eikonal / domain_decomp / selector_head all set the û/σ̂ output
        # assembly -> enable at most one.
        if self.domain_decomp is not None:
            if self.domain_decomp < 2:
                raise ValueError("domain_decomp must be >= 2 (or None to disable).")
            if self.factored_eikonal:
                raise ValueError(
                    "domain_decomp and factored_eikonal are mutually exclusive "
                    "output-assembly ansätze -> enable at most one."
                )
        if self.selector_head:
            if self.selector_branches < 2:
                raise ValueError("selector_branches must be >= 2.")
            if self.factored_eikonal or self.domain_decomp is not None:
                raise ValueError(
                    "selector_head, factored_eikonal and domain_decomp "
                    "are mutually exclusive output-assembly ansätze -> enable at most one."
                )
        return self


class FourierFeatureSpec(BaseModel):
    """Random-Fourier-feature input encoding applied to the spatial inputs.

    The mapping is γ(v) = [sin(2π B v), cos(2π B v)] with B ∈ ℝ^{n_features × 2}
    and B_ij ~ 𝒩(0, σ²). Lifts the (x₁, x₂) input from a 2-D smooth manifold
    to a high-frequency basis that the tanh MLP can resolve sharp features in
    (lens-refraction lobes, C⁰ caustic ridges). The angle inputs (cos θ, sin θ)
    are passed through unchanged. They are already a bounded 2-D embedding.

    Init: B is sampled once with the current torch RNG (which PINNRunner has
    already seeded), then frozen as a non-trainable buffer.
    """

    n_features: int = Field(gt=0)
    """Number of Fourier features per channel. The MLP input dim becomes
    2·n_features + 2 (sin + cos pairs from spatial, plus cos θ + sin θ)."""

    sigma: float = Field(gt=0)
    """Std-dev of the Gaussian B-matrix entries. Higher σ -> higher
    frequencies. Typical range 1-15."""

    angular: bool = False
    """3D only: also apply a Fourier lift to the angular inputs. When
    ``False`` (default, conventional) the angular ``(cos θ, sin θ, cos φ,
    sin φ)`` embedding is passed through raw. When ``True`` that 4-vector
    embedding is itself projected through a dedicated B-matrix. Ignored in 2D."""


class FCMLPConfiguration(_NetworkConfiguration):
    """Configuration for a fully-connected MLP escape net."""

    name: Literal["fc-mlp"] = "fc-mlp"
    hidden_layers: int
    hidden_neurons: int
    activation: Literal["tanh"]

    trunk: Literal["tanh", "adaptive_tanh", "pirate", "siren"] = "tanh"
    """MLP trunk family (see networks.components.trunks).
    - ``tanh`` is the baseline
    - ``adaptive_tanh`` adds learnable per-layer slopes
    - ``pirate`` is the gated-residual PirateNet form

    The input Fourier encoding composes with any trunk."""

    fourier_features: Optional[FourierFeatureSpec] = None
    """When set, prepend a random-Fourier-feature spatial encoding. When None
    (default) the network falls back to (x1_norm, x2_norm, cos θ, sin θ) input."""

    per_channel_heads: bool = False
    """Replace the single trunk readout with one independent ``Linear`` head per
    output channel (shared trunk body), so channels of differing regularity get
    dedicated readouts. Default off."""


class SIRENConfiguration(_NetworkConfiguration):
    """Configuration for a SIREN escape net (sine activations + ω₀ init)."""

    name: Literal["siren"] = "siren"
    hidden_layers: int
    hidden_neurons: int
    omega_0: float = 30.0


NetworkConfiguration = Annotated[
    Union[FCMLPConfiguration, SIRENConfiguration],
    Field(discriminator="name"),
]
