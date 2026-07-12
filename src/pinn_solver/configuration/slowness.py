from abc import ABC, abstractmethod
from typing import Annotated, Callable, Literal, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor


def gaussian_lens_perturbation(
    x: Tensor, center: Tensor, sigma: Tensor, n1: Tensor | float
) -> Tensor:
    """Gaussian lens without the background ``n1·exp(-Σᵢ (xᵢ-cᵢ)²/(2σᵢ²))``
    with n1 being the amplitude. This function is vectorized over ``x``, so
    it solves multiple gaussians at once."""
    diff = x - center
    exp_arg = ((diff**2) / (2.0 * sigma**2)).sum(dim=-1)
    return n1 * torch.exp(-exp_arg)


def gaussian_lens_n(
    x: Tensor, center: Tensor, sigma: Tensor, n0: float, n1: Tensor | float
) -> Tensor:
    """Slowness ``n0 + perturbation`` (see :func:`gaussian_lens_perturbation`) with
    ``n0`` being the background slowness. This function is vectorized over ``x``, so
    it solves multiple gaussians at once."""
    return n0 + gaussian_lens_perturbation(x, center, sigma, n1)


def gaussian_lens_grad(
    x: Tensor, center: Tensor, sigma: Tensor, n1: Tensor | float
) -> Tensor:
    """``∇ₓn = -perturbation·(x-c)/σ²`` → ``(..., d)`` (``n0`` drops out).
    Input may be a single lens or a batch of lenses."""
    diff = x - center
    bump = gaussian_lens_perturbation(x, center, sigma, n1).unsqueeze(-1)
    return -bump * diff / (sigma**2)


class _SlownessSpecification(BaseModel, ABC):

    @abstractmethod
    def get_slowness(self) -> Callable[[Tensor], Tensor]:
        """Return ``n(x)`` callable accepting ``x: (N, d)`` and returning ``(N,)``."""

    @abstractmethod
    def get_slowness_gradient(self) -> Callable[[Tensor], Tensor]:
        """Return ``∇n(x)`` callable accepting ``x: (N, d)`` and returning ``(N, d)``."""

    def get_slowness_scaled(self, scale: float) -> Callable[[Tensor], Tensor]:
        """Returns ``n(x)`` callable accepting scaled by ``scale`` ∈ [0,1].
        scale=0 -> constant background, scale=1 -> full lens amplitude."""
        return self.get_slowness()

    def get_slowness_gradient_scaled(self, scale: float) -> Callable[[Tensor], Tensor]:
        return self.get_slowness_gradient()


class ConstantSlowness(_SlownessSpecification):
    type: Literal["constant"] = "constant"
    n0: float

    def get_slowness(self) -> Callable[[Tensor], Tensor]:
        n0 = self.n0

        def slowness(x: Tensor) -> Tensor:
            return torch.full(x.shape[:-1], n0, dtype=x.dtype, device=x.device)

        return slowness

    def get_slowness_gradient(self) -> Callable[[Tensor], Tensor]:
        return lambda x: torch.zeros_like(x)


class GaussianLensSlowness(_SlownessSpecification):
    type: Literal["gaussian_lens"] = "gaussian_lens"
    center: list[float]
    """Lens centre as a ``d``-vector matching the physical domain dim."""
    sigma: list[float]
    """Per-axis Gaussian standard deviations as a ``d``-vector."""
    n0: float
    """Base slowness (background medium)."""
    n1: float
    """Slowness amplitude (peak slowness at the centre of the lens is ``n0 + n1``)."""

    @field_validator("sigma")
    @classmethod
    def _sigma_matches_center(cls, sigma, info):
        center = info.data.get("center")
        if center is not None and len(sigma) != len(center):
            raise ValueError(
                f"sigma length ({len(sigma)}) must match center length ({len(center)})"
            )
        return sigma

    def _perturbation(self) -> Callable[[Tensor], Tensor]:
        """The lens contribution ``n1·exp(-Σᵢ (xᵢ-cᵢ)²/(2σᵢ²))`` only (no n0).

        Used by :class:`CompositeGaussianLensSlowness` so summing several lenses
        does not duplicate the background.
        """
        center_list = self.center
        sigma_list = self.sigma
        n1 = self.n1

        def perturbation(x: Tensor) -> Tensor:
            center = torch.tensor(center_list, dtype=x.dtype, device=x.device)
            sigma = torch.tensor(sigma_list, dtype=x.dtype, device=x.device)
            return gaussian_lens_perturbation(x, center, sigma, n1)

        return perturbation

    def get_slowness(self) -> Callable[[Tensor], Tensor]:
        perturbation = self._perturbation()
        n0 = self.n0
        return lambda x: n0 + perturbation(x)

    def get_slowness_scaled(self, scale: float) -> Callable[[Tensor], Tensor]:
        perturbation = self._perturbation()
        n0 = self.n0
        s = float(scale)
        return lambda x: n0 + s * perturbation(x)

    def get_slowness_gradient_scaled(self, scale: float) -> Callable[[Tensor], Tensor]:
        g = self.get_slowness_gradient()
        s = float(scale)
        return lambda x: s * g(x)

    def get_slowness_gradient(self) -> Callable[[Tensor], Tensor]:
        center_list = self.center
        sigma_list = self.sigma
        n1 = self.n1

        def slowness_gradient(x: Tensor) -> Tensor:
            center = torch.tensor(center_list, dtype=x.dtype, device=x.device)
            sigma = torch.tensor(sigma_list, dtype=x.dtype, device=x.device)
            return gaussian_lens_grad(x, center, sigma, n1)

        return slowness_gradient


class CompositeGaussianLensSlowness(_SlownessSpecification):
    type: Literal["composite_gaussian_lens"] = "composite_gaussian_lens"
    n0: float
    """Shared background slowness. The member lenses' own ``n0`` fields are
    ignored. Only their ``n1``-scaled Gaussian bumps are summed on top of this
    background."""
    lenses: list[GaussianLensSlowness]

    def get_slowness(self) -> Callable[[Tensor], Tensor]:
        perturbations = [lens._perturbation() for lens in self.lenses]
        n0 = self.n0

        def slowness_fn(x: Tensor) -> Tensor:
            total = torch.full(x.shape[:-1], n0, dtype=x.dtype, device=x.device)
            for perturbation in perturbations:
                total = total + perturbation(x)
            return total

        return slowness_fn

    def get_slowness_gradient(self) -> Callable[[Tensor], Tensor]:
        gradient_fns = [lens.get_slowness_gradient() for lens in self.lenses]

        def slowness_gradient(x: Tensor) -> Tensor:
            total = torch.zeros_like(x)
            for gradient in gradient_fns:
                total = total + gradient(x)
            return total

        return slowness_gradient


class _ConditionedFamily(_SlownessSpecification):
    """Base for slowness families that drive the conditioned PINN.

    Concrete families expose: ``cond_dim`` (network conditioning width),
    ``param_dim``, ``sample_params``, ``cond_for(size, params)`` (the network
    conditioning vector), ``slowness_at``/``slowness_grad_at``, and
    ``to_concrete(params)`` (a fixed spec for the exact reference)..
    """


class GaussianLensFamilySlowness(_ConditionedFamily):
    """A family of single Gaussian lenses, parametrized by ``(n1, center, σ)``.

    Each member of this family has a descriptor vector of length 1+2d :
    ``params = [n1, c₁..c_d, σ₁..σ_d]``. ``slowness_at``/``slowness_grad_at``
    evaluate ``n(x; params)`` with a different lens per row. ``to_concrete``
    materializes a fixed :class:`GaussianLensSlowness` for one descriptor
    (used to build the exact held-out ODE reference). ``get_slowness``
    falls back to the mean descriptor so the spec is still usable as a fixed
    medium if some path treats it that way."""

    type: Literal["gaussian_lens_family"] = "gaussian_lens_family"
    n0: float
    n1_range: tuple[float, float]
    center_ranges: list[tuple[float, float]]
    """Per-axis ``(lo, hi)`` for the lens centre — length = physical dim ``d``."""
    sigma_ranges: list[tuple[float, float]]
    """Per-axis ``(lo, hi)`` for the lens width — length ``d``."""
    held_out: list[list[float]] = []
    """Explicit physical descriptors ``[n1, c…, σ…]`` reserved for the
    generalization test (never drawn during training)."""
    n_probe: int = 0
    """field-encoder mode: when > 0, the conditioning vector is the slowness 
    *perturbation* (``n-n0``) sampled at a fixed ``n_probe^d`` grid of probe
    points in the box (fed to a :class:`FieldEncoder` branch net), instead
    of the normalized lens descriptor. ``0`` (default) -> param mode."""

    @field_validator("sigma_ranges")
    @classmethod
    def _sigma_len_matches_center(cls, sigma_ranges, info):
        center_ranges = info.data.get("center_ranges")
        if center_ranges is not None and len(sigma_ranges) != len(center_ranges):
            raise ValueError(
                f"sigma_ranges length ({len(sigma_ranges)}) must match "
                f"center_ranges length ({len(center_ranges)})"
            )
        return sigma_ranges

    @model_validator(mode="after")
    def _held_out_shape(self):
        for t in self.held_out:
            if len(t) != self.param_dim:
                raise ValueError(
                    f"held_out tuple {t} has length {len(t)}, expected "
                    f"param_dim={self.param_dim} (= 1 + 2·{self.dim})"
                )
        return self

    @property
    def dim(self) -> int:
        """Physical dimension"""
        return len(self.center_ranges)

    @property
    def param_dim(self) -> int:
        """Descriptor length: 1 + 2·d = n1 + center + σ."""
        return 1 + 2 * self.dim

    def _bounds(self, device=None, dtype=torch.get_default_dtype()):
        """``(lo: Tensor, hi: Tensor)`` for all the descriptors."""
        lo = (
            [self.n1_range[0]]
            + [c[0] for c in self.center_ranges]
            + [s[0] for s in self.sigma_ranges]
        )
        hi = (
            [self.n1_range[1]]
            + [c[1] for c in self.center_ranges]
            + [s[1] for s in self.sigma_ranges]
        )
        return (
            torch.tensor(lo, device=device, dtype=dtype),
            torch.tensor(hi, device=device, dtype=dtype),
        )

    def sample_params(
        self, n: int, *, generator=None, device=None, dtype=torch.get_default_dtype()
    ) -> Tensor:
        """``(n, param_dim)`` of physical descriptors drawn uniformly in range."""
        lo, hi = self._bounds(device=device, dtype=dtype)
        u = torch.rand(
            n, self.param_dim, generator=generator, device=device, dtype=dtype
        )
        return lo + u * (hi - lo)

    def normalize(self, params: Tensor) -> Tensor:
        """Physical descriptors -> ``[-1, 1]`` (the network input convention).
        If the range is degenerate (hi=lo), the normalized value is 0 (no lens)."""
        lo, hi = self._bounds(device=params.device, dtype=params.dtype)
        span = hi - lo  # if hi=lo, normalized = 0 (no lens, constant medium)
        safe = torch.where(span > 0, span, torch.ones_like(span))
        normed = 2.0 * (params - lo) / safe - 1.0
        return torch.where(span > 0, normed, torch.zeros_like(normed))

    def denormalize(self, normed: Tensor) -> Tensor:
        """Reverts :func:`normalize`."""
        lo, hi = self._bounds(device=normed.device, dtype=normed.dtype)
        return lo + 0.5 * (normed + 1.0) * (hi - lo)

    def split_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``params (..., param_dim)`` -> ``n1 (...,)``, ``center (..., d)``, ``sigma (..., d)``."""
        d = self.dim
        return params[..., 0], params[..., 1 : 1 + d], params[..., 1 + d : 1 + 2 * d]

    def scale_contrast(self, params: Tensor, scale: float) -> Tensor:
        """Multiply the lens amplitude ``n1`` by ``scale``. ``scale=0`` -> the constant
        medium ``n0`` (no caustic), ``scale=1`` -> the full amplitude."""
        if scale == 1.0:
            return params
        out = params.clone()
        out[..., 0] = out[..., 0] * scale
        return out

    # --- Conditioning vector (the network input): param descriptor or probe grid ---
    @property
    def cond_dim(self) -> int:
        """Width of the conditioning vector fed to the network."""
        return self.n_probe**self.dim if self.n_probe > 0 else self.param_dim

    def probe_points(
        self, size, device=None, dtype=torch.get_default_dtype()
    ) -> Tensor:
        """Fixed ``(n_probe^d, d)`` grid of probe points spanning the box ``size``."""
        axes = [
            torch.linspace(0.0, float(L), self.n_probe, device=device, dtype=dtype)
            for L in size
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return grid.reshape(-1, len(size))

    def _probe_grid_cond(self, size, params: Tensor) -> Tensor:
        """Probe-grid perturbation ``n(probe; params)-n0`` -> ``(N, n_probe^d)``."""
        pts = self.probe_points(size, device=params.device, dtype=params.dtype)  # (G,d)
        n1, center, sigma = self.split_params(params)  # (N,), (N,d), (N,d)
        return gaussian_lens_perturbation(
            pts.unsqueeze(0), center.unsqueeze(1), sigma.unsqueeze(1), n1.unsqueeze(1)
        )  # (N, G)

    def cond_for(self, size, params: Tensor) -> Tensor:
        """Conditioning vector for a batch of physical descriptors ``(N, param_dim)``.

        Grid mode (``n_probe>0``) -> probe-grid perturbation ``(N, n_probe^d)``;
        param mode -> normalized descriptor ``(N, param_dim)`` in [-1, 1]."""
        if params.dim() != 2:
            raise ValueError(
                f"cond_for expects params (N, param_dim); got {tuple(params.shape)}"
            )
        if self.n_probe > 0:
            return self._probe_grid_cond(size, params)
        return self.normalize(params)

    def sample_cond(
        self,
        size,
        n: int,
        *,
        generator=None,
        device=None,
        dtype=torch.get_default_dtype(),
    ) -> Tensor:
        """``(n, cond_dim)`` conditioning vectors for ``n`` freshly-sampled media."""
        return self.cond_for(
            size, self.sample_params(n, generator=generator, device=device, dtype=dtype)
        )

    def slowness_at(self, x: Tensor, params: Tensor) -> Tensor:
        """``n(x; params)`` with a per-row lens. ``x (N, d)``, ``params (N, param_dim)``."""
        n1, center, sigma = self.split_params(params)
        return gaussian_lens_n(x, center, sigma, self.n0, n1)

    def slowness_grad_at(self, x: Tensor, params: Tensor) -> Tensor:
        n1, center, sigma = self.split_params(params)
        return gaussian_lens_grad(x, center, sigma, n1)

    def to_concrete(self, params) -> GaussianLensSlowness:
        """Materialize one descriptor into a fixed :class:`GaussianLensSlowness`."""
        p = (
            params
            if isinstance(params, Tensor)
            else torch.tensor(params, dtype=torch.get_default_dtype())
        )
        n1, center, sigma = self.split_params(p)
        return GaussianLensSlowness(
            center=center.tolist(), sigma=sigma.tolist(), n0=self.n0, n1=float(n1)
        )

    def mean_concrete(self) -> GaussianLensSlowness:
        lo, hi = self._bounds()
        return self.to_concrete(0.5 * (lo + hi))

    def get_slowness(self) -> Callable[[Tensor], Tensor]:
        return self.mean_concrete().get_slowness()

    def get_slowness_gradient(self) -> Callable[[Tensor], Tensor]:
        return self.mean_concrete().get_slowness_gradient()


class CompositeGaussianLensFamilySlowness(_ConditionedFamily):
    """A family of **multi-lens** media: each medium is a sum of ``n_lenses``
    Gaussian lenses, each lens drawn from the same per-lens ranges. The flat
    descriptor is ``n_lenses`` single-lens descriptors concatenated. Use with
    field encoder to handle arbitrary number of lenses."""

    type: Literal["composite_gaussian_lens_family"] = "composite_gaussian_lens_family"
    n0: float
    n_lenses: int
    n1_range: tuple[float, float]
    center_ranges: list[tuple[float, float]]
    sigma_ranges: list[tuple[float, float]]
    n_probe: int = 0
    held_out: list[list[float]] = []

    @field_validator("sigma_ranges")
    @classmethod
    def _sig_len(cls, sigma_ranges, info):
        center_ranges = info.data.get("center_ranges")
        if center_ranges is not None and len(sigma_ranges) != len(center_ranges):
            raise ValueError("sigma_ranges length must match center_ranges length")
        return sigma_ranges

    @property
    def dim(self) -> int:
        return len(self.center_ranges)

    @property
    def per_lens_dim(self) -> int:
        return 1 + 2 * self.dim

    @property
    def param_dim(self) -> int:
        return self.n_lenses * self.per_lens_dim

    @property
    def cond_dim(self) -> int:
        return self.n_probe**self.dim if self.n_probe > 0 else self.param_dim

    def _single_bounds(self):
        lo = (
            [self.n1_range[0]]
            + [c[0] for c in self.center_ranges]
            + [s[0] for s in self.sigma_ranges]
        )
        hi = (
            [self.n1_range[1]]
            + [c[1] for c in self.center_ranges]
            + [s[1] for s in self.sigma_ranges]
        )
        return lo, hi

    def _bounds(self, device=None, dtype=torch.get_default_dtype()):
        lo1, hi1 = self._single_bounds()
        return (
            torch.tensor(lo1 * self.n_lenses, device=device, dtype=dtype),
            torch.tensor(hi1 * self.n_lenses, device=device, dtype=dtype),
        )

    def sample_params(
        self, n, *, generator=None, device=None, dtype=torch.get_default_dtype()
    ):
        lo, hi = self._bounds(device=device, dtype=dtype)
        u = torch.rand(
            n, self.param_dim, generator=generator, device=device, dtype=dtype
        )
        return lo + u * (hi - lo)

    def _lens_split(self, params):
        """``params (N, n_lenses*per_lens)`` -> ``n1 (N,K)``, ``center (N,K,d)``, ``sigma (N,K,d)``."""
        d, K = self.dim, self.n_lenses
        p = params.reshape(*params.shape[:-1], K, self.per_lens_dim)
        return p[..., 0], p[..., 1 : 1 + d], p[..., 1 + d : 1 + 2 * d]

    def scale_contrast(self, params, scale: float):
        """Scale every lens' amplitude ``n1`` by ``scale``."""
        if scale == 1.0:
            return params
        out = params.clone()
        out[..., 0 :: self.per_lens_dim] = out[..., 0 :: self.per_lens_dim] * scale
        return out

    def slowness_at(self, x, params):
        n1, center, sigma = self._lens_split(params)  # (N,K),(N,K,d),(N,K,d)
        xx = x.unsqueeze(-2)  # (N,1,d)
        per = gaussian_lens_perturbation(xx, center, sigma, n1)  # (N,K)
        return self.n0 + per.sum(dim=-1)

    def slowness_grad_at(self, x, params):
        n1, center, sigma = self._lens_split(params)
        xx = x.unsqueeze(-2)  # (N,1,d)
        # per-lens grad summed: -pert_k·(x-c_k)/σ_k²
        bump = gaussian_lens_perturbation(xx, center, sigma, n1).unsqueeze(
            -1
        )  # (N,K,1)
        grad = (-bump * (xx - center) / (sigma**2)).sum(dim=-2)  # (N,d)
        return grad

    def probe_points(self, size, device=None, dtype=torch.get_default_dtype()):
        axes = [
            torch.linspace(0.0, float(L), self.n_probe, device=device, dtype=dtype)
            for L in size
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
            -1, len(size)
        )

    def cond_for(self, size, params):
        if params.dim() != 2:
            raise ValueError("cond_for expects params (N, param_dim)")
        if self.n_probe > 0:
            pts = self.probe_points(
                size, device=params.device, dtype=params.dtype
            )  # (G,d)
            n1, center, sigma = self._lens_split(params)  # (N,K),(N,K,d),(N,K,d)
            total = pts.new_zeros(params.shape[0], pts.shape[0])
            for k in range(self.n_lenses):
                total = total + gaussian_lens_perturbation(
                    pts.unsqueeze(0),
                    center[:, k].unsqueeze(1),
                    sigma[:, k].unsqueeze(1),
                    n1[:, k].unsqueeze(1),
                )
            return total
        lo, hi = self._bounds(device=params.device, dtype=params.dtype)
        span = hi - lo  # guard fixed dims (hi==lo): 0/0 -> NaN conditioning
        safe = torch.where(span > 0, span, torch.ones_like(span))
        normed = 2.0 * (params - lo) / safe - 1.0
        return torch.where(span > 0, normed, torch.zeros_like(normed))

    def sample_cond(
        self, size, n, *, generator=None, device=None, dtype=torch.get_default_dtype()
    ):
        return self.cond_for(
            size, self.sample_params(n, generator=generator, device=device, dtype=dtype)
        )

    def to_concrete(self, params) -> CompositeGaussianLensSlowness:
        p = (
            params
            if isinstance(params, Tensor)
            else torch.tensor(params, dtype=torch.get_default_dtype())
        )
        n1, center, sigma = self._lens_split(p)
        lenses = [
            GaussianLensSlowness(
                center=center[k].tolist(),
                sigma=sigma[k].tolist(),
                n0=self.n0,
                n1=float(n1[k]),
            )
            for k in range(self.n_lenses)
        ]
        return CompositeGaussianLensSlowness(n0=self.n0, lenses=lenses)

    def get_slowness(self) -> Callable[[Tensor], Tensor]:
        lo, hi = self._bounds()
        return self.to_concrete(0.5 * (lo + hi)).get_slowness()

    def get_slowness_gradient(self) -> Callable[[Tensor], Tensor]:
        lo, hi = self._bounds()
        return self.to_concrete(0.5 * (lo + hi)).get_slowness_gradient()


SlownessSpecification = Annotated[
    Union[
        ConstantSlowness,
        GaussianLensSlowness,
        CompositeGaussianLensSlowness,
        GaussianLensFamilySlowness,
        CompositeGaussianLensFamilySlowness,
    ],
    Field(discriminator="type"),
]
