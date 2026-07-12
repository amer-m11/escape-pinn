from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from pinn_solver.networks.base_network import EscapeNet2D

from .boundary import (
    compute_boundary_loss_2d,
    compute_boundary_loss_3d,
    compute_boundary_loss_cartesian_3d,
    sample_boundary_points_2d,
    sample_boundary_points_3d,
    sample_boundary_points_cartesian_3d,
)
from .configuration import (
    BoundarySpecification,
    LossConfiguration,
    LossType,
    PhysicalSize,
)
from .configuration.slowness import GaussianLensFamilySlowness
from .configuration.training import full_data_channels
from .data import ReferenceSolution, ReferenceSolution3D
from .networks import EscapeNet
from .pde import (
    compute_pde_residual_2d,
    compute_pde_residual_3d,
    compute_pde_residual_cartesian_3d,
    consistency_residual_2d,
    consistency_residual_3d,
    consistency_residual_cartesian_3d,
)
from .sampler import Samples2D, Samples3D, Samples3DCartesian


class LossFunction(ABC):
    """Abstract base for loss functions.

    Allows the loss function to be swapped out for experimentation.

    Currently only MSE is implemented."""

    @abstractmethod
    def __call__(self, pred, target) -> torch.Tensor:
        pass


class MSELoss(LossFunction):
    """Mean squared error loss function."""

    def __call__(self, pred, target) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


class LossFactory:
    """Factory for creating loss functions based on configuration."""

    @staticmethod
    def create_loss_function(type: LossType) -> LossFunction:
        if type == "mse":
            return MSELoss()
        raise ValueError(f"Unsupported loss type: {type}")


@dataclass
class LossBreakdown:
    """Loss for a single training step."""

    total: torch.Tensor
    """Differentiable total loss scalar tensor. Backprop is called on this."""
    pde: float
    """PDE residual loss component (detached)."""
    bc: float
    """Boundary condition loss component (detached)."""
    data: float
    """Data loss component (detached)."""
    circle: float = 0.0
    """Circle regularizer loss component (detached)."""
    consistency: float = 0.0
    """Characteristic self-consistency loss component (detached)."""
    bc_components: dict[str, float] | None = None
    """Per-channel BC loss breakdown per training step (detached). 
    6 keys in 2D, 9 in 3D."""


class Loss:
    """PINN loss assembly: PDE residual + BC + (optional) data + (optional)
    circle regularizer.

    Dispatches on ``physical_size.dim`` so the same class serves both 2D
    and 3D paths. Holds the pre-sampled boundary set. The
    :meth:`resample_boundary` hook lets the Adam trainer re-draw it each
    epoch when ``boundary.resample`` is True (L-BFGS keeps the BC frozen
    to avoid line-search thrashing).
    """

    def __init__(
        self,
        config: LossConfiguration,
        boundary: BoundarySpecification,
        physical_size: PhysicalSize,
        device: torch.device,
        sampling_rng: torch.Generator | None = None,
        parametrization: Literal["spherical", "cartesian"] = "spherical",
        family: GaussianLensFamilySlowness | None = None,
    ):
        """
        Parameters
        ----------
        sampling_rng
            The generator is only consumed when ``boundary.sample_mode ==
            "random"``. The "grid" branch is fully deterministic regardless.
        parametrization
            3D momentum parametrization. Selects the Cartesian PDE/BC path in 3D.
            Ignored in 2D.
        family
            Optional :class:`GaussianLensFamilySlowness` (generalization).
            When set, the interior residual evaluates the per-point ``n(x; params)``
            from ``collocation.cond`` and the boundary points draw their own
            conditioning. ``None`` (default) => single fixed medium.
        """
        self._family = family
        self.loss_fn = LossFactory.create_loss_function(config.function)
        self.device = device
        self.config = config
        self.boundary = boundary
        self.size = physical_size
        self.parametrization = parametrization
        self._cartesian = physical_size.dim == 3 and parametrization == "cartesian"
        self._sampling_rng = sampling_rng

        # Current vanishing-viscosity ε (the trainer anneals this in place when
        # ``viscosity_anneal`` is set. Otherwise, it stays at the configured value).
        self._viscosity_eps_current = float(getattr(config, "viscosity_eps", 0.0))
        if self.config.data_channels is None:
            self.config.data_channels = full_data_channels(
                physical_size.dim, parametrization
            )
        self._sample_boundary()

    # ------------------------------------------------------------------
    # Boundary sampling
    # ------------------------------------------------------------------
    def _draw_boundary_cond(self) -> torch.Tensor | None:
        """Per-boundary-point normalized conditioning vector for a family, else None.

        The BC targets are medium-independent. This only tells the network which
        medium each boundary point belongs to (matching the interior conditioning)."""
        if self._family is None:
            return None
        n_bc = self._bc[0].shape[0]
        gen = self._sampling_rng if self.boundary.sample_mode == "random" else None
        return self._family.sample_cond(
            self.size.as_tuple(), n_bc, generator=gen, device=self.device
        )

    def _resolve_conditioning(self, coll, slowness_fn, slowness_grad_fn):
        """For a conditioned batch, return ``(cond, slowness_fn, slowness_grad_fn)``
        with the slowness callables rebound to evaluate the per-point ``n(x; params)``
        from the batch's physical descriptors ``coll.params`` (works in both param
        and probe-grid conditioning modes). Single-medium => ``(None, fn, grad)``."""
        cond = getattr(coll, "cond", None)
        params = getattr(coll, "params", None)
        if cond is not None and params is not None and self._family is not None:
            slowness_fn = lambda xx: self._family.slowness_at(xx, params)
            slowness_grad_fn = lambda xx: self._family.slowness_grad_at(xx, params)
        return cond, slowness_fn, slowness_grad_fn

    def _sample_boundary(self) -> None:
        """(Re)draw the boundary collocation set with dimensions based on
        dim/parametrization of the configuration.

        The drawn set is stored in ``self._bc`` and the per-point conditioning
        vector in ``self._bc_cond``. The latter is ``None`` for a single fixed
        medium.

        NOTE: In the cartesian formulation, n_phi is rounded up to the next odd
        integer to ensure that the equator (phi=pi/2) is sampled.
        """
        gen = self._sampling_rng if self.boundary.sample_mode == "random" else None
        if self._cartesian:
            n_phi_raw = int(self.boundary.phi_samples_at_boundary)
            # rounded up to the next odd integer
            n_phi_grid = n_phi_raw if n_phi_raw % 2 == 1 else n_phi_raw + 1
            self._bc = sample_boundary_points_cartesian_3d(
                self.size.as_tuple(),
                n_per_long_edge=self.boundary.physical_boundary_samples,
                n_phi_grid=n_phi_grid,
                n_theta_grid=int(self.boundary.theta_samples_at_boundary),
                device=self.device,
                mode=self.boundary.sample_mode,
                generator=gen,
            )
            self._bc_cond = self._draw_boundary_cond()
            return
        if self.size.dim == 3:
            self._bc = sample_boundary_points_3d(
                self.size.as_tuple(),
                n_per_long_edge=self.boundary.physical_boundary_samples,
                n_theta=self.boundary.theta_samples_at_boundary,
                n_phi=self.boundary.phi_samples_at_boundary,
                device=self.device,
                mode=self.boundary.sample_mode,
                generator=gen,
                phi_delta=self.boundary.resolved_phi_delta(),
            )
        else:
            self._bc = sample_boundary_points_2d(
                self.size.as_tuple(),
                n_per_long_edge=self.boundary.physical_boundary_samples,
                n_theta=self.boundary.theta_samples_at_boundary,
                device=self.device,
                mode=self.boundary.sample_mode,
                generator=gen,
            )
        self._bc_cond = self._draw_boundary_cond()

    def resample_boundary(self) -> None:
        """Re-draw the boundary collocation set if ``boundary.resample`` is on.

        A no-op when the BC is in grid mode or resampling is disabled.
        """
        if self.boundary.resample and self.boundary.sample_mode == "random":
            self._sample_boundary()

    # ------------------------------------------------------------------
    # Main loss assembly
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        model: EscapeNet,
        collocation: Samples2D | Samples3D | Samples3DCartesian,
        slowness_fn: Callable,
        slowness_grad_fn: Callable | None,
        reference: ReferenceSolution | ReferenceSolution3D | None,
    ) -> LossBreakdown:
        """Evaluate the total loss and its components.

        This function builds a full autograd graph for the PDE residuals.
        It must not be called inside ``torch.no_grad()``.
        """
        if self._cartesian:
            assert isinstance(collocation, Samples3DCartesian)
            return self._compute_cartesian_3d(
                model, collocation, slowness_fn, slowness_grad_fn, reference
            )
        if self.size.dim == 3:
            assert isinstance(collocation, Samples3D)
            return self._compute_3d(
                model, collocation, slowness_fn, slowness_grad_fn, reference
            )
        assert isinstance(collocation, Samples2D)
        return self._compute_2d(
            model, collocation, slowness_fn, slowness_grad_fn, reference
        )

    def pointwise_pde_residual(
        self,
        model: EscapeNet,
        pool: Samples2D | Samples3D | Samples3DCartesian,
        slowness_fn: Callable,
        slowness_grad_fn: Callable | None,
    ) -> torch.Tensor:
        """Per-point ``sqrt(Σ_c res_c²)`` of the PDE residual (detached).

        Used by residual-adaptive resampling to score a candidate pool and
        decide which points to use for the next training step.

        Returns a detached 1-D tensor of length ``len(pool)``.
        """
        cond, slowness_fn, slowness_grad_fn = self._resolve_conditioning(
            pool, slowness_fn, slowness_grad_fn
        )
        if self._cartesian:
            res = compute_pde_residual_cartesian_3d(
                model,
                pool.x1,
                pool.x2,
                pool.x3,
                pool.d1,
                pool.d2,
                pool.d3,
                slowness_fn,
                slowness_grad_fn,
                cond=cond,
            )
        elif self.size.dim == 3:
            res = compute_pde_residual_3d(
                model,
                pool.x1,
                pool.x2,
                pool.x3,
                pool.phi,
                pool.theta,
                slowness_fn,
                slowness_grad_fn,
                pole_safe=self.config.pole_safe_residual,
                cond=cond,
            )
        else:
            res = compute_pde_residual_2d(
                model, pool.x1, pool.x2, pool.theta, slowness_fn, slowness_grad_fn
            )
        score = sum(r.detach() ** 2 for r in res).sqrt()
        return score

    # ------------------------------------------------------------------
    # Second-order / causality residual levers (3D only; default-off)
    # ------------------------------------------------------------------
    @staticmethod
    def _laplacian_of(f: torch.Tensor, coords) -> torch.Tensor:
        """Spatial Laplacian ``Σ_i ∂²f/∂x_i²`` of a per-point output ``f`` w.r.t.
        the spatial coords. ``create_graph`` keeps it differentiable for backprop.
        """
        lap = torch.zeros_like(f)
        for xi in coords:
            g = torch.autograd.grad(f.sum(), xi, create_graph=True)[0]
            g2 = torch.autograd.grad(g.sum(), xi, create_graph=True)[0]
            lap = lap + g2
        return lap

    def _fwd(self, model, base, coll):
        """Call ``model`` with base parameters ``base`` and conditioning if
        available in ``coll``."""
        cond = getattr(coll, "cond", None)
        return model(*base) if cond is None else model(*base, cond)

    def _spatial_laplacian(self, model, coll):
        """``(∇²û, ∇²σ̂)`` per collocation point."""
        x1 = coll.x1.detach().requires_grad_(True)
        x2 = coll.x2.detach().requires_grad_(True)
        x3 = coll.x3.detach().requires_grad_(True)
        if self._cartesian:
            base = (x1, x2, x3, coll.d1.detach(), coll.d2.detach(), coll.d3.detach())
        else:
            base = (x1, x2, x3, coll.phi.detach(), coll.theta.detach())
        out = self._fwd(model, base, coll)
        coords = (x1, x2, x3)
        return self._laplacian_of(out[0], coords), self._laplacian_of(out[1], coords)

    def _causality_sqrt_weight(self, model, coll, residuals):
        """Detached per-point ``√(exp(-ε·M_i))`` causality weight (Wang 2022). ``M_i``
        = cumulative residual of points with smaller predicted σ̂ (causally upstream,
        nearer the exit boundary). The weights are normalized to [0,1] so the gate
        self-relaxes to ~uniform as residuals fall during training. The gate is a
        no-op when ``causality_eps==0`` or in 2D. Returns a 1-D tensor of length
        ``len(coll)``."""

        eps = float(getattr(self.config, "causality_eps", 0.0))
        if eps <= 0.0 or self.size.dim != 3:
            return 1.0
        with torch.no_grad():
            if self._cartesian:
                base = (coll.x1, coll.x2, coll.x3, coll.d1, coll.d2, coll.d3)
            else:
                base = (coll.x1, coll.x2, coll.x3, coll.phi, coll.theta)
            sigma = self._fwd(model, base, coll)[1]
        res_mag = sum(r.detach() ** 2 for r in residuals).sqrt()
        order = torch.argsort(sigma.detach())
        sorted_res = res_mag[order]
        cum_upstream = torch.cumsum(sorted_res, 0) - sorted_res  # exclusive prefix
        M = torch.empty_like(cum_upstream)
        M[order] = cum_upstream
        # Normalize the cumulative residual to [0,1]
        M = M / (M.max() + 1e-12)
        return torch.sqrt(torch.exp(-eps * M))

    def _apply_3d_residual_levers(self, model, coll, residuals):
        """Compose the 3D PDE-residual levers and return ``loss_pde``.

        Applies (in order): the vanishing-viscosity correction ``r_{u,σ} -= ε∇²``
        and the per-point causality √-weight. Both are no-ops at their default
        config values."""
        eps_visc = float(self._viscosity_eps_current)
        if eps_visc > 0.0:
            lap_u, lap_sigma = self._spatial_laplacian(model, coll)
            residuals = (
                residuals[0] - eps_visc * lap_u,
                residuals[1] - eps_visc * lap_sigma,
                *residuals[2:],
            )
        caus = self._causality_sqrt_weight(model, coll, residuals)
        loss_pde = sum(self.loss_fn(r * caus, 0.0) for r in residuals)
        return loss_pde

    def loss_terms(
        self,
        model: EscapeNet,
        coll: Samples2D | Samples3D | Samples3DCartesian,
        slowness_fn: Callable,
        slowness_grad_fn: Callable | None,
    ) -> dict[str, torch.Tensor]:
        """Unweighted PDE and BC loss tensors with autograd graph
        (``{"pde": …, "bc": …}``).

        Used by grad-norm adaptive weighting to measure each term's parameter
        gradient norm separately and balance their weights accordingly."""
        if self._cartesian:
            res = compute_pde_residual_cartesian_3d(
                model,
                coll.x1,
                coll.x2,
                coll.x3,
                coll.d1,
                coll.d2,
                coll.d3,
                slowness_fn,
                slowness_grad_fn,
            )
            bc = compute_boundary_loss_cartesian_3d(
                model, self.size.as_tuple(), *self._bc
            )
        elif self.size.dim == 3:
            res = compute_pde_residual_3d(
                model,
                coll.x1,
                coll.x2,
                coll.x3,
                coll.phi,
                coll.theta,
                slowness_fn,
                slowness_grad_fn,
                pole_safe=self.config.pole_safe_residual,
            )
            bc = compute_boundary_loss_3d(model, self.size.as_tuple(), *self._bc)
        else:
            res = compute_pde_residual_2d(
                model, coll.x1, coll.x2, coll.theta, slowness_fn, slowness_grad_fn
            )
            bc = compute_boundary_loss_2d(model, self.size.as_tuple(), *self._bc)
        loss_pde = sum(self.loss_fn(r, 0.0) for r in res)
        return {"pde": loss_pde, "bc": bc.total}

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------
    def _compute_2d(
        self,
        model: EscapeNet2D,
        coll: Samples2D,
        slowness_fn: Callable,
        slowness_grad_fn: Callable | None,
        reference: ReferenceSolution | None,
    ) -> LossBreakdown:
        # PDE residual: six escape-equation residuals.
        res_u, res_sigma, res_y1, res_y2, res_cos, res_sin = compute_pde_residual_2d(
            model,
            coll.x1,
            coll.x2,
            coll.theta,
            slowness_fn,
            slowness_grad_fn,
        )
        loss_pde = (
            self.loss_fn(res_u, 0.0)
            + self.loss_fn(res_sigma, 0.0)
            + self.loss_fn(res_y1, 0.0)
            + self.loss_fn(res_y2, 0.0)
            + self.loss_fn(res_cos, 0.0)
            + self.loss_fn(res_sin, 0.0)
        )

        x1_bc, x2_bc, th_bc = self._bc
        bc = compute_boundary_loss_2d(model, self.size.as_tuple(), x1_bc, x2_bc, th_bc)
        loss_bc = bc.total

        total = self.config.lambda_pde * loss_pde + self.config.lambda_bc * loss_bc

        # Circle regularizer (θ-circle only in 2D).
        loss_circle = torch.tensor(0.0, device=self.device)
        if self.config.lambda_circle > 0:
            _, _, _, _, c_th, s_th = model(coll.x1, coll.x2, coll.theta)
            loss_circle = ((c_th**2 + s_th**2 - 1.0) ** 2).mean()
            total = total + self.config.lambda_circle * loss_circle

        # Characteristic consistency (finite difference).
        loss_consistency = torch.tensor(0.0, device=self.device)
        if self.config.lambda_consistency > 0:
            cres = consistency_residual_2d(
                model,
                coll.x1,
                coll.x2,
                coll.theta,
                slowness_fn,
                slowness_grad_fn,
                self.config.consistency_dsigma,
            )
            loss_consistency = sum(self.loss_fn(r, 0.0) for r in cres)
            total = total + self.config.lambda_consistency * loss_consistency

        # Supervised data MSE on the selected channels (2D reference).
        loss_data = torch.tensor(0.0, device=self.device)
        if reference is not None:
            u_pred, sigma_pred, y1_pred, y2_pred, c_pred, s_pred = model(
                reference.x1, reference.x2, reference.theta
            )
            channel_terms = {
                "u": (u_pred, reference.u),
                "sigma": (sigma_pred, reference.sigma),
                "y1": (y1_pred, reference.y1),
                "y2": (y2_pred, reference.y2),
                "cos": (c_pred, reference.cos_theta_exit),
                "sin": (s_pred, reference.sin_theta_exit),
            }
            for ch in self.config.data_channels:
                pred, tgt = channel_terms[ch]
                loss_data = loss_data + self.loss_fn(pred, tgt)
            total = total + self.config.lambda_data * loss_data

        return LossBreakdown(
            total=total,
            pde=loss_pde.item(),
            bc=loss_bc.item(),
            data=loss_data.item(),
            circle=float(loss_circle.item()),
            consistency=float(loss_consistency.item()),
            bc_components={
                name: float(getattr(bc, name).item())
                for name in ("u", "sigma", "y1", "y2", "cos", "sin")
            },
        )

    # ------------------------------------------------------------------
    # 3D polar momentum
    # ------------------------------------------------------------------
    def _compute_3d(
        self,
        model: EscapeNet,
        coll: Samples3D,
        slowness_fn: Callable,
        slowness_grad_fn: Callable | None,
        reference: ReferenceSolution3D | None,
    ) -> LossBreakdown:
        if reference is not None and not isinstance(reference, ReferenceSolution3D):
            raise TypeError(
                "3D Loss requires a ReferenceSolution3D; got "
                f"{type(reference).__name__}"
            )

        cond, slowness_fn, slowness_grad_fn = self._resolve_conditioning(
            coll, slowness_fn, slowness_grad_fn
        )

        residuals = compute_pde_residual_3d(
            model,
            coll.x1,
            coll.x2,
            coll.x3,
            coll.phi,
            coll.theta,
            slowness_fn,
            slowness_grad_fn,
            pole_safe=self.config.pole_safe_residual,
            cond=cond,
        )
        loss_pde = self._apply_3d_residual_levers(model, coll, residuals)

        x1_bc, x2_bc, x3_bc, phi_bc, th_bc = self._bc
        bc = compute_boundary_loss_3d(
            model,
            self.size.as_tuple(),
            x1_bc,
            x2_bc,
            x3_bc,
            phi_bc,
            th_bc,
            cond=getattr(self, "_bc_cond", None),
        )
        loss_bc = bc.total

        total = self.config.lambda_pde * loss_pde + self.config.lambda_bc * loss_bc

        # Circle regularizer (θ-circle + φ-circle in 3D).
        loss_circle = torch.tensor(0.0, device=self.device)
        if self.config.lambda_circle > 0:
            _ca = (coll.x1, coll.x2, coll.x3, coll.phi, coll.theta)
            _, _, _, _, _, c_th, s_th, c_ph, s_ph = (
                model(*_ca) if cond is None else model(*_ca, cond)
            )
            theta_drift = ((c_th**2 + s_th**2 - 1.0) ** 2).mean()
            phi_drift = ((c_ph**2 + s_ph**2 - 1.0) ** 2).mean()
            loss_circle = theta_drift + phi_drift
            total = total + self.config.lambda_circle * loss_circle

        loss_consistency = torch.tensor(0.0, device=self.device)
        if self.config.lambda_consistency > 0:
            cres = consistency_residual_3d(
                model,
                coll.x1,
                coll.x2,
                coll.x3,
                coll.phi,
                coll.theta,
                slowness_fn,
                slowness_grad_fn,
                self.config.consistency_dsigma,
            )
            loss_consistency = sum(self.loss_fn(r, 0.0) for r in cres)
            total = total + self.config.lambda_consistency * loss_consistency

        loss_data = torch.tensor(0.0, device=self.device)
        if reference is not None:
            (
                u_pred,
                sigma_pred,
                y1_pred,
                y2_pred,
                y3_pred,
                cth_pred,
                sth_pred,
                cph_pred,
                sph_pred,
            ) = model(
                reference.x1,
                reference.x2,
                reference.x3,
                reference.phi,
                reference.theta,
            )
            # 3D channel filter: ``cfg.loss.data_channels`` picks which of
            # the 9 channels contribute to the supervised MSE.
            channel_terms = {
                "u": (u_pred, reference.u),
                "sigma": (sigma_pred, reference.sigma),
                "y1": (y1_pred, reference.y1),
                "y2": (y2_pred, reference.y2),
                "y3": (y3_pred, reference.y3),
                "cos_theta": (cth_pred, reference.cos_theta_exit),
                "sin_theta": (sth_pred, reference.sin_theta_exit),
                "cos_phi": (cph_pred, reference.cos_phi_exit),
                "sin_phi": (sph_pred, reference.sin_phi_exit),
            }
            for ch in self.config.data_channels:
                if ch not in channel_terms:
                    continue
                pred, tgt = channel_terms[ch]
                loss_data = loss_data + self.loss_fn(pred, tgt)
            total = total + self.config.lambda_data * loss_data

        return LossBreakdown(
            total=total,
            pde=loss_pde.item(),
            bc=loss_bc.item(),
            data=loss_data.item(),
            circle=float(loss_circle.item()),
            consistency=float(loss_consistency.item()),
            bc_components={
                name: float(getattr(bc, name).item())
                for name in (
                    "u",
                    "sigma",
                    "y1",
                    "y2",
                    "y3",
                    "cos_theta",
                    "sin_theta",
                    "cos_phi",
                    "sin_phi",
                )
            },
        )

    # ------------------------------------------------------------------
    # 3D Cartesian momentum
    # ------------------------------------------------------------------
    def _compute_cartesian_3d(
        self,
        model: EscapeNet,
        coll: Samples3DCartesian,
        slowness_fn: Callable,
        slowness_grad_fn: Callable | None,
        reference: ReferenceSolution3D | None,
    ) -> LossBreakdown:
        if reference is not None and not isinstance(reference, ReferenceSolution3D):
            raise TypeError(
                "Cartesian 3D Loss requires a ReferenceSolution3D; got "
                f"{type(reference).__name__}"
            )

        cond, slowness_fn, slowness_grad_fn = self._resolve_conditioning(
            coll, slowness_fn, slowness_grad_fn
        )

        residuals = compute_pde_residual_cartesian_3d(
            model,
            coll.x1,
            coll.x2,
            coll.x3,
            coll.d1,
            coll.d2,
            coll.d3,
            slowness_fn,
            slowness_grad_fn,
            cond=cond,
        )

        loss_pde = self._apply_3d_residual_levers(model, coll, residuals)

        x1_bc, x2_bc, x3_bc, d1_bc, d2_bc, d3_bc = self._bc
        bc = compute_boundary_loss_cartesian_3d(
            model,
            self.size.as_tuple(),
            x1_bc,
            x2_bc,
            x3_bc,
            d1_bc,
            d2_bc,
            d3_bc,
            cond=getattr(self, "_bc_cond", None),
        )
        loss_bc = bc.total

        total = self.config.lambda_pde * loss_pde + self.config.lambda_bc * loss_bc

        # Sphere regularizer: the |d̂|=1 analogue of the 2D unit-circle reg.
        loss_circle = torch.tensor(0.0, device=self.device)
        if self.config.lambda_circle > 0:
            _ca = (coll.x1, coll.x2, coll.x3, coll.d1, coll.d2, coll.d3)
            _, _, _, _, _, e1, e2, e3 = (
                model(*_ca) if cond is None else model(*_ca, cond)
            )
            loss_circle = ((e1**2 + e2**2 + e3**2 - 1.0) ** 2).mean()
            total = total + self.config.lambda_circle * loss_circle

        loss_consistency = torch.tensor(0.0, device=self.device)
        if self.config.lambda_consistency > 0:
            cres = consistency_residual_cartesian_3d(
                model,
                coll.x1,
                coll.x2,
                coll.x3,
                coll.d1,
                coll.d2,
                coll.d3,
                slowness_fn,
                slowness_grad_fn,
                self.config.consistency_dsigma,
            )
            loss_consistency = sum(self.loss_fn(r, 0.0) for r in cres)
            total = total + self.config.lambda_consistency * loss_consistency

        loss_data = torch.tensor(0.0, device=self.device)
        if reference is not None:
            # Reference inputs come as (φ, θ). Convert to a unit direction
            # for the model, and the exit (φ̂, θ̂) target to a unit vector d̂.
            sin_phi_in = torch.sin(reference.phi)
            d1_in = sin_phi_in * torch.cos(reference.theta)
            d2_in = sin_phi_in * torch.sin(reference.theta)
            d3_in = torch.cos(reference.phi)
            (
                u_pred,
                sigma_pred,
                y1_pred,
                y2_pred,
                y3_pred,
                e1_pred,
                e2_pred,
                e3_pred,
            ) = model(reference.x1, reference.x2, reference.x3, d1_in, d2_in, d3_in)
            d1_tgt = reference.sin_phi_exit * reference.cos_theta_exit
            d2_tgt = reference.sin_phi_exit * reference.sin_theta_exit
            d3_tgt = reference.cos_phi_exit
            channel_terms = {
                "u": (u_pred, reference.u),
                "sigma": (sigma_pred, reference.sigma),
                "y1": (y1_pred, reference.y1),
                "y2": (y2_pred, reference.y2),
                "y3": (y3_pred, reference.y3),
                "d1": (e1_pred, d1_tgt),
                "d2": (e2_pred, d2_tgt),
                "d3": (e3_pred, d3_tgt),
            }
            for ch in self.config.data_channels:
                if ch not in channel_terms:
                    continue
                pred, tgt = channel_terms[ch]
                loss_data = loss_data + self.loss_fn(pred, tgt)
            total = total + self.config.lambda_data * loss_data

        return LossBreakdown(
            total=total,
            pde=loss_pde.item(),
            bc=loss_bc.item(),
            data=loss_data.item(),
            circle=float(loss_circle.item()),
            consistency=float(loss_consistency.item()),
            bc_components={
                name: float(getattr(bc, name).item())
                for name in ("u", "sigma", "y1", "y2", "y3", "d1", "d2", "d3")
            },
        )
