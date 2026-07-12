import dataclasses
import math

import torch

from ..configuration import AdamConfiguration, PhysicalSize, SlownessFn, SlownessGradFn
from ..data import ReferenceSolution
from ..loss import Loss
from ..networks import EscapeNet
from ..sampler import Sampler
from .history import TrainHistory


def _index_samples(samples, idx: torch.Tensor):
    """Re-index every tensor field of a ``Samples*`` dataclass:
        - preserve ``None``
        - Re-index according to ``idx`` and detach (no autograd graph)
        - For coordinate fields, additionally set ``requires_grad=True``
        (fresh leaves) so the PDE residual can be differentiated w.r.t.
        the network parameters.

    Return a new dataclass instance of the same type as ``samples``.
    """
    kw = {}
    for f in dataclasses.fields(samples):
        v = getattr(samples, f.name)
        if v is None:
            kw[f.name] = None
        elif f.name in ("cond", "params"):
            kw[f.name] = v[idx].detach()
        else:
            kw[f.name] = v[idx].detach().requires_grad_(True)
    return type(samples)(**kw)


class AdamTrainer:
    config: AdamConfiguration
    loss: Loss
    size: PhysicalSize
    device: torch.device
    sampling_rng: torch.Generator | None
    """Random number generator for sampling. When ``None`` the global
    default RNG is used (seeded by `torch.manual_seed` or similar)."""
    phi_delta: float
    """φ safe-band for 3D interior sampling (ignored in 2D)."""
    parametrization: str
    """3D momentum parametrization ('spherical' | 'cartesian')."""
    family: object | None
    """Family object for multi-medium training (or ``None`` for single-medium)."""

    def __init__(
        self,
        config: AdamConfiguration,
        loss: Loss,
        size: PhysicalSize,
        device: torch.device,
        sampling_rng: torch.Generator | None = None,
        phi_delta: float = math.pi / 8,
        parametrization: str = "spherical",
        family=None,
    ):
        self.config = config
        self.loss = loss
        self.size = size
        self.device = device
        self.sampling_rng = sampling_rng
        self.phi_delta = phi_delta
        self.parametrization = parametrization
        self.family = family

    def train(
        self,
        model: EscapeNet,
        history: TrainHistory,
        slowness_fn: SlownessFn,
        slowness_grad_fn: SlownessGradFn,
        reference: ReferenceSolution | None = None,
        slowness_spec=None,
    ) -> dict:
        """Run the Adam phase. Returns a stats dict for diagnostics.

        ``slowness_spec`` (the config slowness object) is consumed only by the
        lens amplitude curriculum (``lens_curriculum>0``). It is used to scale
        the lens amplitude during training for single medium scenarios."""
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
        )
        scheduler = None
        if getattr(self.config, "lr_decay", None) == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr * getattr(self.config, "lr_decay_floor", 0.01),
            )

        model.train()

        samples = Sampler.sample_interior(
            self.config.collocation_samples,
            self.size,
            self.device,
            mode=self.config.sample_mode,
            generator=self.sampling_rng,
            phi_delta=self.phi_delta,
            parametrization=self.parametrization,
            family=self.family,
        )

        rar = getattr(self.config, "rar", False)
        rar_every = getattr(self.config, "rar_every", 1000)
        adaptive = getattr(self.config, "adaptive_weighting", False)
        adaptive_every = getattr(self.config, "adaptive_every", 1000)
        visc_eps0 = float(getattr(self.loss.config, "viscosity_eps", 0.0))
        visc_anneal = getattr(self.loss.config, "viscosity_anneal", "none")
        curr_frac = float(getattr(self.config, "lens_curriculum", 0.0))
        total_epochs = max(1, self.config.epochs - 1)

        for epoch in range(1, self.config.epochs + 1):
            optimizer.zero_grad()
            training_progress = (epoch - 1) / total_epochs
            # Curriculum scale: ease the lens contrast 0 -> 1 over the first
            # ``curr_frac`` of the phase (constant medium first, then full lens).
            curr_scale = (
                min(1.0, training_progress / curr_frac) if curr_frac > 0.0 else 1.0
            )
            if self.config.resample:
                samples = Sampler.sample_interior(
                    self.config.collocation_samples,
                    self.size,
                    self.device,
                    mode=self.config.sample_mode,
                    generator=self.sampling_rng,
                    phi_delta=self.phi_delta,
                    parametrization=self.parametrization,
                    family=self.family,
                    medium_scale=(curr_scale if self.family is not None else 1.0),
                )
            if rar and (epoch == 1 or epoch % rar_every == 0):
                # Residual-adaptive resampling
                samples = self._rar_select(model, slowness_fn, slowness_grad_fn)
            self.loss.resample_boundary()  # no-op when resampling is disabled

            if visc_eps0 > 0.0 and visc_anneal == "cosine":
                self.loss._viscosity_eps_current = (
                    visc_eps0 * 0.5 * (1.0 + math.cos(math.pi * training_progress))
                )
            if adaptive and (epoch == 1 or epoch % adaptive_every == 0):
                # Grad-norm adaptive loss balancing: rescale λ_pde/λ_bc to equalize their gradient norms.
                self._update_adaptive_weights(
                    model, samples, slowness_fn, slowness_grad_fn
                )
            # Single-medium curriculum: scale the global slowness fn
            sfn, sgfn = slowness_fn, slowness_grad_fn
            if curr_frac > 0.0 and slowness_spec is not None and self.family is None:
                sfn = slowness_spec.get_slowness_scaled(curr_scale)
                sgfn = slowness_spec.get_slowness_gradient_scaled(curr_scale)
            breakdown = self.loss.compute_loss(model, samples, sfn, sgfn, reference)
            breakdown.total.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.grad_clip
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            history.append(breakdown)

            if epoch % self.config.log_every == 0 or epoch == 1:
                print(
                    f"[Adam {epoch:>6d}/{self.config.epochs}]  loss={breakdown.total.item():.4e}  "
                    f"pde={breakdown.pde:.4e}  bc={breakdown.bc:.4e}  "
                    f"data={breakdown.data:.4e}"
                )

        return {}

    def _rar_select(self, model, slowness_fn, slowness_grad_fn):
        """Draw a candidate pool and keep the top-residual collocation points.

        Pool size = ``rar_pool_factor x collocation_samples``. The top
        ``collocation_samples`` by pointwise |PDE residual| are returned as a
        fresh :class:`Samples*`.
        """
        n = self.config.collocation_samples
        pool_n = n * int(getattr(self.config, "rar_pool_factor", 4))
        pool = Sampler.sample_interior(
            pool_n,
            self.size,
            self.device,
            mode=self.config.sample_mode,
            generator=self.sampling_rng,
            phi_delta=self.phi_delta,
            parametrization=self.parametrization,
            family=self.family,
        )
        score = self.loss.pointwise_pde_residual(
            model, pool, slowness_fn, slowness_grad_fn
        )
        idx = torch.topk(score, n).indices
        return _index_samples(pool, idx)

    def _update_adaptive_weights(self, model, samples, slowness_fn, slowness_grad_fn):
        """Grad-norm balance λ_pde/λ_bc. Updates them in place in the loss config.

        Measures ‖∇_θ L_term‖ per unweighted term and sets target weights
        ``λ̂_i = max_j‖∇L_j‖ / ‖∇L_i‖`` so each term contributes a comparable
        gradient. Uses EMA toward the targets to avoid thrashing.
        """
        terms = self.loss.loss_terms(model, samples, slowness_fn, slowness_grad_fn)
        params = [p for p in model.parameters() if p.requires_grad]
        norms = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(
                term, params, retain_graph=True, allow_unused=True
            )
            sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
            norms[name] = float(sq.sqrt())
        max_norm = max(norms.values())
        if max_norm <= 0.0:
            return
        alpha = 0.9  # EMA: keep 90% of the old weight, move 10% toward target
        cfg = self.loss.config
        for name, attr in (("pde", "lambda_pde"), ("bc", "lambda_bc")):
            target = max_norm / (norms[name] + 1e-12)
            old = float(getattr(cfg, attr))
            setattr(cfg, attr, alpha * old + (1.0 - alpha) * target)
