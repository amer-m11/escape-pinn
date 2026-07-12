from typing import Literal, Optional

from pydantic import BaseModel

from .types import SampleMode


class OptimizerConfiguration(BaseModel):
    collocation_samples: int
    """Number of interior collocation points per epoch."""

    sample_mode: SampleMode
    """How to sample interior collocation points.

    - ``"random"``: i.i.d. uniform-random draws.
    - ``"grid"``: fixed deterministic lattice (not yet implemented).
    """
    resample: bool
    """Whether to resample collocation points every epoch. Standard is True.
    Relevant only if sample_mode='random'."""

    log_every: int
    """log training progress every N epochs."""


class AdamConfiguration(OptimizerConfiguration):
    epochs: int
    lr: float
    """Learning rate. Standard is 1e-3."""

    grad_clip: float = 0.0
    """Max global grad-norm for ``clip_grad_norm_`` during Adam. PDE residuals
    can spike early in training and blow a step up. Clipping caps that. ``0``
    (default) disables it."""

    rar: bool = False
    """Residual-adaptive resampling: every ``rar_every`` epochs, draw a pool of
    ``rar_pool_factor x collocation_samples`` candidates, score each by the
    pointwise |PDE residual|, and keep the top-``collocation_samples`` as the
    interior set. Concentrates collocation on the high-residual caustic. Default
    off. Independent of ``resample`` (which is uniform RNG resampling)."""

    rar_pool_factor: int = 4
    """Candidate-pool size as a multiple of ``collocation_samples`` for RAR."""

    rar_every: int = 1000
    """Re-run RAR selection every this many Adam epochs (the residual landscape
    drifts during training, so the hot region is periodically re-found)."""

    adaptive_weighting: bool = False
    """Grad-norm loss balancing (Wang et al. 2022): every ``adaptive_every``
    epochs, rescale λ_pde/λ_bc/λ_data so each term's parameter-gradient norm is
    comparable (λ_i <- Σ_j‖∇L_j‖ / ‖∇L_i‖, EMA-smoothed). Self-tunes the
    multi-objective trade-off. Default off -> fixed YAML weights."""

    adaptive_every: int = 1000
    """Re-estimate the adaptive loss weights every this many Adam epochs."""

    lr_decay: Optional[Literal["cosine"]] = None
    """Optional LR schedule over the Adam phase. ``"cosine"`` anneals the LR
    from ``lr`` to ``lr_decay_floor·lr`` over ``epochs`` (CosineAnnealingLR) —
    a cheap end-of-phase polish that needs no extra epochs. Default ``None``
    (constant LR, the historical behavior)."""

    lr_decay_floor: float = 0.01
    """Final LR as a fraction of the initial ``lr`` for ``lr_decay``."""

    lens_curriculum: float = 0.0
    """lens-amplitude curriculum. The slowness *perturbation* is ramped
    from 0 (constant medium, where the wall-grazing step resolves to L∞~0.003) to full
    over the first ``lens_curriculum`` fraction of Adam epochs (e.g. 0.5), so the step
    resolves first and the caustic is learned after without losing grazing resolution.
    ``0.0`` (default) = off (full medium throughout)."""


class LbfgsConfiguration(OptimizerConfiguration):
    steps: int
    lr: float
    """Learning rate. Standard is 1.0."""

    max_iter: int
    """inner line-search evaluations per outer step. Standard is 20."""
    history_size: int
    """Number of (s, y) curvature pairs stored. Standard is 50."""
    line_search_fn: Literal["strong_wolfe"]
    """Line search function for L-BFGS. Standard is 'strong_wolfe'."""

    resample_per_step: bool = False
    """Resample collocation once at the start of each outer step, keeping
    it fixed during that step's strong-Wolfe line search. When ``True``, 
    ``resample`` is ignored (the closure never resamples). Standard is ``False``."""


class OptimizersConfiguration(BaseModel):
    adam: AdamConfiguration
    lbfgs: Optional[LbfgsConfiguration] = None
    """Optional L-BFGS fine-tuning phase. When ``None`` only Adam runs."""
