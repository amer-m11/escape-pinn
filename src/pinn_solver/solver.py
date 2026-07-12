"""Top-level orchestrator for a PINN training run.

Wires the discrete pieces together from a single :class:`TrainConfiguration`:

  - resolves the device (CPU/CUDA)
  - resolves the seed (drawing one at random if ``cfg.seed is None`` so the
    run is still replayable via ``run.resolved_seed`` in the metrics JSON)
  - seeds torch's default RNG (covers weight init) and builds a dedicated
    :class:`torch.Generator` for the interior collocation sampler so its
    sequence does not depend on other consumers of the default RNG
  - optionally enables :func:`torch.use_deterministic_algorithms` when
    ``cfg.deterministic=True``
  - constructs the network via :class:`NetworkFactory`
  - builds the shared :class:`Loss` (pre-samples the boundary set)
  - constructs Adam and L-BFGS trainers wired with the sampling generator
  - drives both phases on a shared :class:`TrainHistory`

The training data sources (slowness fn + gradient) are derived from
``config.slowness``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from .configuration import TrainConfiguration
from .data import ReferenceSolution
from .device import get_device
from .loss import Loss
from .networks import EscapeNet, NetworkFactory
from .trainers import AdamTrainer, LbfgsTrainer, TrainHistory

_SEED_MIN = 0
_SEED_MAX = (1 << 63) - 1


@dataclass
class TrainingResult:
    """Everything a caller needs after :meth:`PINNRunner.run`.

    ``phase_stats`` is keyed by phase (``"adam"``, ``"lbfgs"``, ``"total"``)
    and always carries ``time_s``. The L-BFGS entry additionally carries
    ``closure_calls_per_step`` (see :meth:`LbfgsTrainer.train`).
    """

    model: EscapeNet
    history: TrainHistory
    phase_stats: dict[str, dict] = field(default_factory=dict)


class PINNRunner:
    """Build + drive one PINN training run from a :class:`TrainConfiguration`."""

    def __init__(self, config: TrainConfiguration):
        self.config = config
        self.device = get_device(config.device)

        # Seed before network construction. Covers torch's default RNG.
        self.resolved_seed = (
            int(config.seed)
            if config.seed is not None
            else int(torch.randint(_SEED_MIN, _SEED_MAX, (1,)).item())
        )
        torch.manual_seed(self.resolved_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.resolved_seed)

        # cuDNN/cuBLAS determinism (off by default).
        self.deterministic = bool(config.deterministic)
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Decoupled RNG for collocation sampling.
        self.sampling_rng = torch.Generator(device=self.device)
        self.sampling_rng.manual_seed(self.resolved_seed)

        self.slowness_fn = config.slowness.get_slowness()
        self.slowness_grad_fn = config.slowness.get_slowness_gradient()

        # Slowness-conditioning (generalization phase): when the medium is a
        # parametrized family, the net is conditioned on a per-point descriptor.
        # ``self.slowness_fn`` above is the mean-descriptor fallback. The Loss
        # rebinds it per batch from each point's sampled params.
        from .configuration.slowness import _ConditionedFamily

        self._family = (
            config.slowness
            if isinstance(config.slowness, _ConditionedFamily)
            else None
        )
        if self._family is not None:
            cd = self._family.cond_dim  # param_dim (Stage 1) or n_probe^d (Stage 2)
            if config.net.conditioning_dim == 0:
                config.net.conditioning_dim = cd  # auto-match the conditioning width
            elif config.net.conditioning_dim != cd:
                raise ValueError(
                    f"net.conditioning_dim ({config.net.conditioning_dim}) must "
                    f"equal the family conditioning width ({cd})."
                )
            # Fail fast on lever combos whose model/slowness calls do not thread the
            # conditioning vector
            _unsupported = []
            if config.loss.lambda_data > 0:
                _unsupported.append("loss.lambda_data>0 / supervised reference "
                                    "(the reference path is single-medium)")
            if getattr(config.loss, "lambda_consistency", 0.0) > 0:
                _unsupported.append("loss.lambda_consistency>0 (consistency residual)")
            if getattr(config.net, "selector_head", False):
                _unsupported.append("net.selector_head")
            if getattr(config.net, "domain_decomp", None) is not None:
                _unsupported.append("net.domain_decomp")
            if _unsupported:
                raise ValueError(
                    "These levers do not yet support slowness-conditioning (their "
                    "model/slowness calls omit `cond`): " + ", ".join(_unsupported)
                    + ". Disable them or thread `cond` through their call sites first."
                )

        self.network = NetworkFactory.create_network(
            config.net, config.physical_size, config.parametrization,
        )
        self.network.to(self.device)

        self.loss = Loss(
            config.loss,
            config.boundary,
            config.physical_size,
            self.device,
            sampling_rng=self.sampling_rng,
            parametrization=config.parametrization,
            family=self._family,
        )
        # Single source of truth for the 3D φ safe-band, shared by the
        # interior sampler (trainers), the boundary sampler (Loss), and the
        # metric φ-sweep so all three agree on the trained band. Ignored in
        # the Cartesian path. The parametrization selects the interior sampler
        # (spherical (φ,θ) vs Cartesian d-on-S²).
        phi_delta = config.boundary.resolved_phi_delta()
        param = config.parametrization
        self.adam_trainer = AdamTrainer(
            config.optimizers.adam,
            self.loss,
            config.physical_size,
            self.device,
            sampling_rng=self.sampling_rng,
            phi_delta=phi_delta,
            parametrization=param,
            family=self._family,
        )
        self.lbfgs_trainer = (
            LbfgsTrainer(
                config.optimizers.lbfgs,
                self.loss,
                config.physical_size,
                self.device,
                sampling_rng=self.sampling_rng,
                phi_delta=phi_delta,
                parametrization=param,
                family=self._family,
            )
            if config.optimizers.lbfgs is not None
            else None
        )

    def run(self, reference: ReferenceSolution | None = None) -> TrainingResult:
        """Run the configured Adam + (optional) L-BFGS phases.

        Returns
        -------
        TrainingResult
            ``model`` is the trained ``self.network`` (same instance);
            ``history`` is the per-step :class:`TrainHistory` accumulated
            across both phases.
            ``phase_stats`` carries per-phase wall time and L-BFGS closure-call counts.
        """
        history = TrainHistory()
        phase_stats: dict[str, dict] = {}

        t_total_start = self._now()

        print("=== Phase 1: Adam ===")
        t0 = self._now()
        adam_extras = self.adam_trainer.train(
            self.network,
            history,
            self.slowness_fn,
            self.slowness_grad_fn,
            reference,
            slowness_spec=self.config.slowness,
        )
        phase_stats["adam"] = {"time_s": self._now() - t0, **adam_extras}

        if self.lbfgs_trainer is not None:
            print("=== Phase 2: L-BFGS ===")
            t0 = self._now()
            lbfgs_extras = self.lbfgs_trainer.train(
                self.network,
                history,
                self.slowness_fn,
                self.slowness_grad_fn,
                reference,
            )
            phase_stats["lbfgs"] = {"time_s": self._now() - t0, **lbfgs_extras}

        phase_stats["total"] = {"time_s": self._now() - t_total_start}

        return TrainingResult(
            model=self.network, history=history, phase_stats=phase_stats
        )

    def _now(self) -> float:
        """Wall-clock time after a CUDA sync so async kernels are accounted for."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.time()
