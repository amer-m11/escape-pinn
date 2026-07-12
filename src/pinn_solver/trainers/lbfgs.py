import math

import torch

from ..configuration import LbfgsConfiguration, PhysicalSize, SlownessFn, SlownessGradFn
from ..data import ReferenceSolution
from ..loss import Loss, LossBreakdown
from ..networks import EscapeNet
from ..sampler import Sampler
from .history import TrainHistory


class LbfgsTrainer:
    config: LbfgsConfiguration
    loss: Loss
    size: PhysicalSize
    device: torch.device
    sampling_rng: torch.Generator | None
    """Random number generator for sampling. When ``None`` the global
    default RNG is used (seeded by `torch.manual_seed` or similar)."""
    phi_delta: float
    """φ safe-band half-width for 3D interior sampling (ignored in 2D)."""
    parametrization: str
    """3D momentum parametrization ('spherical' | 'cartesian'); ignored in 2D."""

    def __init__(
        self,
        config: LbfgsConfiguration,
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
    ) -> dict:
        """Run the L-BFGS phase. Returns a stats dict for diagnostics:
        {
            "closure_calls_per_step": list[int] # number of closure calls per outer step
        }
        """
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=self.config.lr,
            max_iter=self.config.max_iter,  # inner line-search evaluations per outer step
            history_size=self.config.history_size,  # number of (s, y) curvature pairs stored
            line_search_fn=self.config.line_search_fn,
        )

        n = self.config.collocation_samples
        sample_mode = self.config.sample_mode

        model.train()

        # Single-cells so the closure access them.
        current = [
            Sampler.sample_interior(
                n,
                self.size,
                self.device,
                mode=sample_mode,
                generator=self.sampling_rng,
                phi_delta=self.phi_delta,
                parametrization=self.parametrization,
                family=self.family,
            )
        ]
        last: list[LossBreakdown | None] = [None]
        call_counter = [0]
        closure_calls_per_step: list[int] = []

        resample_per_step = getattr(self.config, "resample_per_step", False)

        for step in range(1, self.config.steps + 1):

            # Refresh collocation once per outer step, fixed during the step's
            # line search. Mutually exclusive with the per-closure-call ``resample`` below.
            if resample_per_step:
                current[0] = Sampler.sample_interior(
                    n,
                    self.size,
                    self.device,
                    mode=sample_mode,
                    generator=self.sampling_rng,
                    phi_delta=self.phi_delta,
                    parametrization=self.parametrization,
                    family=self.family,
                )

            def closure():
                call_counter[0] += 1
                optimizer.zero_grad()
                if self.config.resample and not resample_per_step:
                    current[0] = Sampler.sample_interior(
                        n,
                        self.size,
                        self.device,
                        mode=sample_mode,
                        generator=self.sampling_rng,
                        phi_delta=self.phi_delta,
                        parametrization=self.parametrization,
                        family=self.family,
                    )
                breakdown = self.loss.compute_loss(
                    model, current[0], slowness_fn, slowness_grad_fn, reference
                )
                total = breakdown.total
                # NaN safety-net: if a trial step lands in a non-finite region,
                # don't backward (would set NaN grads -> NaN params, unrecoverable).
                # return a large finite loss so the strong-Wolfe line search rejects
                # this step and backs off. Keeps last[0] at the last good breakdown.
                if not torch.isfinite(total):
                    optimizer.zero_grad()
                    return torch.full_like(total.detach(), 1e8)
                total.backward()
                last[0] = breakdown
                return total

            call_counter[0] = 0
            optimizer.step(closure)
            closure_calls_per_step.append(call_counter[0])

            assert last[0] is not None, "L-BFGS closure was never invoked"
            breakdown = last[0]
            history.append(breakdown)

            if step % self.config.log_every == 0 or step == 1:
                print(
                    f"[L-BFGS {step:>4d}/{self.config.steps}]  loss={breakdown.total.item():.4e}  "
                    f"pde={breakdown.pde:.4e}  bc={breakdown.bc:.4e}  "
                    f"data={breakdown.data:.4e}  "
                    f"closure_calls={call_counter[0]}"
                )

        return {"closure_calls_per_step": closure_calls_per_step}
