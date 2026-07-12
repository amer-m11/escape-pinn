from dataclasses import dataclass, field

from ..loss import LossBreakdown


@dataclass
class TrainHistory:
    """Training history of a PINN solver run.

    Stores the loss breakdowns for each step, and the per-channel BC components
    """

    loss: list[float] = field(default_factory=list)
    loss_pde: list[float] = field(default_factory=list)
    loss_bc: list[float] = field(default_factory=list)
    loss_data: list[float] = field(default_factory=list)
    loss_circle: list[float] = field(default_factory=list)
    loss_consistency: list[float] = field(default_factory=list)
    bc_components: list[dict[str, float]] = field(default_factory=list)
    """Per channel BC component time series. One dict per step."""

    def append(self, breakdown: LossBreakdown):
        self.loss.append(breakdown.total.item())
        self.loss_pde.append(breakdown.pde)
        self.loss_bc.append(breakdown.bc)
        self.loss_data.append(breakdown.data)
        self.loss_circle.append(breakdown.circle)
        self.loss_consistency.append(getattr(breakdown, "consistency", 0.0))
        if getattr(breakdown, "bc_components", None) is not None:
            self.bc_components.append(breakdown.bc_components)
