from typing import Callable, Literal, Optional

import torch
from pydantic import BaseModel

SampleMode = Literal["grid", "random"]
LossType = Literal["mse"]

# Slowness callables take a single batched spatial tensor of shape (N, d) where
# d ∈ {2, 3} and return a (N,) value tensor or a (N, d) gradient tensor. The
# dim-agnostic signature lets the same machinery serve 2D and 3D physical
# domains.
SlownessFn = Callable[[torch.Tensor], torch.Tensor]
SlownessGradFn = Callable[[torch.Tensor], torch.Tensor]


class PhysicalSize(BaseModel):
    """Physical-domain extents.

    Supplying ``L_x3`` switches the configuration into 3D mode (5D phase
    space with the second polar angle φ). 2D YAML files load unchanged
    because ``L_x3`` defaults to ``None``.
    """

    L_x1: float
    L_x2: float
    L_x3: Optional[float] = None

    @property
    def dim(self) -> int:
        return 3 if self.L_x3 is not None else 2

    def as_tuple(self) -> tuple[float, ...]:
        if self.L_x3 is None:
            return (self.L_x1, self.L_x2)
        return (self.L_x1, self.L_x2, self.L_x3)
