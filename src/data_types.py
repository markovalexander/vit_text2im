from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch import Tensor
from torch.nn import Module

TLayer = Module


@dataclass
class QuantizerOutput:
    codebook_vectors: Tensor
    loss: Optional[Tensor] = None
    codebook_indices: Optional[Tensor] = None


class StepType(int, Enum):
    AUTOENCODER: int = 0
    DISCRIMINATOR: int = 1

    def __str__(self):
        return self.value

    @classmethod
    def from_global_step(cls, global_step: int) -> 'StepType':
        if global_step % 2 == 0:
            return StepType.AUTOENCODER
        return StepType.DISCRIMINATOR
