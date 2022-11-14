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

@dataclass
class ViTVQGANOutput:
    quantizer_output: QuantizerOutput
    loss: Tensor
    reconstructed: Tensor

class StepType(int, Enum):
    DISCRIMINATOR = 0
    MODEL = 1

    def __int__(self):
        return self.value

    @classmethod
    def from_global_step(cls, step: int):
        if step % 2:
            return StepType.MODEL
        return StepType.DISCRIMINATOR
