from typing import Optional

from pydantic import BaseModel
from torch import Tensor
from torch.nn import Module

TLayer = Module


class QuantizerOutput(BaseModel):
    codebook_vectors: Tensor
    loss: Optional[Tensor] = None
    codebook_indices: Optional[Tensor] = None
