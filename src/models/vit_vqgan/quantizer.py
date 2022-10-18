from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from src.data_types import QuantizerOutput
from src.models.vit_vqgan.layers import NormalizeLayer


class BaseQuantizer(nn.Module, ABC):
    def __init__(
      self,
      codebook_dim: int,
      codebook_size: int,
      use_norm: bool = True,
      use_straight_through: bool = True,
    ):
        super().__init__()

        self.use_straight_through = use_straight_through
        self.norm_layer = NormalizeLayer() if use_norm else nn.Identity()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.normal_()


    @abstractmethod
    def quantize(self, z: Tensor) -> QuantizerOutput:
        raise NotImplementedError("Your Qunatizer must implement quantize() method!")

    def forward(self, x: Tensor) -> QuantizerOutput:
        quantized = self.quantize(x)
        z_q = quantized.codebook_vectors
        loss = quantized.loss
        indices = quantized.codebook_indices

        if self.use_straight_through:
            z_q = x + (z_q - x).detach()

        return QuantizerOutput(codebook_vectors=z_q, loss=loss, codebook_indices=indices)


class VectorQuantizer(BaseQuantizer):
    def __init__(
      self,
      codebook_dim: int,
      codebook_size: int,
      use_norm: bool = True,
      use_straight_through: bool = True,
      beta: float = 0.25,
    ):
        super().__init__(codebook_dim, codebook_size, use_norm, use_straight_through)
        self.beta = beta

        self.distance_calculator = nn.PairwiseDistance(p=2)


    def quantize(self, z: Tensor) -> QuantizerOutput:
        z_reshaped = self.norm(z.view(-1, self.codebook_dim))
        codebook = self.norm(self.embedding.weight)

        distances = self.distance_calculator(z_reshaped, codebook)
        code_indices = torch.argmin(distances, dim=1)
        code_indices = code_indices.view(*z.size()[:-1])

        z_q = self.embedding(code_indices).view(z.size())
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)

        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)

        return QuantizerOutput(codebook_vectors=z_qnorm, loss=loss, codebook_indices=code_indices)
