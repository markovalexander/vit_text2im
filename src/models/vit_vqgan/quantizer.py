from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from src.data_types import QuantizerOutput


class NormalizeLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, dim=-1)


class PairwiseBatchedDistance(nn.Module):
    def __init__(self, compute_sqrt: bool = False):
        """
        Create a layer that computes pairwise L2 distances between two sets of vectors

        Args:
            compute_sqrt (bool, optional): Whether to return squared distance or the correct one. Defaults to False.
        """
        super().__init__()
        self.compute_sqrt = compute_sqrt

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): set of vectors of shape [B, D]
            y (Tensor): set of vectors of shape [N, D]

        Returns:
            Tensor: (squared) pairwise L2 distances of shape [B, N]
        """
        diff = x[:, None, :] - y[None, :, :]  # [B, N, D]
        diff = diff ** 2
        diff = torch.sum(diff, dim=-1)

        if self.compute_sqrt:
            diff = torch.sqrt(diff)

        return diff


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

        self.distance_calculator = PairwiseBatchedDistance()


    def quantize(self, z: Tensor) -> QuantizerOutput:
        z_reshaped = self.norm_layer(z.view(-1, self.codebook_dim))
        codebook = self.norm_layer(self.embedding.weight)

        distances = self.distance_calculator(z_reshaped, codebook)

        code_indices = torch.argmin(distances, dim=1)
        code_indices = code_indices.view(*z.size()[:-1])

        z_q = self.embedding(code_indices).view(z.size())
        z_qnorm, z_norm = self.norm_layer(z_q), self.norm_layer(z)

        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)

        return QuantizerOutput(codebook_vectors=z_qnorm, loss=loss, codebook_indices=code_indices)
