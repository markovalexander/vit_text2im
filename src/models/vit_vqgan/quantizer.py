from abc import ABC, abstractmethod

import torch
from einops import rearrange
from torch import Tensor, nn

from src.data_types import QuantizerOutput
from src.models.vit_vqgan.layers import NormalizeLayer, PairwiseBatchedDistanceLayer


class BaseQuantizer(nn.Module, ABC):
    def __init__(
      self,
      input_dim: int,
      codebook_dim: int,
      codebook_size: int,
      use_norm: bool = True,
      use_straight_through: bool = True,
      encode_images: bool = False
    ):
        super().__init__()

        self.pre_codebook = nn.Linear(input_dim, codebook_dim)
        self.out_codebook = nn.Linear(codebook_dim, input_dim)

        self.use_straight_through = use_straight_through
        self.norm_layer = NormalizeLayer() if use_norm else nn.Identity()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.normal_()

        self.encode_images = encode_images


    @abstractmethod
    def quantize(self, z: Tensor) -> QuantizerOutput:
        raise NotImplementedError("Your Qunatizer must implement quantize() method!")

    def forward(self, x: Tensor) -> QuantizerOutput:
        if self.encode_images:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.pre_codebook(x)
        quantized = self.quantize(x)

        z_q = quantized.codebook_vectors
        loss = quantized.loss
        indices = quantized.codebook_indices

        if self.use_straight_through:
            z_q = x + (z_q - x).detach()

        z_q = self.out_codebook(z_q)
        if self.encode_images:
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=height, w=width)
            indices = rearrange(indices, 'b (h w) ... -> b h w ...', h=height, w=width)

        return QuantizerOutput(codebook_vectors=z_q, loss=loss, codebook_indices=indices)


class VectorQuantizer(BaseQuantizer):
    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        codebook_size: int,
        use_norm: bool = True,
        use_straight_through: bool = True,
        encode_images: bool = False,
        beta: float = 0.25,
    ):
        super().__init__(
            input_dim,
            codebook_dim,
            codebook_size,
            use_norm,
            use_straight_through,
            encode_images,
        )
        self.beta = beta

        self.distance_calculator = PairwiseBatchedDistanceLayer()


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
