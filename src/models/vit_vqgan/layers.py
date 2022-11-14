import math
from typing import List, Optional

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn.functional import pad


class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        kernel_sizes: List,
        dim_out: Optional[int] = None,
        stride: int = 2
    ):
        assert kernel_sizes is not None
        assert all(kernel_size % 2 == stride % 2 for kernel_size in kernel_sizes)

        super().__init__()

        dim_out = dim_out if dim_out else dim_in

        kernel_sizes = sorted(kernel_sizes)
        dim_scales = [dim_out // (2 ** i) for i in range(1, len(kernel_sizes))]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride)//2),
            )

    def forward(self, x: Tensor) -> Tensor:
        feature_maps = [conv(x) for conv in self.convs]
        return torch.cat(feature_maps, dim = 1)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        num_groups = 8
    ):
        super().__init__()
        dim_out = dim_out or dim_in

        self.transform = nn.Sequential(
            nn.GroupNorm(num_groups, dim_in),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_in, dim_out, 3, padding=1)
        )
        self.skip_connection = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x) + self.skip_connection(x)


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

class RelevancePositionEmbeddingsLayer(nn.Module):
    def __init__(self, size: int, heads: int):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        positions = torch.stack(
            torch.meshgrid(
                torch.arange(size), torch.arange(size), indexing = 'ij'
            ), dim = -1,
        )

        positions = rearrange(positions, '... c -> (...) c')
        relevance_positions = positions[:, None, :] - positions[None, :, :]

        relevance_positions = relevance_positions + size - 1
        h_rel, w_rel = relevance_positions.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel

        self.register_buffer('pos_indices', pos_indices)

    def forward(self, *args, **kwargs) -> Tensor:
        rel_pos_bias = self.pos_bias(self.pos_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> h i j')
        return rel_pos_bias

class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        height_or_width: int,
        theta: int = 10000,
    ):
        super().__init__()
        self.dim = hidden_dim
        self.theta = theta

        hw_range = torch.arange(height_or_width)
        coors = torch.stack(torch.meshgrid(hw_range, hw_range, indexing='ij'), dim=-1)
        coors = coors.contiguous()
        self.register_buffer('coors', coors, persistent = False)

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(self.coors, 'h w c -> h w c 1') * rearrange(emb, 'j -> 1 1 1 j')

        fourier = torch.cat((emb.sin(), emb.cos()), dim=-1)
        fourier = repeat(fourier, 'h w c d -> b (c d) h w', b=x.shape[0])

        concat = torch.cat((x, fourier), dim = 1)
        return concat


class LayerNormScaleOnlyLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = hidden_dim
        self.scale = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))
        self.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim=1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.scale


class ShiftedPatchTokenizationLayer(nn.Module):
    def __init__(self, out_dim: int, image_size: int, patch_size: int, channels: int = 3):
        assert image_size % patch_size == 0
        super().__init__()

        self.patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            LayerNormScaleOnlyLayer(self.patch_dim),
            nn.Conv2d(self.patch_dim, out_dim, 1)
        )
        self.shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))

    def forward(self, x: Tensor) -> Tensor:
        shifted_x = [pad(x, shift) for shift in self.shifts]
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        x = self.to_patch_tokens(x_with_shifts)
        return x


class InnerConv2d(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels, num_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=num_channels, stride=1,
        )

    def forward(self, x):
        return self.proj(x)


class Attention2d(nn.Module):
    def __init__(
      self,
      hidden_dim: int,
      num_heads: int = 8,
      head_channels: int = 64,
      num_patches: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.norm = LayerNormScaleOnlyLayer(hidden_dim)

        inner_dim = head_channels *  num_heads
        project_out = not (num_heads == 1 and head_channels == hidden_dim)

        self.num_heads = num_heads
        self.scale = head_channels ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Conv2d(hidden_dim, inner_dim * 3, kernel_size=1, bias=False)

        self.inner_convs = nn.ModuleList(InnerConv2d(inner_dim) for _ in range(3))

        self.to_out = nn.Conv2d(inner_dim, hidden_dim, kernel_size=1, bias=False) if project_out else nn.Identity()

        self.rel_pos_embedding = None
        if num_patches:
            self.rel_pos_embedding = RelevancePositionEmbeddingsLayer(size=num_patches, heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        image_size = x.size(-1)

        x = self.norm(x)
        print('after_norm:', x[0, 0, 0, 0:4])

        qkv = self.to_qkv(x).chunk(3, dim=1)
        print('after q, k, v', qkv[0][0, 0, 0, 0:4])

        qkv = (inner_conv(t) for t, inner_conv in zip(qkv, self.inner_convs))

        q, k, v = [rearrange(t, 'b (h d) x y -> b h (x y) d', h = self.num_heads) for t in qkv]
        q = q * self.scale

        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        if self.rel_pos_embedding:
            attn = attn + self.rel_pos_embedding(attn)

        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=image_size, y=image_size)
        return self.to_out(out)


class TransformerLayer(nn.Module):
    def __init__(
      self,
      hidden_dim: int,
      num_heads: int = 8,
      heads_channels: int = 64,
      fc_multiplier: int = 4,
      pre_transform_kernel_size: int = 3,
      num_patches: Optional[int] = None,
    ):
        super().__init__()
        self.pre_transform = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            pre_transform_kernel_size,
            stride=1,
            groups=hidden_dim,
            padding=pre_transform_kernel_size // 2,
        )

        self.attention = Attention2d(hidden_dim, num_heads, heads_channels, num_patches)

        self.fc = nn.Sequential(
            LayerNormScaleOnlyLayer(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim * fc_multiplier, 1, bias=False),
            nn.GELU(),
            InnerConv2d(hidden_dim * fc_multiplier),
            nn.Conv2d(hidden_dim * fc_multiplier, hidden_dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre_transform(x) + x
        x = self.attention(x) + x
        x = self.fc(x) + x
        return x

class Transformer(nn.Module):
    def __init__(
      self,
      hidden_dim: int,
      num_layers: int = 6,
      num_heads: int = 8,
      heads_channels: int = 64,
      fc_multiplier: int = 4,
      pre_transform_kernel_size: int = 3,
      num_patches: Optional[int] = None,
    ):
        super().__init__()
        layers = [TransformerLayer(
            hidden_dim, num_heads, heads_channels, fc_multiplier, pre_transform_kernel_size, num_patches,
        ) for _ in range(num_layers)]

        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormScaleOnlyLayer(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
