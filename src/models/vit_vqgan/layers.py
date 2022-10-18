import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from src.data_types import TLayer
from src.utils.position_embeddings import get_2d_sincos_pos_embed


class NormalizeLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1)


class Transformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, mlp_hidden_dim, dropout: float = 0):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            dim, num_heads, mlp_hidden_dim, dropout=dropout, activation=nn.Tanh(),
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(x)


class ViTEncoder(nn.Module):
    def __init__(
      self,
      image_size: int,
      patch_size: int,
      hidden_dim: int,
      num_layers: int,
      num_heads: int,
      mlp_dim: int,
      channels: int = 3,
      ):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        pos_embed = get_2d_sincos_pos_embed(hidden_dim, (image_height // patch_height, image_width // patch_width))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)

        self.to_patch = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.transformer = Transformer(hidden_dim, num_layers, num_heads, mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_patch(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(
      self,
      image_size: int,
      patch_size: int,
      hidden_dim: int,
      num_layers: int,
      num_heads: int,
      mlp_dim: int,
      channels: int = 3,
      ):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        pos_embed = get_2d_sincos_pos_embed(hidden_dim, (image_height // patch_height, image_width // patch_width))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)

        self.to_patch = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.transformer = Transformer(hidden_dim, num_layers, num_heads, mlp_dim)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height, w=image_width // patch_width),
            nn.ConvTranspose2d(hidden_dim, channels, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.to_pixel(x)
        return x
