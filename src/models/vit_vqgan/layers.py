import torch.nn.functional as F
from torch import Tensor, nn

from src.data_types import TLayer


class NormalizeLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1)


class NormAndTransform(nn.Module):
    def __init__(self, dim: int, layer: TLayer):
        super().__init__()

        self.layer = layer
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x:Tensor, **kwargs) -> Tensor:
        return self.layer(self.layer_norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)


class Transformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, mlp_hidden_dim, dropout: float = 0):
        super().__init__()
        self.layers  = []
        for _ in range(num_layers):
            transformer_layer = nn.ModuleList([
                NormAndTransform(dim, nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)),
                NormAndTransform(dim, FeedForward(dim, mlp_hidden_dim)),
            ])
            self.layers.append(transformer_layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x: Tensor) -> Tensor:
        for attention_layer, feed_forward_layer in self.layers:
            x = attention_layer(x) + x
            x = feed_forward_layer(x) + x

        return x
