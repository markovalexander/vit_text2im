from itertools import chain
from typing import Optional

from einops import rearrange
from torch import Tensor, nn

from src.data_types import QuantizerOutput, StepType, ViTVQGANOutput
from src.models.vit_vqgan.discriminator import LossNetwork
from src.models.vit_vqgan.layers import ShiftedPatchTokenizationLayer, SinusoidalPosEmb, Transformer
from src.models.vit_vqgan.quantizer import VectorQuantizer
from src.params import LossSettings, VectorQuantizerSettings, ViTSettings


class ViTEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        image_size: int,
        input_channels: int = 3,
        num_layers: int = 4,
        patch_size: int= 16,
        heads_channels: int = 32,
        num_heads: int = 8,
        fc_multiplier: int = 4
    ):
        super().__init__()
        self.encoded_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        self.encoder = nn.Sequential(
            ShiftedPatchTokenizationLayer(
                out_dim=hidden_dim,
                image_size=image_size,
                patch_size=patch_size,
                channels=input_channels,
            ),
            Transformer(
                hidden_dim=hidden_dim,
                heads_channels=heads_channels,
                num_heads=num_heads,
                fc_multiplier=fc_multiplier,
                num_layers=num_layers,
                num_patches=self.num_patches,
            ),
        )

    def forward(self, x):
        return self.encoder(x)


class ViTDecoder(nn.Module):
    def __init__(
      self,
      hidden_dim: int,
      image_size: int,
      input_channels: int = 3,
      num_layers: int = 4,
      patch_size: int= 16,
      heads_channels: int = 32,
      num_heads: int = 8,
      fc_multiplier: int = 4
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        self.transformer = Transformer(
            hidden_dim=hidden_dim,
            heads_channels=heads_channels,
            num_heads=num_heads,
            fc_multiplier=fc_multiplier,
            num_layers=num_layers,
            num_patches=self.num_patches,
        )
        self.pre_out = nn.Sequential(
            SinusoidalPosEmb(hidden_dim // 2, height_or_width=self.num_patches),
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, 3, padding=1, bias=False),
            nn.Tanh(),
        )
        self.out = nn.Conv2d(4 * hidden_dim, input_channels * (patch_size ** 2), 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.transformer(x)
        x = self.out(self.pre_out(x))
        x = rearrange(
            x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size,
        )
        return x

    def last_layer(self):
        return self.out.weight

class ViTVQGAN(nn.Module):
    def __init__(
      self,
      vit_params: ViTSettings,
      vq_params: VectorQuantizerSettings,
      loss_params: LossSettings,
    ):
        super().__init__()
        self.encoder = ViTEncoder(**vit_params.encoder.dict())
        self.decoder = ViTDecoder(**vit_params.decoder.dict())

        self.quantizer = VectorQuantizer(**vq_params.dict())

        layer_mults = list(map(lambda t: 2 ** t, range(loss_params.discr_layers)))
        layer_dims = [vit_params.encoder.hidden_dim * mult for mult in layer_mults]
        dims = (vit_params.encoder.hidden_dim, *layer_dims)

        self.loss_net = LossNetwork(dims=dims, **loss_params.dict())

    def forward(self, images: Tensor, step: Optional[StepType] = None) -> ViTVQGANOutput:
        encoded = self.encoder(images)
        quantized: QuantizerOutput = self.quantizer(encoded)

        reconstructed = self.decoder(quantized.codebook_vectors)

        loss = None
        if step is not None:
            loss = self.loss_net(
                images, quantized.loss, reconstructed, step, self.decoder.last_layer(),
            )
        return ViTVQGANOutput(quantizer_output=quantized, loss=loss, reconstructed=reconstructed)

    def get_model_params(self):
        params = chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
        )
        return params

    def get_loss_params(self):
        return self.loss_net.parameters()
