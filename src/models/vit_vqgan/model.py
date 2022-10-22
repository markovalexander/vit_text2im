from itertools import chain
from typing import Optional

from torch import Tensor, nn

from src.data_types import QuantizerOutput, StepType, TLayer, ViTVQGANOutput
from src.models.vit_vqgan.layers import ViTDecoder, ViTEncoder
from src.models.vit_vqgan.loss import VQLPIPSWithDiscriminator
from src.models.vit_vqgan.quantizer import VectorQuantizer
from src.params import LossSettings, ModelSettings, VectorQuantizerSettings


class ViT_VQGAN(nn.Module):
    def __init__(
      self,
      encoder_params: ModelSettings,
      decoder_params: ModelSettings,
      quantizer_params: VectorQuantizerSettings,
      loss_params: LossSettings
    ):
        super().__init__()

        self.loss_net = LossNetwork(loss_params)
        self.encoder = ViTEncoder(**encoder_params.dict())
        self.decoder = ViTDecoder(**decoder_params.dict())
        self.quantizer = VectorQuantizer(**quantizer_params.dict())

        self.pre_quant = nn.Linear(encoder_params.hidden_dim, quantizer_params.codebook_dim)
        self.post_quant = nn.Linear(quantizer_params.codebook_dim, decoder_params.hidden_dim)

    def forward(
        self,
        x: Tensor,
        step_type: Optional[StepType] = None,
        global_step: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ) -> ViTVQGANOutput:
        encoded = self.encode(x)

        encoded_vectors = encoded.codebook_vectors
        quantizer_loss = encoded.loss

        reconstructed = self.decode(encoded_vectors)

        if step_type is not None and global_step is not None and batch_idx is not None:
            loss = self.loss_net(
                quantizer_loss,
                x,
                reconstructed,
                step_type,
                global_step,
                batch_idx,
                self.decoder.get_last_layer(),
            )
        else:
            loss = None

        return ViTVQGANOutput(encoded_vectors, quantizer_loss, reconstructed, loss)

    def encode(self, x: Tensor) -> QuantizerOutput:
        z = self.encoder(x)
        z = self.pre_quant(z)
        return self.quantizer(z)

    def decode(self, z: Tensor) -> Tensor:
        x = self.post_quant(z)
        return self.decoder(x)

    def get_train_params(self):
        params = chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.quantizer.parameters(),
            self.pre_quant.parameters(),
            self.post_quant.parameters(),
        )
        return params

    def get_loss_params(self):
        return self.loss_net.parameters()


class LossNetwork(nn.Module):
    def __init__(self, loss_params: LossSettings):
        super().__init__()

        self.loss_net = VQLPIPSWithDiscriminator(**loss_params.dict())

    def forward(
      self,
      quantizer_loss: Tensor,
      x: Tensor,
      reconstructed: Tensor,
      step_type: StepType,
      global_step: int,
      batch_idx: int,
      last_layer: TLayer,
    ) -> Tensor:

        if step_type == StepType.AUTOENCODER:
            loss = self.loss_net(
                quantizer_loss, x, reconstructed, step_type, global_step, batch_idx, last_layer,
            )
        else:
            loss = self.loss_net(
                quantizer_loss, x, reconstructed, step_type, global_step, batch_idx, last_layer,
            )

        return loss
