from torch import Tensor, nn

from src.data_types import QuantizerOutput, StepType
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
      loss_params: LossSettings,
    ):
        super().__init__()

        self.encoder = ViTEncoder(**encoder_params.dict())
        self.decoder = ViTDecoder(**decoder_params.dict())
        self.quantizer = VectorQuantizer(**quantizer_params.dict())

        self.loss = VQLPIPSWithDiscriminator(**loss_params.dict())

        self.pre_quant = nn.Linear(encoder_params.hidden_dim, quantizer_params.codebook_dim)
        self.post_quant = nn.Linear(quantizer_params.codebook_dim, decoder_params.hidden_dim)

    def forward(self, x: Tensor, step_type: StepType, global_step: int, batch_idx: int) -> Tensor:
        encoded = self.encode(x)

        encoded_vectors = encoded.codebook_vectors
        quantizer_loss = encoded.loss

        reconstructed = self.decode(encoded_vectors)

        if step_type == StepType.AUTOENCODER:
            loss = self.loss(
                quantizer_loss, x, reconstructed, step_type, global_step, batch_idx, self.decoder.get_last_layer(),
            )
        else:
            loss = self.loss(
                quantizer_loss, x, reconstructed, step_type, global_step, batch_idx, self.decoder.get_last_layer(),
            )

        return loss

    def encode(self, x: Tensor) -> QuantizerOutput:
        z = self.encoder(x)
        z = self.pre_quant(z)
        return self.quantizer(z)

    def decode(self, z: Tensor) -> Tensor:
        x = self.post_quant(z)
        return self.decoder(x)
