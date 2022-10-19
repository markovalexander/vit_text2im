from torch import Tensor, nn

from src.data_types import QuantizerOutput, StepType
from src.models.vit_vqgan.layers import ViTDecoder, ViTEncoder
from src.models.vit_vqgan.loss import VQLPIPSWithDiscriminator
from src.models.vit_vqgan.quantizer import VectorQuantizer


class ViT_VQGAN(nn.Module):
    def __init__(self, encoder_params, decoder_params, quantizer_params, loss_params):
        super().__init__()

        self.encoder = ViTEncoder(**encoder_params)
        self.decoder = ViTDecoder(**decoder_params)
        self.quantizer = VectorQuantizer(**quantizer_params)

        self.loss = VQLPIPSWithDiscriminator(**loss_params)

        self.pre_quant = nn.Linear(encoder_params.hidden_dim, quantizer_params.hidden_dim)
        self.post_quant = nn.Linear(quantizer_params.hidden_dim, decoder_params.hidden_dim)

    def forward(self, x: Tensor, step_type: StepType, global_step: int, batch_idx: int) -> Tensor:
        encoded = self.encode(x)

        encoded_vectors = encoded.codebook_vectors
        quantizer_loss = encoded.loss

        reconstructed = self.decode(encoded_vectors)

        if step_type == StepType.AUTOENCODER:
            loss = self.loss(quantizer_loss, x, reconstructed, int(step_type), global_step, batch_idx,
                               last_layer=self.decoder.get_last_layer(), split="train")
        else:
            loss = self.loss(quantizer_loss, x, reconstructed, int(step_type), global_step, batch_idx,
                                 last_layer=self.decoder.get_last_layer(), split="train")

        return loss

    def encode(self, x: Tensor) -> QuantizerOutput:
        x = self.encoder(x)
        x = self.pre_quant(x)
        return self.quantizer(x)

    def decode(self, z: Tensor) -> Tensor:
        z = self.post_quant(z)
        return self.decoder(z)
